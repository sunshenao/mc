// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <glog/logging.h>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <ctype.h>
#include <dirent.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "topology.h"

namespace mooncake {

/**
 * @struct InfinibandDevice
 * @brief InfiniBand网卡设备信息结构
 *
 * 记录了IB网卡的基本信息：
 * - 设备名称（如mlx5_0）
 * - PCI总线ID
 * - NUMA节点编号
 */
struct InfinibandDevice {
    std::string name;         // 设备名称
    std::string pci_bus_id;   // PCI总线ID
//    这里的“PCI总线ID”指的是该InfiniBand网卡在主机PCI总线上的唯一标识符。它通常表示为域:总线:设备.功能（如0000:3b:00.0）
    int numa_node;           // NUMA节点编号
};

/**
 * @brief 列出系统中所有的InfiniBand设备
 * @return InfiniBand设备列表
 *
 * 该函数通过扫描sysfs获取IB设备信息：
 * 1. 遍历/sys/class/infiniband目录
 * 2. 解析每个设备的PCI信息
 * 3. 获取设备的NUMA节点信息
 */
static std::vector<InfinibandDevice> listInfiniBandDevices() {
    // 打开IB设备目录
    DIR *dir = opendir("/sys/class/infiniband");
    struct dirent *entry;
    std::vector<InfinibandDevice> devices;

    if (dir == NULL) {
        PLOG(WARNING) << "Failed to open /sys/class/infiniband";
        return {};
    }

    // 遍历所有IB设备
    while ((entry = readdir(dir))) {
        // 跳过隐藏文件
        if (entry->d_name[0] == '.') {
            continue;
        }

        std::string device_name = entry->d_name;

        // 解析设备的PCI总线信息
        char path[PATH_MAX + 32];
        char resolved_path[PATH_MAX];
        // InfiniBand设备目录实际上是一个符号链接：
        // "/sys/class/infiniband/mlx5_X/" -> "/sys/devices/pciXXXX:XX/XXXX:XX:XX.X/infiniband/mlx5_X/"
        snprintf(path, sizeof(path), "/sys/class/infiniband/%s/../..",
                 entry->d_name);
        if (realpath(path, resolved_path) == NULL) {
            PLOG(ERROR) << "Failed to parse realpath";
            continue;
        }
        std::string pci_bus_id = basename(resolved_path);

        // 获取设备的NUMA节点信息
        int numa_node = -1;
        snprintf(path, sizeof(path), "%s/numa_node", resolved_path);
        std::ifstream(path) >> numa_node;

        // 添加设备信息到列表
        devices.push_back(InfinibandDevice{.name = std::move(device_name),
                                           .pci_bus_id = std::move(pci_bus_id),
                                           .numa_node = numa_node});
    }
    (void)closedir(dir);
    return devices;
}

/**
 * @brief 发现CPU的NUMA拓扑结构
 * @param all_hca 所有IB设备的列表
 * @return 拓扑条目列表
 *
 * 该函数分析系统的NUMA拓扑：
 * 1. 遍历所有NUMA节点
 * 2. 为每个节点构建本地和远程设备列表
 * 3. 生成优先级矩阵
 */
static std::vector<TopologyEntry> discoverCpuTopology(
    const std::vector<InfinibandDevice> &all_hca) {
    // 打开NUMA节点目录
    DIR *dir = opendir("/sys/devices/system/node");
    struct dirent *entry;
    std::vector<TopologyEntry> topology;

    if (dir == NULL) {
        PLOG(WARNING) << "Failed to open /sys/devices/system/node";
        return {};
    }

    // 遍历所有NUMA节点
    while ((entry = readdir(dir))) {
        const char *prefix = "node";
        // 只处理格式为"nodeX"的目录
        if (entry->d_type != DT_DIR ||
            strncmp(entry->d_name, prefix, strlen(prefix)) != 0) {
            continue;
        }
        int node_id = atoi(entry->d_name + strlen(prefix));
        std::vector<std::string> preferred_hca;
        std::vector<std::string> avail_hca;
        // an HCA connected to the same cpu NUMA node is preferred
        for (const auto &hca : all_hca) {
            if (hca.numa_node == node_id) {
                preferred_hca.push_back(hca.name);
            } else {
                avail_hca.push_back(hca.name);
            }
        }
        topology.push_back(
            TopologyEntry{.name = "cpu:" + std::to_string(node_id),
                          .preferred_hca = std::move(preferred_hca),
                          .avail_hca = std::move(avail_hca)});
    }
    (void)closedir(dir);
    return topology;
}

#ifdef USE_CUDA

/**
 * @brief 计算两个PCI设备之间的距离
 * @param bus1 PCI设备1的总线ID
 * @param bus2 PCI设备2的总线ID
 * @return 距离值，表示两个设备之间的层级距离
 *
 * 该函数通过比较两个PCI设备的路径，计算它们之间的层级距离：
 * - 路径越接近，距离值越小
 * - 路径越远，距离值越大
 */
static int getPciDistance(const char *bus1, const char *bus2) {
    char buf[PATH_MAX];
    char path1[PATH_MAX];
    char path2[PATH_MAX];
    snprintf(buf, sizeof(buf), "/sys/bus/pci/devices/%s", bus1);
    if (realpath(buf, path1) == NULL) {
        return -1;
    }
    snprintf(buf, sizeof(buf), "/sys/bus/pci/devices/%s", bus2);
    if (realpath(buf, path2) == NULL) {
        return -1;
    }

    char *ptr1 = path1;
    char *ptr2 = path2;
    while (*ptr1 && *ptr1 == *ptr2) {
        ptr1++;
        ptr2++;
    }
    int distance = 0;
    for (; *ptr1; ptr1++) {
        distance += (*ptr1 == '/');
    }
    for (; *ptr2; ptr2++) {
        distance += (*ptr2 == '/');
    }

    return distance;
}

/**
 * @brief 发现CUDA设备的拓扑结构
 * @param all_hca 所有IB设备的列表
 * @return CUDA拓扑条目列表
 *
 * 该函数分析系统中的CUDA设备及其拓扑：
 * 1. 遍历所有CUDA设备
 * 2. 根据PCI距离将设备分为优先和可用两类
 * 3. 生成拓扑条目列表
 */
static std::vector<TopologyEntry> discoverCudaTopology(
    const std::vector<InfinibandDevice> &all_hca) {
    std::vector<TopologyEntry> topology;
    int device_count;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
        device_count = 0;
    }
    for (int i = 0; i < device_count; i++) {
        char pci_bus_id[20];
        if (cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), i) !=
            cudaSuccess) {
            continue;
        }
        for (char *ch = pci_bus_id; (*ch = tolower(*ch)); ch++);

        std::vector<std::string> preferred_hca;
        std::vector<std::string> avail_hca;
        for (const auto &hca : all_hca) {
            // FIXME: currently we only identify the NICs connected to the same
            // PCIe switch/RC with GPU as preferred.
            if (getPciDistance(hca.pci_bus_id.c_str(), pci_bus_id) == 0) {
                preferred_hca.push_back(hca.name);
            } else {
                avail_hca.push_back(hca.name);
            }
        }
        topology.push_back(
            TopologyEntry{.name = "cuda:" + std::to_string(i),
                          .preferred_hca = std::move(preferred_hca),
                          .avail_hca = std::move(avail_hca)});
    }
    return topology;
}

#endif  // USE_CUDA

Topology::Topology() {}

Topology::~Topology() {}

bool Topology::empty() const { return matrix_.empty(); }

void Topology::clear() {
    matrix_.clear();
    hca_list_.clear();
    resolved_matrix_.clear();
}

/**
 * @brief 发现系统的网络拓扑
 * @return 成功返回0，失败返回错误码
 *
 * 该函数是拓扑发现的主入口：
 * 1. 清空现有拓扑信息
 * 2. 列出所有InfiniBand设备
 * 3. 发现CPU和CUDA的拓扑结构
 * 4. 解析并生成拓扑矩阵
 */
int Topology::discover() {
    matrix_.clear();
    auto all_hca = listInfiniBandDevices();
    for (auto &ent : discoverCpuTopology(all_hca)) {
        matrix_[ent.name] = ent;
    }
#ifdef USE_CUDA
    for (auto &ent : discoverCudaTopology(all_hca)) {
        matrix_[ent.name] = ent;
    }
#endif
    return resolve();
}

/**
 * @brief 解析给定的拓扑JSON字符串
 * @param topology_json 拓扑信息的JSON字符串
 * @return 成功返回0，失败返回错误码
 *
 * 该函数用于从外部加载拓扑信息：
 * 1. 解析JSON格式的拓扑字符串
 * 2. 将解析结果存入拓扑矩阵
 */
int Topology::parse(const std::string &topology_json) {
    std::set<std::string> rnic_set;
    Json::Value root;
    Json::Reader reader;

    if (topology_json.empty() || !reader.parse(topology_json, root)) {
        LOG(ERROR) << "Topology: malformed json format";
        return ERR_MALFORMED_JSON;
    }

    matrix_.clear();
    for (const auto &key : root.getMemberNames()) {
        const Json::Value &value = root[key];
        if (value.isArray() && value.size() == 2) {
            TopologyEntry topo_entry;
            topo_entry.name = key;
            for (const auto &array : value[0]) {
                auto device_name = array.asString();
                topo_entry.preferred_hca.push_back(device_name);
            }
            for (const auto &array : value[1]) {
                auto device_name = array.asString();
                topo_entry.avail_hca.push_back(device_name);
            }
            matrix_[key] = topo_entry;
        } else {
            LOG(ERROR) << "Topology: malformed json format";
            return ERR_MALFORMED_JSON;
        }
    }

    return resolve();
}

/**
 * @brief 将拓扑信息转换为字符串
 * @return 拓扑信息字符串
 *
 * 该函数将当前拓扑矩阵转换为可读的字符串格式：
 * - 主要用于调试和日志输出
 */
std::string Topology::toString() const {
    Json::Value value(Json::objectValue);
    for (auto &entry : matrix_) {
        value[entry.first] = entry.second.toJson();
    }
    return value.toStyledString();
}

/**
 * @brief 将拓扑信息转换为JSON格式
 * @return JSON格式的拓扑信息
 *
 * 该函数将当前拓扑矩阵转换为JSON对象：
 * - 主要用于外部接口返回和持久化存储
 */
Json::Value Topology::toJson() const {
    Json::Value root;
    Json::Reader reader;
    reader.parse(toString(), root);
    return root;
}

/**
 * @brief 选择合适的设备进行存储操作
 * @param storage_type 存储类型标识
 * @param retry_count 重试次数
 * @return 选择的设备ID，失败返回错误码
 *
 * 该函数根据拓扑信息和负载均衡策略，选择一个合适的设备：
 * - 首先尝试从preferred_hca中选择
 * - 如果失败，则从avail_hca中选择
 * - 支持重试机制，以应对临时故障
 */
int Topology::selectDevice(const std::string storage_type, int retry_count) {
    if (!resolved_matrix_.count(storage_type)) return ERR_DEVICE_NOT_FOUND;
    auto &entry = resolved_matrix_[storage_type];
    if (retry_count == 0) {
        int rand_value = SimpleRandom::Get().next();
        if (!entry.preferred_hca.empty())
            return entry.preferred_hca[rand_value % entry.preferred_hca.size()];
        else
            return entry.avail_hca[rand_value % entry.avail_hca.size()];
    } else {
        size_t index = (retry_count - 1) %
                       (entry.preferred_hca.size() + entry.avail_hca.size());
        if (index < entry.preferred_hca.size())
            return entry.preferred_hca[index];
        else {
            index -= entry.preferred_hca.size();
            return entry.avail_hca[index];
        }
    }
    return 0;
}

/**
 * @brief 解析并生成拓扑解析结果
 * @return 成功返回0，失败返回错误码
 *
 * 该函数将原始拓扑矩阵解析为一个更简单的格式，便于后续处理：
 * - 为每个设备分配唯一的ID
 * - 生成resolved_matrix用于快速查找
 */
int Topology::resolve() {
    std::map<std::string, int> hca_id_map;
    int next_hca_map_index = 0;
    for (auto &entry : matrix_) {
        for (auto &hca : entry.second.preferred_hca) {
            if (!hca_id_map.count(hca)) {
                hca_list_.push_back(hca);
                hca_id_map[hca] = next_hca_map_index;
                next_hca_map_index++;
            }
            resolved_matrix_[entry.first].preferred_hca.push_back(
                hca_id_map[hca]);
        }
        for (auto &hca : entry.second.avail_hca) {
            if (!hca_id_map.count(hca)) {
                hca_list_.push_back(hca);
                hca_id_map[hca] = next_hca_map_index;
                next_hca_map_index++;
            }
            resolved_matrix_[entry.first].avail_hca.push_back(hca_id_map[hca]);
        }
    }
    return 0;
}
}  // namespace mooncake
