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

package p2pstore

/*
 * All memory pointed to by the "char *" parameters will not be used
 * after the C function returns.
 * This means that the caller can free the memory pointed to by "char *"
 * parameters, after the call is completed.
 * All the C functions used here follow this convention.
 */

/*
#cgo LDFLAGS: -L../../../build/mooncake-transfer-engine/src -L../../../thirdparties/lib -ltransfer_engine -lstdc++ -lnuma -lglog -libverbs -ljsoncpp -letcd-cpp-api -lprotobuf -lgrpc++ -lgrpc
#include "../../../mooncake-transfer-engine/include/transfer_engine_c.h"
#include <stdlib.h>
*/
import "C"

import (
	"net"
	"strconv"
	"unsafe"
)

type BatchID int64

type TransferEngine struct {
	engine C.transfer_engine_t
	xport  C.transport_t
}

func parseServerName(serverName string) (host string, port int) {
	defaultPort := "12345"
	host, portStr, err := net.SplitHostPort(serverName)
	if err != nil {
		host = serverName
		portStr = defaultPort
	}
	port, err = strconv.Atoi(portStr)
	if err != nil {
		port = 12345
	}
	return host, port
}

var (
	rdmaCStr = C.CString("rdma")
)

func NewTransferEngine(metadata_uri string, local_server_name string, nic_priority_matrix string) (*TransferEngine, error) {
	// For simplifiy, local_server_name must be a valid IP address or hostname
	connectable_name, rpc_port := parseServerName(local_server_name)

	metadataUri := C.CString(metadata_uri)
	localServerName := C.CString(local_server_name)
	connectableName := C.CString(connectable_name)
	defer C.free(unsafe.Pointer(metadataUri))
	defer C.free(unsafe.Pointer(localServerName))
	defer C.free(unsafe.Pointer(connectableName))

	native_engine := C.createTransferEngine(metadataUri, localServerName, connectableName, C.uint64_t(rpc_port))
	if native_engine == nil {
		return nil, ErrTransferEngine
	}

	var xport C.transport_t
	if nic_priority_matrix != "" {
		nicPriorityMatrix := C.CString(nic_priority_matrix)
		defer C.free(unsafe.Pointer(nicPriorityMatrix))
		var args [2]unsafe.Pointer
		args[0] = unsafe.Pointer(nicPriorityMatrix)
		args[1] = nil
		xport = C.installTransport(native_engine, rdmaCStr, &args[0])
		if xport == nil {
			C.destroyTransferEngine(native_engine)
			return nil, ErrTransferEngine
		}
	}

	return &TransferEngine{
		engine: native_engine,
		xport:  xport,
	}, nil
}

func (engine *TransferEngine) Close() error {
	if engine.xport != nil {
		ret := C.uninstallTransport(engine.engine, rdmaCStr)
		if ret < 0 {
			return ErrTransferEngine
		}
	}

	if engine.engine != nil {
		C.destroyTransferEngine(engine.engine)
	}
	return nil
}

func (engine *TransferEngine) registerLocalMemory(addr uintptr, length uint64, location string) error {
	locationCStr := C.CString(location)
	defer C.free(unsafe.Pointer(locationCStr))
	ret := C.registerLocalMemory(engine.engine, unsafe.Pointer(addr), C.size_t(length), locationCStr, 1)
	if ret < 0 {
		return ErrTransferEngine
	}
	return nil
}

func (engine *TransferEngine) unregisterLocalMemory(addr uintptr) error {
	ret := C.unregisterLocalMemory(engine.engine, unsafe.Pointer(addr))
	if ret < 0 {
		return ErrTransferEngine
	}
	return nil
}

func (engine *TransferEngine) allocateBatchID(batchSize int) (BatchID, error) {
	ret := C.allocateBatchID(engine.engine, C.size_t(batchSize))
	if ret == C.UINT64_MAX {
		return BatchID(-1), ErrTransferEngine
	}
	return BatchID(ret), nil
}

const (
	OPCODE_READ      = 0
	OPCODE_WRITE     = 1
	STATUS_WAITING   = 0
	STATUS_PENDING   = 1
	STATUS_INVALID   = 2
	STATUS_CANNELED  = 3
	STATUS_COMPLETED = 4
	STATUS_TIMEOUT   = 5
	STATUS_FAILED    = 6
)

type TransferRequest struct {
	Opcode       int
	Source       uint64
	TargetID     int64
	TargetOffset uint64
	Length       uint64
}

func (engine *TransferEngine) submitTransfer(batchID BatchID, requests []TransferRequest) error {
	requestSlice := make([]C.transfer_request_t, len(requests))
	for i, req := range requests {
		requestSlice[i] = C.transfer_request_t{
			opcode:        C.int(req.Opcode),
			source:        unsafe.Pointer(uintptr(req.Source)),
			target_id:     C.segment_id_t(req.TargetID),
			target_offset: C.uint64_t(req.TargetOffset),
			length:        C.uint64_t(req.Length),
		}
	}

	ret := C.submitTransfer(engine.engine, C.batch_id_t(batchID), &requestSlice[0], C.size_t(len(requests)))
	if ret < 0 {
		return ErrTransferEngine
	}
	return nil
}

func (engine *TransferEngine) getTransferStatus(batchID BatchID, taskID int) (int, uint64, error) {
	var status C.transfer_status_t
	ret := C.getTransferStatus(engine.engine, C.batch_id_t(batchID), C.size_t(taskID), &status)
	if ret < 0 {
		return -1, 0, ErrTransferEngine
	}
	return int(status.status), uint64(status.transferred_bytes), nil
}

func (engine *TransferEngine) freeBatchID(batchID BatchID) error {
	ret := C.freeBatchID(engine.engine, C.batch_id_t(batchID))
	if ret < 0 {
		return ErrTransferEngine
	}
	return nil
}

func (engine *TransferEngine) openSegment(name string) (int64, error) {
	nameCStr := C.CString(name)
	defer C.free(unsafe.Pointer(nameCStr))

	ret := C.openSegment(engine.engine, nameCStr)
	if ret < 0 {
		return -1, ErrTransferEngine
	}
	return int64(ret), nil
}

func (engine *TransferEngine) closeSegment(segment_id int64) error {
	ret := C.closeSegment(engine.engine, C.segment_id_t(segment_id))
	if ret < 0 {
		return ErrTransferEngine
	}
	return nil
}

func (engine *TransferEngine) syncSegmentCache() error {
	ret := C.syncSegmentCache(engine.engine)
	if ret < 0 {
		return ErrTransferEngine
	}
	return nil
}
