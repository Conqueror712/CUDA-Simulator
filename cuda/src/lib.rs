#![allow(non_snake_case)]
use cuda_sys::*;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref LIBCUDA: libloading::Library = unsafe {
        libloading::Library::new(std::env::var("LIBCUDA").unwrap_or("/usr/lib/wsl/lib/libcuda.so".to_string())).unwrap()
    };
    static ref TABEL: Mutex<HashMap<(CString, c_int, cuuint64_t), usize>> = Default::default();
}

#[no_mangle]
pub unsafe extern "C" fn cuGetProcAddress_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cudaVersion: c_int,
    flags: cuuint64_t,
    status: *mut CUdriverProcAddressQueryResult,
) -> CUresult {
    let lookup: libloading::Symbol<
        unsafe extern "C" fn(
            *const c_char,
            *mut *mut c_void,
            c_int,
            cuuint64_t,
            *mut CUdriverProcAddressQueryResult,
        ) -> CUresult,
    > = LIBCUDA.get(b"cuGetProcAddress_v2").unwrap();

    let res = lookup(symbol, pfn, cudaVersion, flags, status);

    let symbol = CStr::from_ptr(symbol);

    TABEL
        .lock()
        .unwrap()
        .insert((symbol.into(), cudaVersion, flags), *pfn as _);

    match (symbol.to_str().unwrap(), cudaVersion, flags) {
        ("cuInit", 2000, 0) => {
            *pfn = cuInit as _;
        }
        ("cuDeviceGetCount", 2000, 0) => {
            *pfn = cuDeviceGetCount as _;
        }
        ("cuDeviceGet", 2000, 0) => {
            *pfn = cuDeviceGet as _;
        }
        ("cuDeviceGetName", 2000, 0) => {
            *pfn = cuDeviceGetName as _;
        }
        ("cuDeviceGetAttribute", 2000, 0) => {
            *pfn = cuDeviceGetAttribute as _;
        }
        ("cuDeviceTotalMem", 3020, 0) => {
            *pfn = cuDeviceTotalMem_v2 as _;
        }
        ("cuGetProcAddress", _, 0) => {
            *pfn = cuGetProcAddress_v2 as _;
        }
        ("cuGetExportTable", 3000, 0) => {
            *pfn = cuGetExportTable as _;
        }
        _ => {
            eprintln!(
                "cuGetProcAddress_v2({:?}, {:?}, {}, {}, {:?}) -> {:?}",
                symbol,
                pfn.as_ref(),
                cudaVersion,
                flags,
                status.as_ref(),
                res
            );
        }
    }

    res
}

pub unsafe extern "C" fn cuDeviceGetAttribute(
    pi: *mut c_int,
    attrib: CUdevice_attribute,
    dev: CUdevice,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut c_int, CUdevice_attribute, CUdevice) -> CUresult,
    > = LIBCUDA.get(b"cuDeviceGetAttribute").unwrap();

    let result = func(pi, attrib, dev);
    eprintln!(
        "cuDeviceGetAttribute(pi: {:?}, attrib: {:?}, dev: {}) -> {:?}",
        pi.as_ref(),
        attrib,
        dev,
        result
    );
    result
}

pub unsafe extern "C" fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(*mut usize, CUdevice) -> CUresult> =
        LIBCUDA.get(b"cuDeviceTotalMem_v2").unwrap();

    let result = func(bytes, dev);
    eprintln!(
        "cuDeviceTotalMem(bytes: {:?}, dev {}) -> {:?}",
        bytes.as_ref(),
        dev,
        result
    );
    result
}

pub unsafe extern "C" fn cuInit(flags: c_uint) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(c_uint) -> CUresult> =
        LIBCUDA.get(b"cuInit").unwrap();

    let result = func(flags);
    eprintln!("cuInit(flags: {}) -> {:?}", flags, result);
    result
}

pub unsafe extern "C" fn cuDeviceGetCount(count: *mut c_int) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(*mut c_int) -> CUresult> =
        LIBCUDA.get(b"cuDeviceGetCount").unwrap();

    let result = func(count);
    eprintln!(
        "cuDeviceGetCount(count: {:?}) -> {:?}",
        count.as_ref(),
        result
    );
    result
}

pub unsafe extern "C" fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult> =
        LIBCUDA.get(b"cuDeviceGet").unwrap();

    let result = func(device, ordinal);
    eprintln!(
        "cuDeviceGet(device: {:?}, ordinal: {}) -> {:?}",
        device.as_ref(),
        ordinal,
        result
    );
    result
}

pub unsafe extern "C" fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult> =
        LIBCUDA.get(b"cuDeviceGetName").unwrap();

    let result = func(name, len, dev);
    eprintln!(
        "cuDeviceGetName(name: {:?}, len: {}, dev: {}) -> {:?}",
        CStr::from_ptr(name),
        len,
        dev,
        result
    );
    result
}

pub unsafe extern "C" fn cuGetExportTable(
    ppExportTable: *mut *const c_void,
    pExportTableId: *const CUuuid,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut *const c_void, *const CUuuid) -> CUresult,
    > = LIBCUDA.get(b"cuGetExportTable").unwrap();

    let result = func(ppExportTable, pExportTableId);
    eprintln!(
        "cuGetExportTable(ppExportTable: {:?}, pExportTableId: {:?}) -> {:?}",
        ppExportTable.as_ref(),
        pExportTableId.as_ref(),
        result
    );
    result
}

// 2023.07.29 Sum = 8

pub unsafe extern "C" fn cudaMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,  
) -> cudaError_t {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void, *const c_void, usize, cudaMemcpyKind) -> cudaError_t,
    > = LIBCUDA.get(b"cudaMemcpy").unwrap();

    let result = func(dst, src, count, kind);
    eprintln!(
        "cudaMemcpy(dst: {:?}, src: {:?}, count: {:?}, kind: {:?}) -> {:?}",
        dst,
        src,
        count,
        kind,
        result
    );
    result
}

pub unsafe extern "C" fn cudaMalloc(
    dev_ptr: *mut *mut c_void,
    size: usize,
    ) -> cudaError_t {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut *mut c_void, usize) -> cudaError_t,
    > = LIBCUDA.get(b"cudaMalloc_v2").unwrap();
    let result = func(dev_ptr, size);
    eprintln!(
        "cudaMalloc(dev_ptr: {:?}, size: {:?}) -> {:?}",
        dev_ptr.as_ref(),
        size,
        result
    );
    result
}

pub unsafe extern "C" fn cudaFree(
    dev_ptr: *mut c_void,
) -> cudaError_t {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void) -> cudaError_t,
    > = LIBCUDA.get(b"cudaFree_v2").unwrap();

    let result = func(dev_ptr);
    eprintln!(
        "cudaFree(dev_ptr: {:?}) -> {:?}",
        dev_ptr,
        result
    );
    result
}

pub unsafe extern "C" fn cudaGetDeviceCount(
    count: *mut c_int
) -> cudaError_t {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut c_int) -> cudaError_t,
    > = LIBCUDA.get(b"cudaGetDeviceCount_v2").unwrap();

    let result = func(count);
    eprintln!(
        "cudaGetDeviceCount(count: {:?}) -> {:?}",
        count,
        result
    );
    result
}

pub unsafe extern "C" fn cudaGetDeviceProperties(
    prop: *mut cudaDeviceProp, dev: c_int
) -> cudaError_t {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut cudaDeviceProp, c_int) -> cudaError_t,
    > = LIBCUDA.get(b"cudaGetDeviceProperties").unwrap();

    let result = func(prop, dev);
    eprintln!(
        "cudaGetDeviceProperties(prop: {:?}, dev: {:?}) -> {:?}",
        *prop,
        dev, 
        result
    );
    result
}

pub unsafe extern "C" fn cudaDeviceSynchronize(
    // NULL
) -> cudaError_t {
    let func: libloading::Symbol<
    unsafe extern "C" fn() -> cudaError_t,
    > = LIBCUDA.get(b"cudaDeviceSynchronize_v2").unwrap();

    let result = func();
    eprintln!(
        "cudaDeviceSynchronize() -> {:?}", 
        result
    );
    result
}

pub unsafe extern "C" fn cudaGetLastError(
    // NULL
) -> cudaError_t {
    let func: libloading::Symbol<
    unsafe extern "C" fn() -> cudaError_t,
    > = LIBCUDA.get(b"cudaGetLastError").unwrap();
    
    let result = func();
    eprintln!(
        "cudaGetLastError() -> {:?}", 
        result
    );
    result
}

pub unsafe extern "C" fn cudaGetErrorString(
    error: cudaError_t
) -> *const c_char {
    let func: libloading::Symbol<
    unsafe extern "C" fn(cudaError_t) -> *const c_char,
    > = LIBCUDA.get(b"cudaGetErrorString").unwrap();
    let result = func(error);
    eprintln!(
        "cudaGetErrorString(error: {:?}) -> {:?}",
        error,
        CStr::from_ptr(result)
    );
    result
}

// 2023.08.05 Sum = 10

pub unsafe extern "C" fn cuDeviceGetP2PAttribute(
    value: *mut c_int,
    attrib: CUdevice_P2PAttribute,
    srcDevice: CUdevice,
    dstDevice: CUdevice,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut c_int, CUdevice_P2PAttribute, CUdevice, CUdevice) -> CUresult,
    > = LIBCUDA.get(b"cuDeviceGetP2PAttribute").unwrap();

    let result = func(value, attrib, srcDevice, dstDevice);
    eprintln!(
        "cuDeviceGetP2PAttribute(value: {:?}, attrib: {:?}, srcDevice: {:?}, dstDevice: {:?}) -> {:?}",
        value,
        attrib,
        srcDevice,
        dstDevice,
        result
    );
    result
}

pub unsafe extern "C" fn cuDriverGetVersion(
    version: *mut c_int,
) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(*mut c_int) -> CUresult> =
        LIBCUDA.get(b"cuDriverGetVersion").unwrap();

    let result = func(version);
    eprintln!(
        "cuDriverGetVersion(version: {:?}) -> {:?}",
        version,
        result
    );
    result
}

pub unsafe extern "C" fn cuDeviceGetByPCIBusId(
    dev: *mut CUdevice,
    pciBusId: *const c_char,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUdevice, *const c_char) -> CUresult,
    > = LIBCUDA.get(b"cuDeviceGetByPCIBusId").unwrap();

    let result = func(dev, pciBusId);
    eprintln!(
        "cuDeviceGetByPCIBusId(dev: {:?}, pciBusId: {:?}) -> {:?}",
        dev, CStr::from_ptr(pciBusId), result
    );
    result
}

pub unsafe extern "C" fn cuDeviceGetPCIBusId(
    pciBusId: *mut c_char,
    len: c_int,
    dev: CUdevice,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult,
    > = LIBCUDA.get(b"cuDeviceGetPCIBusId").unwrap();

    let result = func(pciBusId, len, dev);
    eprintln!(
        "cuDeviceGetPCIBusId(pciBusId: {:?}, len: {:?}, dev: {:?}) -> {:?}",
        CStr::from_ptr(pciBusId),
        len,
        dev,
        result
    );
    result
}

pub unsafe extern "C" fn cuDeviceGetUuid(
    uuid: *mut CUuuid,
    dev: CUdevice,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUuuid, CUdevice) -> CUresult,
    > = LIBCUDA.get(b"cuDeviceGetUuid").unwrap();

    let result = func(uuid, dev);
    eprintln!(
        "cuDeviceGetUuid(uuid: {:?}, dev: {:?}) -> {:?}",
        *uuid,
        dev,
        result
    );
    result
}

pub unsafe extern "C" fn cuDeviceGetTexture1DLinearMaxWidth(
    max_width: *mut c_size_t,
    fmt: CUarray_format,
    num_channels: c_int,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut c_size_t, CUarray_format, c_int) -> CUresult,
    > = LIBCUDA.get(b"cuDeviceGetTexture1DLinearMaxWidth").unwrap();

    let result = func(max_width, fmt, num_channels);
    eprintln!(
        "cuDeviceGetTexture1DLinearMaxWidth(max_width: {:?}, fmt: {:?}, num_channels: {:?}) -> {:?}",
        *max_width,
        fmt,
        num_channels,
        result
    );
    result
}

pub unsafe extern "C" fn cuDeviceGetDefaultMemPool(
    memPool: *mut CUmemoryPool,
    dev: CUdevice,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUmemoryPool, CUdevice) -> CUresult,
    > = LIBCUDA.get(b"cuDeviceGetDefaultMemPool").unwrap();

    let result = func(memPool, dev);
    eprintln!(
        "cuDeviceGetDefaultMemPool(memPool: {:?}, dev: {:?}) -> {:?}",
        *memPool,
        dev,
        result
    );
    result
}

pub unsafe extern "C" fn cuDeviceSetMemPool(
    dev: CUdevice,
    memPool: CUmemoryPool,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(CUdevice, CUmemoryPool) -> CUresult,
    > = LIBCUDA.get(b"cuDeviceSetMemPool").unwrap();

    let result = func(dev, memPool);
    eprintln!(
        "cuDeviceSetMemPool(dev: {:?}, memPool: {:?}) -> {:?}",
        dev, *memPool, result
    );
    result
}

pub unsafe extern "C" fn cuDeviceGetMemPool(
    memPool: *mut CUmemoryPool,
    dev: CUdevice,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUmemoryPool, CUdevice) -> CUresult,
    > = LIBCUDA.get(b"cuDeviceGetMemPool").unwrap();

    let result = func(memPool, dev);
    eprintln!(
        "cuDeviceGetMemPool(memPool: {:?}, dev: {:?}) -> {:?}",
        *memPool, dev, result
    );
    result
}

pub unsafe extern "C" fn cuFlushGPUDirectRDMAWrites(
) -> CUresult {
    // 使用Libloading库的get函数从名为LIBCUDA的外部库中获取一个符号
    // 该符号对应于cuFlushGPUDirectRDMAWrites函数
    // LIBCUDA是一个库loader负责加载CUDA库
    let func: libloading::Symbol<
        unsafe extern "C" fn() -> CUresult,
    > = LIBCUDA.get(b"cuFlushGPUDirectRDMAWrites").unwrap();

    // 调用之前获取的函数符号func
    let result = func();
    // 这行代码打印了函数调用的结果，使用了eprintln!宏来将结果输出到标准错误流
    eprintln!(
        "cuFlushGPUDirectRDMAWrites() -> {:?}",
        result
    );
    result
}