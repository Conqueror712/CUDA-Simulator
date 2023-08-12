#![allow(non_snake_case)]
use core::ffi::c_size_t;
use libc::size_t;
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

pub type c_size_t = usize;
pub type c_ssize_t = isize;

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
    let func: libloading::Symbol<
        unsafe extern "C" fn() -> CUresult,
    > = LIBCUDA.get(b"cuFlushGPUDirectRDMAWrites").unwrap();

    let result = func();
    eprintln!(
        "cuFlushGPUDirectRDMAWrites() -> {:?}",
        result
    );
    result
}

pub unsafe extern "C" fn cuDevicePrimaryCtxRetain(
    pctx: *mut CUcontext, dev: CUdevice) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult> =
    LIBCUDA.get(b"cuDevicePrimaryCtxRetain").unwrap();
    
    let result = func(pctx, dev);
    eprintln!(
        "cuDevicePrimaryCtxRetain(pctx: {:?}, dev: {:?}) -> {:?}",
        pctx.as_ref(),
        dev,
        result
    );
    result
}

pub unsafe extern "C" fn cuDevicePrimaryCtxRelease(
    dev: CUdevice) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(CUdevice) -> CUresult> =
        LIBCUDA.get(b"cuDevicePrimaryCtxRelease").unwrap();

    let result = func(dev);
    eprintln!("cuDevicePrimaryCtxRelease(dev: {:?}) -> {:?}", dev, result);
    result
}

pub unsafe extern "C" fn cuDevicePrimaryCtxSetFlags(
    dev: CUdevice, flags: c_uint) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(CUdevice, c_uint) -> CUresult> =
        LIBCUDA.get(b"cuDevicePrimaryCtxSetFlags").unwrap();

    let result = func(dev, flags);
    eprintln!(
        "cuDevicePrimaryCtxSetFlags(dev: {:?}, flags: {:?}) -> {:?}",
        dev, flags, result
    );
    result
}

pub unsafe extern "C" fn cuDevicePrimaryCtxGetState(
    dev: CUdevice, flags: *mut c_uint) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(CUdevice, *mut c_uint) -> CUresult> =
        LIBCUDA.get(b"cuDevicePrimaryCtxGetState").unwrap();

    let result = func(dev, flags);
    eprintln!(
        "cuDevicePrimaryCtxGetState(dev: {:?}, flags: {:?}) -> {:?}",
        dev,
        flags.as_ref(),
        result
    );
    result
}

pub unsafe extern "C" fn cuDevicePrimaryCtxReset(
    dev: CUdevice) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(CUdevice) -> CUresult> =
        LIBCUDA.get(b"cuDevicePrimaryCtxReset").unwrap();

    let result = func(dev);
    eprintln!("cuDevicePrimaryCtxReset(dev: {:?}) -> {:?}", dev, result);
    result
}

pub unsafe extern "C" fn cuCtxCreate(
    pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult> =
        LIBCUDA.get(b"cuCtxCreate").unwrap();

    let result = func(pctx, flags, dev);
    eprintln!(
        "cuCtxCreate(pctx: {:?}, flags: {:?}, dev: {:?}) -> {:?}",
        pctx.as_ref(),
        flags,
        dev,
        result
    );
    result
}

pub unsafe extern "C" fn cuCtxGetFlags(
    pflags: *mut c_uint, ctx: CUcontext) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(*mut c_uint, CUcontext) -> CUresult> =
        LIBCUDA.get(b"cuCtxGetFlags").unwrap();

    let result = func(pflags, ctx);
    eprintln!(
        "cuCtxGetFlags(pflags: {:?}, ctx: {:?}) -> {:?}",
        pflags.as_ref(),
        ctx,
        result
    );
    result
}

pub unsafe extern "C" fn cuCtxSetCurrent(
    ctx: CUcontext) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(CUcontext) -> CUresult> =
        LIBCUDA.get(b"cuCtxSetCurrent").unwrap();

    let result = func(ctx);
    eprintln!("cuCtxSetCurrent(ctx: {:?}) -> {:?}", ctx, result);
    result
}

pub unsafe extern "C" fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(*mut CUcontext) -> CUresult> =
        LIBCUDA.get(b"cuCtxGetCurrent").unwrap();

    let result = func(pctx);
    eprintln!("cuCtxGetCurrent(pctx: {:?}) -> {:?}", pctx.as_ref(), result);
    result
}

pub unsafe extern "C" fn cuCtxDetach(ctx: CUcontext) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(CUcontext) -> CUresult> =
        LIBCUDA.get(b"cuCtxDetach").unwrap();

    let result = func(ctx);
    eprintln!("cuCtxDetach(ctx: {:?}) -> {:?}", ctx, result);
    result
}

pub unsafe extern "C" fn cuCtxGetApiVersion(ctx: CUcontext, version: *mut c_uint) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(CUcontext, *mut c_uint) -> CUresult> =
        LIBCUDA.get(b"cuCtxGetApiVersion").unwrap();

    let result = func(ctx, version);
    eprintln!(
        "cuCtxGetApiVersion(ctx: {:?}, version: {:?}) -> {:?}",
        ctx,
        version.as_ref(),
        result
    );
    result
}

pub unsafe extern "C" fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn(*mut CUdevice) -> CUresult> =
        LIBCUDA.get(b"cuCtxGetDevice").unwrap();

    let result = func(device);
    eprintln!("cuCtxGetDevice(device: {:?}) -> {:?}", device.as_ref(), result);
    result
}

pub unsafe extern "C" fn cuCtxGetLimit(
    pvalue: *mut size_t,
    limit: CUlimit,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut size_t, CUlimit) -> CUresult,
    > = LIBCUDA.get(b"cuCtxGetLimit").unwrap();

    let result = func(pvalue, limit);
    eprintln!(
        "cuCtxGetLimit(pvalue: {:?}, limit: {:?}) -> {:?}",
        pvalue.as_ref(),
        limit,
        result
    );
    result
}

pub unsafe extern "C" fn cuCtxSetLimit(
    limit: CUlimit,
    value: size_t,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(CUlimit, size_t) -> CUresult,
    > = LIBCUDA.get(b"cuCtxSetLimit").unwrap();

    let result = func(limit, value);
    eprintln!(
        "cuCtxSetLimit(limit: {:?}, value: {:?}) -> {:?}",
        limit,
        value,
        result
    );
    result
}

pub unsafe extern "C" fn cuCtxGetCacheConfig(
    pconfig: *mut CUfunc_cache,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUfunc_cache) -> CUresult,
    > = LIBCUDA.get(b"cuCtxGetCacheConfig").unwrap();

    let result = func(pconfig);
    eprintln!(
        "cuCtxGetCacheConfig(pconfig: {:?}) -> {:?}",
        pconfig.as_ref(),
        result
    );
    result
}

pub unsafe extern "C" fn cuCtxSetCacheConfig(
    config: CUfunc_cache,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(CUfunc_cache) -> CUresult,
    > = LIBCUDA.get(b"cuCtxSetCacheConfig").unwrap();

    let result = func(config);
    eprintln!("cuCtxSetCacheConfig(config: {:?}) -> {:?}", config, result);
    result
}

pub unsafe extern "C" fn cuCtxGetSharedMemConfig(
    pConfig: *mut CUsharedconfig,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUsharedconfig) -> CUresult,
    > = LIBCUDA.get(b"cuCtxGetSharedMemConfig").unwrap();

    let result = func(pConfig);
    eprintln!(
        "cuCtxGetSharedMemConfig(pConfig: {:?}) -> {:?}",
        pConfig.as_ref(),
        result
    );
    result
}

pub unsafe extern "C" fn cuCtxGetStreamPriorityRange(
    leastPriority: *mut c_int,
    greatestPriority: *mut c_int,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut c_int, *mut c_int) -> CUresult,
    > = LIBCUDA.get(b"cuCtxGetStreamPriorityRange").unwrap();

    let result = func(leastPriority, greatestPriority);
    eprintln!(
        "cuCtxGetStreamPriorityRange(leastPriority: {:?}, greatestPriority: {:?}) -> {:?}",
        leastPriority.as_ref(),
        greatestPriority.as_ref(),
        result
    );
    result
}

pub unsafe extern "C" fn cuCtxSetSharedMemConfig(
    config: CUsharedconfig,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(CUsharedconfig) -> CUresult,
    > = LIBCUDA.get(b"cuCtxSetSharedMemConfig").unwrap();

    let result = func(config);
    eprintln!(
        "cuCtxSetSharedMemConfig(config: {:?}) -> {:?}",
        config,
        result
    );
    result
}

pub unsafe extern "C" fn cuCtxSynchronize() -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn() -> CUresult> =
        LIBCUDA.get(b"cuCtxSynchronize").unwrap();

    let result = func();
    eprintln!("cuCtxSynchronize() -> {:?}", result);
    result
}

pub unsafe extern "C" fn cuCtxResetPersistingL2Cache() -> CUresult {
    let func: libloading::Symbol<unsafe extern "C" fn() -> CUresult> =
        LIBCUDA.get(b"cuCtxResetPersistingL2Cache").unwrap();

    let result = func();
    eprintln!("cuCtxResetPersistingL2Cache() -> {:?}", result);
    result
}

pub unsafe extern "C" fn cuCtxPopCurrent(
    pctx: *mut CUcontext,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUcontext) -> CUresult,
    > = LIBCUDA.get(b"cuCtxPopCurrent").unwrap();

    let result = func(pctx);
    eprintln!("cuCtxPopCurrent(pctx: {:?}) -> {:?}", pctx.as_ref(), result);
    result
}

pub unsafe extern "C" fn cuCtxPushCurrent(
    ctx: CUcontext,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(CUcontext) -> CUresult,
    > = LIBCUDA.get(b"cuCtxPushCurrent").unwrap();

    let result = func(ctx);
    eprintln!("cuCtxPushCurrent(ctx: {:?}) -> {:?}", ctx, result);
    result
}

pub unsafe extern "C" fn cuModuleLoad(
    module: *mut CUmodule,
    fname: *const c_char,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUmodule, *const c_char) -> CUresult,
    > = LIBCUDA.get(b"cuModuleLoad").unwrap();

    let result = func(module, fname);
    eprintln!(
        "cuModuleLoad(module: {:?}, fname: {:?}) -> {:?}",
        module.as_ref(),
        CStr::from_ptr(fname).to_string_lossy(),
        result
    );
    result
}

pub unsafe extern "C" fn cuModuleLoadData(
    module: *mut CUmodule,
    image: *const c_void,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    > = LIBCUDA.get(b"cuModuleLoadData").unwrap();

    let result = func(module, image);
    eprintln!(
        "cuModuleLoadData(module: {:?}, image: {:?}) -> {:?}",
        module.as_ref(),
        image,
        result
    );
    result
}

pub unsafe extern "C" fn cuModuleLoadFatBinary(
    module: *mut CUmodule,
    fatCubin: *const c_void,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    > = LIBCUDA.get(b"cuModuleLoadFatBinary").unwrap();

    let result = func(module, fatCubin);
    eprintln!(
        "cuModuleLoadFatBinary(module: {:?}, fatCubin: {:?}) -> {:?}",
        module.as_ref(),
        fatCubin,
        result
    );
    result
}

pub unsafe extern "C" fn cuModuleUnload(
    module: CUmodule,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(CUmodule) -> CUresult,
    > = LIBCUDA.get(b"cuModuleUnload").unwrap();

    let result = func(module);
    eprintln!("cuModuleUnload(module: {:?}) -> {:?}", module, result);
    result
}

pub unsafe extern "C" fn cuModuleGetFunction(
    hfunc: *mut CUfunction,
    hmod: CUmodule,
    name: *const c_char,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    > = LIBCUDA.get(b"cuModuleGetFunction").unwrap();

    let result = func(hfunc, hmod, name);
    eprintln!(
        "cuModuleGetFunction(hfunc: {:?}, hmod: {:?}, name: {:?}) -> {:?}",
        hfunc.as_ref(),
        hmod,
        CStr::from_ptr(name).to_string_lossy(),
        result
    );
    result
}

pub unsafe extern "C" fn cuModuleGetGlobal(
    dptr: *mut CUdeviceptr,
    bytes: *mut size_t,
    hmod: CUmodule,
    name: *const c_char,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUdeviceptr, *mut size_t, CUmodule, *const c_char) -> CUresult,
    > = LIBCUDA.get(b"cuModuleGetGlobal").unwrap();

    let result = func(dptr, bytes, hmod, name);
    eprintln!(
        "cuModuleGetGlobal(dptr: {:?}, bytes: {:?}, hmod: {:?}, name: {:?}) -> {:?}",
        dptr.as_ref(),
        bytes.as_ref(),
        hmod,
        CStr::from_ptr(name).to_string_lossy(),
        result
    );
    result
}

pub unsafe extern "C" fn cuModuleGetTexRef(
    pTexRef: *mut CUtexref,
    hmod: CUmodule,
    name: *const c_char,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUtexref, CUmodule, *const c_char) -> CUresult,
    > = LIBCUDA.get(b"cuModuleGetTexRef").unwrap();

    let result = func(pTexRef, hmod, name);
    eprintln!(
        "cuModuleGetTexRef(pTexRef: {:?}, hmod: {:?}, name: {:?}) -> {:?}",
        pTexRef.as_ref(),
        hmod,
        CStr::from_ptr(name).to_string_lossy(),
        result
    );
    result
}

pub unsafe extern "C" fn cuModuleGetSurfRef(
    pSurfRef: *mut CUsurfref,
    hmod: CUmodule,
    name: *const c_char,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUsurfref, CUmodule, *const c_char) -> CUresult,
    > = LIBCUDA.get(b"cuModuleGetSurfRef").unwrap();

    let result = func(pSurfRef, hmod, name);
    eprintln!(
        "cuModuleGetSurfRef(pSurfRef: {:?}, hmod: {:?}, name: {:?}) -> {:?}",
        pSurfRef.as_ref(),
        hmod,
        CStr::from_ptr(name).to_string_lossy(),
        result
    );
    result
}

pub unsafe extern "C" fn cuModuleGetLoadingMode(
    mode: *mut CUjitInputType,
    hmod: CUmodule,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUjitInputType, CUmodule) -> CUresult,
    > = LIBCUDA.get(b"cuModuleGetLoadingMode").unwrap();

    let result = func(mode, hmod);
    eprintln!(
        "cuModuleGetLoadingMode(mode: {:?}, hmod: {:?}) -> {:?}",
        mode.as_ref(),
        hmod,
        result
    );
    result
}