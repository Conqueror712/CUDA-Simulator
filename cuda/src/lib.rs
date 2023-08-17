#![allow(non_snake_case)]
use libc::size_t;
use cuda_sys::*;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint, c_void, c_ulonglong, c_ulong};
use std::sync::Mutex;
// use libloading::Library;

lazy_static::lazy_static! {
    static ref LIBCUDA: libloading::Library = unsafe {
        libloading::Library::new(std::env::var("LIBCUDA").unwrap_or("/usr/lib/wsl/lib/libcuda.so".to_string())).unwrap()
    };
    static ref TABEL: Mutex<HashMap<(CString, c_int, cuuint64_t), usize>> = Default::default();
}

pub type CSizeT = usize;
pub type CSsizeT = isize;
type c_size_t = c_ulong;

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
    max_width: *mut CSizeT,
    fmt: CUarray_format,
    num_channels: c_int,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CSizeT, CUarray_format, c_int) -> CUresult,
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

pub unsafe extern "C" fn cuLibraryLoadData(
    lib: *mut CUlibrary,
    image: *const c_char,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUlibrary, *const c_char) -> CUresult,
    > = LIBCUDA.get(b"cuLibraryLoadData").unwrap();

    let result = func(lib, image);
    eprintln!(
        "cuLibraryLoadData(lib: {:?}, image: {:?}) -> {:?}",
        lib,
        CStr::from_ptr(image).to_string_lossy(),
        result
    );
    result
}

pub unsafe extern "C" fn cuLibraryLoadFromFile(
    lib: *mut CUlibrary,
    filename: *const c_char,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUlibrary, *const c_char) -> CUresult,
    > = LIBCUDA.get(b"cuLibraryLoadFromFile").unwrap();

    let result = func(lib, filename);
    eprintln!(
        "cuLibraryLoadFromFile(lib: {:?}, filename: {:?}) -> {:?}",
        lib,
        CStr::from_ptr(filename).to_string_lossy(),
        result
    );
    result
}

pub unsafe extern "C" fn cuLibraryUnload(
    lib: CUlibrary,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(CUlibrary) -> CUresult,
    > = LIBCUDA.get(b"cuLibraryUnload").unwrap();

    let result = func(lib);
    eprintln!(
        "cuLibraryUnload(lib: {:?}) -> {:?}",
        lib,
        result
    );
    result
}

pub unsafe extern "C" fn cuLibraryGetKernel(
    hfunc: *mut CUfunction,
    lib: CUlibrary,
    name: *const c_char,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUfunction, CUlibrary, *const c_char) -> CUresult,
    > = LIBCUDA.get(b"cuLibraryGetKernel").unwrap();

    let result = func(hfunc, lib, name);
    eprintln!(
        "cuLibraryGetKernel(hfunc: {:?}, lib: {:?}, name: {:?}) -> {:?}",
        hfunc.as_ref(),
        lib,
        CStr::from_ptr(name).to_string_lossy(),
        result
    );
    result
}

pub unsafe extern "C" fn cuLibraryGetModule(
    module: *mut CUmodule,
    lib: CUlibrary,
    name: *const c_char,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUmodule, CUlibrary, *const c_char) -> CUresult,
    > = LIBCUDA.get(b"cuLibraryGetModule").unwrap();

    let result = func(module, lib, name);
    eprintln!(
        "cuLibraryGetModule(module: {:?}, lib: {:?}, name: {:?}) -> {:?}",
        module.as_ref(),
        lib,
        CStr::from_ptr(name).to_string_lossy(),
        result
    );
    result
}

pub unsafe extern "C" fn cuLibraryGetGlobal(
    dptr: *mut CUdeviceptr,
    bytes: *mut usize,
    lib: CUlibrary,
    name: *const c_char,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUdeviceptr, *mut usize, CUlibrary, *const c_char) -> CUresult,
    > = LIBCUDA.get(b"cuLibraryGetGlobal").unwrap();

    let result = func(dptr, bytes, lib, name);
    eprintln!(
        "cuLibraryGetGlobal(dptr: {:?}, bytes: {:?}, lib: {:?}, name: {:?}) -> {:?}",
        dptr.as_ref(),
        bytes.as_ref(),
        lib,
        CStr::from_ptr(name).to_string_lossy(),
        result
    );
    result
}

pub unsafe extern "C" fn cuLibraryGetManaged(
    p_devptr: *mut CUdeviceptr,
    p_size: *mut usize,
    lib: CUlibrary,
    name: *const c_char,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUdeviceptr, *mut usize, CUlibrary, *const c_char) -> CUresult,
    > = LIBCUDA.get(b"cuLibraryGetManaged").unwrap();

    let result = func(p_devptr, p_size, lib, name);
    eprintln!(
        "cuLibraryGetManaged(p_devptr: {:?}, p_size: {:?}, lib: {:?}, name: {:?}) -> {:?}",
        p_devptr.as_ref(),
        p_size.as_ref(),
        lib,
        CStr::from_ptr(name).to_string_lossy(),
        result
    );
    result
}

pub unsafe extern "C" fn cuKernelGetFunction(
    hfunc: *mut CUfunction,
    kernelName: *const c_char,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUfunction, *const c_char) -> CUresult,
    > = LIBCUDA.get(b"cuKernelGetFunction").unwrap();

    let result = func(hfunc, kernelName);
    eprintln!(
        "cuKernelGetFunction(hfunc: {:?}, kernelName: {:?}) -> {:?}",
        hfunc.as_ref(),
        CStr::from_ptr(kernelName).to_string_lossy(),
        result
    );
    result
}

pub unsafe extern "C" fn cuKernelGetAttribute(
    pi: *mut std::os::raw::c_int,
    attrib: CUfunction_attribute,
    hfunc: CUfunction,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut std::os::raw::c_int, CUfunction_attribute, CUfunction) -> CUresult,
    > = LIBCUDA.get(b"cuKernelGetAttribute").unwrap();

    let result = func(pi, attrib, hfunc);
    eprintln!(
        "cuKernelGetAttribute(pi: {:?}, attrib: {:?}, hfunc: {:?}) -> {:?}",
        pi.as_ref(),
        attrib,
        hfunc,
        result
    );
    result
}

pub unsafe extern "C" fn cuKernelSetAttribute(
    hfunc: CUfunction,
    attrib: CUfunction_attribute,
    value: std::os::raw::c_int,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(CUfunction, CUfunction_attribute, std::os::raw::c_int) -> CUresult,
    > = LIBCUDA.get(b"cuKernelSetAttribute").unwrap();

    let result = func(hfunc, attrib, value);
    eprintln!(
        "cuKernelSetAttribute(hfunc: {:?}, attrib: {:?}, value: {:?}) -> {:?}",
        hfunc,
        attrib,
        value,
        result
    );
    result
}

pub unsafe extern "C" fn cuKernelSetCacheConfig(
    hfunc: CUfunction,
    cacheConfig: CUfunc_cache,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(CUfunction, CUfunc_cache) -> CUresult,
    > = LIBCUDA.get(b"cuKernelSetCacheConfig").unwrap();

    let result = func(hfunc, cacheConfig);
    eprintln!(
        "cuKernelSetCacheConfig(hfunc: {:?}, cacheConfig: {:?}) -> {:?}",
        hfunc,
        cacheConfig,
        result
    );
    result
}

pub unsafe extern "C" fn cuLinkCreate(
    numOptions: std::os::raw::c_uint,
    options: *mut CUjit_option,
    optionValues: *mut std::os::raw::c_void,
    stateOut: *mut CUlinkState,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(
            std::os::raw::c_uint,
            *mut CUjit_option,
            *mut std::os::raw::c_void,
            *mut CUlinkState,
        ) -> CUresult,
    > = LIBCUDA.get(b"cuLinkCreate").unwrap();

    let result = func(numOptions, options, optionValues, stateOut);
    eprintln!(
        "cuLinkCreate(numOptions: {:?}, options: {:?}, optionValues: {:?}, stateOut: {:?}) -> {:?}",
        numOptions,
        options.as_ref(),
        optionValues,
        stateOut.as_ref(),
        result
    );
    result
}

pub unsafe extern "C" fn cuLinkAddData(
    state: CUlinkState,
    type_: CUjitInputType,
    data: *mut std::os::raw::c_void,
    size: std::os::raw::c_size_t,
    name: *const c_char,
    numOptions: std::os::raw::c_uint,
    options: *mut CUjit_option,
    optionValues: *mut std::os::raw::c_void,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(
            CUlinkState,
            CUjitInputType,
            *mut std::os::raw::c_void,
            std::os::raw::c_size_t,
            *const c_char,
            std::os::raw::c_uint,
            *mut CUjit_option,
            *mut std::os::raw::c_void,
        ) -> CUresult,
    > = LIBCUDA.get(b"cuLinkAddData").unwrap();

    let result = func(
        state, type_, data, size, name, numOptions, options, optionValues,
    );
    eprintln!(
        "cuLinkAddData(state: {:?}, type_: {:?}, data: {:?}, size: {:?}, name: {:?}, numOptions: {:?}, options: {:?}, optionValues: {:?}) -> {:?}",
        state,
        type_,
        data,
        size,
        CStr::from_ptr(name).to_string_lossy(),
        numOptions,
        options.as_ref(),
        optionValues,
        result
    );
    result
}

pub unsafe extern "C" fn cuLinkAddFile(
    state: CUlinkState,
    type_: CUjitInputType,
    path: *const c_char,
    numOptions: std::os::raw::c_uint,
    options: *mut CUjit_option,
    optionValues: *mut std::os::raw::c_void,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(
            CUlinkState,
            CUjitInputType,
            *const c_char,
            std::os::raw::c_uint,
            *mut CUjit_option,
            *mut std::os::raw::c_void,
        ) -> CUresult,
    > = LIBCUDA.get(b"cuLinkAddFile").unwrap();

    let result = func(
        state, type_, path, numOptions, options, optionValues,
    );
    eprintln!(
        "cuLinkAddFile(state: {:?}, type_: {:?}, path: {:?}, numOptions: {:?}, options: {:?}, optionValues: {:?}) -> {:?}",
        state,
        type_,
        CStr::from_ptr(path).to_string_lossy(),
        numOptions,
        options.as_ref(),
        optionValues,
        result
    );
    result
}

pub unsafe extern "C" fn cuLinkComplete(
    state: CUlinkState,
    cubinOut: *mut *mut std::os::raw::c_void,
    sizeOut: *mut std::os::raw::c_size_t,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(
            CUlinkState,
            *mut *mut std::os::raw::c_void,
            *mut std::os::raw::c_size_t,
        ) -> CUresult,
    > = LIBCUDA.get(b"cuLinkComplete").unwrap();

    let result = func(state, cubinOut, sizeOut);
    eprintln!(
        "cuLinkComplete(state: {:?}, cubinOut: {:?}, sizeOut: {:?}) -> {:?}",
        state,
        cubinOut.as_ref(),
        sizeOut.as_ref(),
        result
    );
    result
}

pub unsafe extern "C" fn cuLinkDestroy(
    state: CUlinkState) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(CUlinkState) -> CUresult,
    > = LIBCUDA.get(b"cuLinkDestroy").unwrap();

    let result = func(state);
    eprintln!("cuLinkDestroy(state: {:?}) -> {:?}", state, result);
    result
}

pub unsafe extern "C" fn cuMemGetInfo(
    free: *mut c_ulonglong,
    total: *mut c_ulonglong,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut c_ulonglong, *mut c_ulonglong) -> CUresult,
    > = LIBCUDA.get(b"cuMemGetInfo").unwrap();

    let result = func(free, total);
    eprintln!("cuMemGetInfo(free: {:?}, total: {:?}) -> {:?}", free, total, result);
    result
}

pub unsafe extern "C" fn cuMemAllocManaged(
    dptr: *mut CUdeviceptr,
    size: c_ulonglong,
    flags: c_uint,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUdeviceptr, c_ulonglong, c_uint) -> CUresult,
    > = LIBCUDA.get(b"cuMemAllocManaged").unwrap();

    let result = func(dptr, size, flags);
    eprintln!("cuMemAllocManaged(dptr: {:?}, size: {:?}, flags: {:?}) -> {:?}", dptr, size, flags, result);
    result
}

pub unsafe extern "C" fn cuMemAlloc(
    dptr: *mut CUdeviceptr,
    size: c_ulonglong,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUdeviceptr, c_ulonglong) -> CUresult,
    > = LIBCUDA.get(b"cuMemAlloc").unwrap();

    let result = func(dptr, size);
    eprintln!("cuMemAlloc(dptr: {:?}, size: {:?}) -> {:?}", dptr, size, result);
    result
}

pub unsafe extern "C" fn cuMemAllocPitch(
    dptr: *mut CUdeviceptr,
    pPitch: *mut c_ulonglong,
    WidthInBytes: c_ulonglong,
    Height: c_ulonglong,
    ElementSizeBytes: c_ulonglong,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(
            *mut CUdeviceptr,
            *mut c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
        ) -> CUresult,
    > = LIBCUDA.get(b"cuMemAllocPitch").unwrap();

    let result = func(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
    eprintln!(
        "cuMemAllocPitch(dptr: {:?}, pPitch: {:?}, WidthInBytes: {:?}, Height: {:?}, ElementSizeBytes: {:?}) -> {:?}",
        dptr,
        pPitch,
        WidthInBytes,
        Height,
        ElementSizeBytes,
        result
    );
    result
}

pub unsafe extern "C" fn cuMemFree(dptr: CUdeviceptr) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    > = LIBCUDA.get(b"cuMemFree").unwrap();

    let result = func(dptr);
    eprintln!("cuMemFree(dptr: {:?}) -> {:?}", dptr, result);
    result
}

pub unsafe extern "C" fn cuMemGetAddressRange(
    pbase: *mut *mut c_void,
    psize: *mut c_ulonglong,
    dptr: CUdeviceptr,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut *mut c_void, *mut c_ulonglong, CUdeviceptr) -> CUresult,
    > = LIBCUDA.get(b"cuMemGetAddressRange").unwrap();

    let result = func(pbase, psize, dptr);
    eprintln!(
        "cuMemGetAddressRange(pbase: {:?}, psize: {:?}, dptr: {:?}) -> {:?}",
        pbase,
        psize,
        dptr,
        result
    );
    result
}

pub unsafe extern "C" fn cuMemFreeHost(p: *mut c_void) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void) -> CUresult,
    > = LIBCUDA.get(b"cuMemFreeHost").unwrap();

    let result = func(p);
    eprintln!("cuMemFreeHost(p: {:?}) -> {:?}", p, result);
    result
}

pub unsafe extern "C" fn cuMemHostAlloc(
    pp: *mut *mut c_void,
    bytesize: usize,
    flags: c_uint,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult,
    > = LIBCUDA.get(b"cuMemHostAlloc").unwrap();

    let result = func(pp, bytesize, flags);
    eprintln!("cuMemHostAlloc(pp: {:?}, bytesize: {:?}, flags: {:?}) -> {:?}", pp, bytesize, flags, result);
    result
}

pub unsafe extern "C" fn cuMemHostGetDevicePointer(
    pdptr: *mut CUdeviceptr,
    p: *mut c_void,
    flags: c_uint,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUdeviceptr, *mut c_void, c_uint) -> CUresult,
    > = LIBCUDA.get(b"cuMemHostGetDevicePointer").unwrap();

    let result = func(pdptr, p, flags);
    eprintln!("cuMemHostGetDevicePointer(pdptr: {:?}, p: {:?}, flags: {:?}) -> {:?}", pdptr, p, flags, result);
    result
}

pub unsafe extern "C" fn cuMemHostGetFlags(
    pFlags: *mut c_uint,
    p: *mut c_void,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut c_uint, *mut c_void) -> CUresult,
    > = LIBCUDA.get(b"cuMemHostGetFlags").unwrap();

    let result = func(pFlags, p);
    eprintln!("cuMemHostGetFlags(pFlags: {:?}, p: {:?}) -> {:?}", pFlags, p, result);
    result
}

pub unsafe extern "C" fn cuMemHostRegister(
    p: *mut c_void,
    bytesize: usize,
    flags: c_uint,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void, usize, c_uint) -> CUresult,
    > = LIBCUDA.get(b"cuMemHostRegister").unwrap();

    let result = func(p, bytesize, flags);
    eprintln!("cuMemHostRegister(p: {:?}, bytesize: {:?}, flags: {:?}) -> {:?}", p, bytesize, flags, result);
    result
}

pub unsafe extern "C" fn cuMemHostUnregister(
    p: *mut c_void) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void) -> CUresult,
    > = LIBCUDA.get(b"cuMemHostUnregister").unwrap();

    let result = func(p);
    eprintln!("cuMemHostUnregister(p: {:?}) -> {:?}", p, result);
    result
}

pub unsafe extern "C" fn cuPointerGetAttribute(
    data: *mut c_void,
    attribute: u32,
    ptr: CUdeviceptr,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void, u32, CUdeviceptr) -> CUresult,
    > = LIBCUDA.get(b"cuPointerGetAttribute").unwrap();

    let result = func(data, attribute, ptr);
    eprintln!(
        "cuPointerGetAttribute(data: {:?}, attribute: {:?}, ptr: {:?}) -> {:?}",
        data, attribute, ptr, result
    );
    result
}

pub unsafe extern "C" fn cuPointerGetAttributes(
    numAttributes: u32,
    attributes: *mut u32,
    data: *mut *mut c_void,
    ptrs: *mut CUdeviceptr,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(u32, *mut u32, *mut *mut c_void, *mut CUdeviceptr) -> CUresult,
    > = LIBCUDA.get(b"cuPointerGetAttributes").unwrap();

    let result = func(numAttributes, attributes, data, ptrs);
    eprintln!(
        "cuPointerGetAttributes(numAttributes: {:?}, attributes: {:?}, data: {:?}, ptrs: {:?}) -> {:?}",
        numAttributes, attributes, data, ptrs, result
    );
    result
}

pub unsafe extern "C" fn cuMemAllocAsync(
    dptr: *mut CUdeviceptr,
    bytesize: usize,
    stream: CUstream,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUdeviceptr, usize, CUstream) -> CUresult,
    > = LIBCUDA.get(b"cuMemAllocAsync").unwrap();

    let result = func(dptr, bytesize, stream);
    eprintln!(
        "cuMemAllocAsync(dptr: {:?}, bytesize: {:?}, stream: {:?}) -> {:?}",
        dptr, bytesize, stream, result
    );
    result
}

pub unsafe extern "C" fn cuMemAllocFromPoolAsync(
    dptr: *mut CUdeviceptr,
    pool: CUmemoryPool,
    bytesize: usize,
    stream: CUstream,
) -> CUresult {
    let func: libloading::Symbol<
        unsafe extern "C" fn(*mut CUdeviceptr, CUmemoryPool, usize, CUstream) -> CUresult,
    > = LIBCUDA.get(b"cuMemAllocFromPoolAsync").unwrap();

    let result = func(dptr, pool, bytesize, stream);
    eprintln!(
        "cuMemAllocFromPoolAsync(dptr: {:?}, pool: {:?}, bytesize: {:?}, stream: {:?}) -> {:?}",
        dptr, pool, bytesize, stream, result
    );
    result
}
