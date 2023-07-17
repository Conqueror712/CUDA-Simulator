#include <cuda.h>

CUresult mock_cuInit(unsigned int flags);
CUresult actual_cuInit(unsigned int flags);

// 重定向函数
CUresult redirect_cuInit(unsigned int flags) {
    mock_cuInit(flags);
    return actual_cuInit(flags);
}

// mock_cuInit 函数
CUresult mock_cuInit(unsigned int flags) {
    // 模拟CUDA调用
    printf("mock_cuInit called with flags %d\n", flags);
    return CUDA_SUCCESS;
}

// actual_cuInit 函数，用于调用 cuInit
CUresult actual_cuInit(unsigned int flags) {
    printf("actual_cuInit called with flags %d\n", flags);
    return cuInit(flags);
}