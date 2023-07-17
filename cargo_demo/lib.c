#include <cuda.h>

CUresult mock_cuInit(unsigned int flags);
CUresult actual_cuInit(unsigned int flags);

// 重定向函数
CUresult redirect_cuInit(unsigned int flags) {
    mock_cuInit(flags);
    return actual_cuInit(flags);
}