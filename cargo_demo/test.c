#include <stdio.h>
#include <cuda.h>

// 函数声明
CUresult mock_cuInit(unsigned int flags);
CUresult actual_cuInit(unsigned int flags);

// 重定向函数
CUresult redirect_cuInit(unsigned int flags) {
    mock_cuInit(flags);
    return actual_cuInit(flags);
}

// 创建 mock_cuInit 函数
CUresult mock_cuInit(unsigned int flags) {
    // 模拟CUDA调用
    printf("mock_cuInit called with flags %d\n", flags);
    return CUDA_SUCCESS;
}

// 创建一个 actual_cuInit 函数，用于调用 cuInit
CUresult actual_cuInit(unsigned int flags) {
    printf("actual_cuInit called with flags %d\n", flags);
    return cuInit(flags);
}

// 修改 cuInitPtr 的类型
typedef CUresult (*cuInitFunc)(unsigned int);

int main() { 
    CUresult result;
    cuInit cuInitPtr;
    
    int pi; 
    int dev;  
    cuDeviceGet(&dev, 0);
    
    char* cuInitStr = (char*)"cuInit";
    
    // 修改 test.c 中 cuInitPtr 的获取和调用
    result = cuGetProcAddress_v2((char*)"redirect_cuInit", (void **)&cuInitPtr, 0, 0, NULL);

    if (result == CUDA_SUCCESS) {      
        cuInitPtr(0);        
    }
      
    result = cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev); 
    
    if (result == CUDA_SUCCESS) {
        result = cuInitPtr(0);
    }
    
    if (result != CUDA_SUCCESS) {
        printf("Error: cuInit failed with error %d\n", result); 
        return -1; 
    }   
     
    printf("cuInit succeeded\n");
    return 0;   
}