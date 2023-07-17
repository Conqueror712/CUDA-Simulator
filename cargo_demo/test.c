#include <stdio.h>
#include <cuda.h>

typedef CUresult (*cuInitFunc)(unsigned int);

int main() { 
    CUresult result;
    cuInitFunc cuInitPtr;
    
    int pi; 
    int dev;  
    cuDeviceGet(&dev, 0);
    
    char* cuInitStr = (char*)"cuInit";
    
    result = cuGetProcAddress_v2(cuInitStr, (void **)&cuInitPtr, 0, 0, NULL);
    if (result != CUDA_SUCCESS) {
        printf("Error: cuGetProcAddress_v2 failed with error %d\n", result);
        return -1;  
    }
      
    result = cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev); 
    
    if (result == CUDA_SUCCESS) {
        result = cuInitPtr(0);
        // result = cuInitPtr(0x1);    // 添加 CUDA_IPC_ENABLE 标志
        // result = cuInitPtr(0x2); // 添加 CUDA_SCHEDULE_AUTO 标志
        // result = cuInitPtr(0x1 | 0x2);  // 同时启用IPC和自动线程调度
    }
    
    if (result != CUDA_SUCCESS) {
        printf("Error: cuInit failed with error %d\n", result); 
        return -1; 
    }   
     
    printf("cuInit succeeded\n");
    return 0;   
}