#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda_runtime_api.h>

typedef cudaError_t (*cudaMallocManaged_t)(void **devPtr, size_t size, unsigned int flags);

cudaError_t my_cudaMallocManaged(void **devPtr, size_t size, unsigned int flags){
    printf("cudaMallocManaged called with size %zu and flags %u\n", size, flags);
    void *handle = dlopen("libcuda.so", RTLD_NOW);
    if (handle == NULL){
        fprintf(stderr, "Error: cannot load CUDA library: %s\n", dlerror());
        exit(1);
    }
    cudaMallocManaged_t original_cudaMallocManaged = (cudaMallocManaged_t)dlsym(handle, "cudaMallocManaged");
    if (original_cudaMallocManaged == NULL){
        fprintf(stderr, "Error: cannot find function cudaMallocManaged: %s\n", dlerror());
        dlclose(handle);
        exit(1);
    }
    cudaError_t result = original_cudaMallocManaged(devPtr, size, flags);
    printf("cudaMallocManaged returned %d\n", result);
    dlclose(handle);
    return result;
}

__attribute__((constructor))
void initialize(void){
    cudaMallocManaged_t original_cudaMallocManaged = NULL;
    void *handle = dlopen("libcuda.so", RTLD_NOW);
    if (handle == NULL){
        fprintf(stderr, "Error: cannot load CUDA library: %s\n", dlerror());
        exit(1);
    }
    original_cudaMallocManaged = (cudaMallocManaged_t)dlsym(handle, "cudaMallocManaged");
    if (original_cudaMallocManaged == NULL){
        fprintf(stderr, "Error: cannot find function cudaMallocManaged: %s\n", dlerror());
        dlclose(handle);
        exit(1);
    }
    cudaMallocManaged_t new_cudaMallocManaged = my_cudaMallocManaged;
    if (new_cudaMallocManaged != original_cudaMallocManaged){
        printf("Interposing cudaMallocManaged\n");
        void *devPtr;
        size_t size = 1024;
        unsigned int flags = cudaMemAttachGlobal;
        cudaError_t result = new_cudaMallocManaged(&devPtr, size, flags);
        if (result != cudaSuccess){
            fprintf(stderr, "Error: cudaMallocManaged interpose failed: %s\n", cudaGetErrorString(result));
            exit(1);
        }
    }
    dlclose(handle);
}

int main(){
    printf("Hello, world!\n");
    return 0;
}
