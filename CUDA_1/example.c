#include <stdio.h>
#include <dlfcn.h>
#include <cuda_runtime_api.h>

int main(){
    void *handle = dlopen("./libmycuda.so", RTLD_NOW);
    if (handle == NULL){
        fprintf(stderr, "Error: cannot load library: %s\n", dlerror());
        return 1;
    }
    cudaError_t (*my_cudaMallocManaged)(void **, size_t, unsigned int) = dlsym(handle, "my_cudaMallocManaged");
    if (my_cudaMallocManaged == NULL){
        fprintf(stderr, "Error: cannot find function: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }
    void *devPtr;
    size_t size = 1024;
    unsigned int flags = cudaMemAttachGlobal;
    cudaError_t result = my_cudaMallocManaged(&devPtr, size, flags);
    if (result != cudaSuccess){
        fprintf(stderr, "Error: my_cudaMallocManaged failed: %s\n", cudaGetErrorString(result));
        dlclose(handle);
        return 1;
    }
    printf("Allocated device memory at %p\n", devPtr);
    result = cudaFree(devPtr);
    if (result != cudaSuccess){
        fprintf(stderr, "Error: cudaFree failed: %s\n", cudaGetErrorString(result));
        dlclose(handle);
        return 1;
    }
    dlclose(handle);
    return 0;
}
