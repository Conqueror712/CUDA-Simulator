// #include <stdio.h>

// __global__ void vectorAdd(int *a, int *b, int *c, int n) {
//     int i = threadIdx.x;
//     if (i < n) {
//         c[i] = a[i] + b[i];
//     }
// }

// extern "C" void launchKernel(int *a, int *b, int *c, int n) {
//     int *d_a, *d_b, *d_c;
//     cudaMalloc(&d_a, n * sizeof(int));
//     cudaMalloc(&d_b, n * sizeof(int));
//     cudaMalloc(&d_c, n * sizeof(int));
//     cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);
//     vectorAdd<<<1, n>>>(d_a, d_b, d_c, n);
//     cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_c);
// }

#include <stdio.h>

__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

/*
nvcc -ptx test.cu
g++ -o test test.c -L/path/to/cuda/lib -lcudart -lcuda -ldl -I/path/to/cuda/include
其中，/path/to/cuda/lib 是 CUDA 库文件的路径，
/path/to/cuda/include 是 CUDA 头文件的路径。
编译完成后，你可以运行可执行文件 test 来测试 cuGetProcAddress_v2 函数的正确性。

g++ -o your_executable your_test_program.cpp -L/path/to/cuda/lib -lcudart -lcuda -ldl -I/path/to/cuda/include
其中，your_test_program.cpp 是你的测试程序的源代码文件名，
your_executable 是你要编译的可执行文件名，/path/to/cuda/lib 是 CUDA 库文件的路径，
/path/to/cuda/include 是 CUDA 头文件的路径。
*/

// nvcc -ptx test.cu
// g++ -o test test.c -L/path/to/cuda/lib -lcudart -lcuda -ldl -I/path/to/cuda/include