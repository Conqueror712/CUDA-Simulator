## 2023.07.16当前问题：
在cuda-gdb中执行`break cuDeviceGetAttribute` && `file ./test` && `run`之后，
虽然test.c文件中cuDeviceGetAttribute调用处有声明result变量，但还是无法`p result`，显示结果如下：
> No symbol "result" in current context.

我猜测，可能的原因是：
1. 没有真正断点在cuDeviceGetAttribute调用处；
2. 设置的断点没有停在cuDeviceGetAttribute函数内；
3. 云服务器上没有硬件。

另外，我试着`break cuGetProcAddress_v2`，也是类似的结果。

## 2023.07.17当前问题：
> C++版本
1. `g++ -o test test.c -L/usr/local/cuda-12.2/lib64 -lcudart -lcuda -ldl -I/usr/local/cuda-12.2/include`
2. `gcc -fPIC -shared -o libredirect.so lib.c`
3. `LD_PRELOAD=/home/lighthouse/ospp/CUDA-Practice/cargo_demo/libredirect.so` --> `./test`报错Segmentation fault (core dumped)

> Rust版本
`LD_PRELOAD=/home/lighthouse/ospp/CUDA-Practice/cargo_demo/target/release/libcuda.so` --> `./test`报错Segmentation fault (core dumped)
By the way, `sudo ./test`报错Segmentation fault，即没有(core dumped)了...

## 2023.08.04：
上述问题已解决，正在着手编写需要的CUDA函数的Rust实现。
