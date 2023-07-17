## 2023.07.17当前问题：
在cuda-gdb中执行`break cuDeviceGetAttribute` && `file ./test` && `run`之后，
虽然test.c文件中cuDeviceGetAttribute调用处有声明result变量，但还是无法`p result`，显示结果如下：
> No symbol "result" in current context.

我猜测，可能的原因是：
1. 没有真正断点在cuDeviceGetAttribute调用处；
2. 设置的断点没有停在cuDeviceGetAttribute函数内；
3. 云服务器上没有硬件。

另外，我试着`break cuGetProcAddress_v2`，也是类似的结果。

---

重新试一试LD_PRELOAD，需要的步骤：
1. 封装实际CUDA调用，需要修改test.c文件
2. 创建模拟和重定向函数，也需要修改test.c文件
3. 编译为动态库
4. 使用LD_PRELOAD加载动态库
5. 运行程序,测试效果

