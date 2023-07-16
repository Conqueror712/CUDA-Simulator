## 2023.07.17当前问题：
在cuda-gdb中执行`break cuDeviceGetAttribute` & `run`之后，
虽然test.c文件中cuDeviceGetAttribute调用处有声明result变量，但还是无法`p result`，显示结果如下：
> No symbol "result" in current context.

我猜测，可能的原因是：
1. 你没有真正断点在cuDeviceGetAttribute调用处；
2. 你设置的断点没有停在cuDeviceGetAttribute函数内。
