## Welcome to my first CUDA project!
这是我的第一份CUDA程序，用于测试环境，以及对比CPU下和GPU下的运行速度差别，建立Baseline

# 零、前言：

📕**欢迎访问**：

> 个人博客：[conqueror712.github.io/](https://link.juejin.cn?target=https%3A%2F%2Fconqueror712.github.io%2F)
>
> 知乎：[www.zhihu.com/people/soeu…](https://link.juejin.cn?target=https%3A%2F%2Fwww.zhihu.com%2Fpeople%2Fsoeur712%2Fposts)
>
> Bilibili：[space.bilibili.com/57089326](https://link.juejin.cn?target=https%3A%2F%2Fspace.bilibili.com%2F57089326)
>
> 掘金：[juejin.cn/user/129787…](https://juejin.cn/user/1297878069809725/posts)

有任何疏忽和错误欢迎各位读者指出！

# 一、一切从C++开始！

首先我们要得到一份C++代码，功能很简单，就是做加法：

注意这是Linux版本

```CPP
#include <iostream>
#include <math.h>
#include <sys/time.h>

// function to add the elements of two arrays
void add(int n, float *x, float *y){
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(){
  int N = 1<<25; // 30M elements

  float *x = new float[N];
  float *y = new float[N];
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  struct timeval t1,t2;
  double timeuse;
  gettimeofday(&t1,NULL);
  // Run kernel on 30M elements on the CPU
  void add(N, x, y);
  gettimeofday(&t2,NULL);
  timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000.0;

  std::cout << "add(int, float*, float*) time: " << timeuse << "ms" << std::endl;
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete [] x;
  delete [] y;
  
  return 0;
}
```

添加一些有趣的东西就可以变成CUDA代码：

- `__global__`
- `cudaMallocManaged(&x, N*sizeof(float));`
    `cudaMallocManaged(&y, N*sizeof(float));`
- `add<<<1, 1>>>(N, x, y);`
- `cudaFree(x);`
    `cudaFree(y);`

## 1. Linux版本：

```CPP
#include <iostream>
#include <math.h>
#include <sys/time.h>

// function to add the elemsents of two arrays
__global__
void add(int n, float *x, float *y){
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(){
  int N = 1<<25; // 30M elements

  float *x = new float[N];
  float *y = new float[N];
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  struct timeval t1,t2;
  double timeuse;
  gettimeofday(&t1,NULL);
  // Run kernel on 30M elements on the CPU
  add<<<1, 1>>>(N, x, y);
  gettimeofday(&t2,NULL);
  timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000.0;

  std::cout << "add(int, float*, float*) time: " << timeuse << "ms" << std::endl;
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete [] x;
  delete [] y;

  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
```

## 2. Windows版本：

```CPP
#include <iostream>
#include <math.h>
#include <windows.h>

// function to add the elements of two arrays
__global__
void add(int n, float *x, float *y){
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(){
  int N = 1<<25; // 30M elements

  float *x = new float[N];
  float *y = new float[N];
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  ULONGLONG t1, t2, freq;
  double timeuse;
  QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
  QueryPerformanceCounter((LARGE_INTEGER*)&t1);
  // Run kernel on 30M elements on the CPU
  add<<<1, 1>>>(N, x, y);
  QueryPerformanceCounter((LARGE_INTEGER*)&t2);
  timeuse = (double)(t2 - t1) / (double)freq * 1000.0;

  std::cout << "add(int, float*, float*) time: " << timeuse << "ms" << std::endl;
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete [] x;
  delete [] y;

  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
```

# 二、一份CUDA代码如何在CPU上跑起来？

这个很简单，直接运行就可以了。

![avatar](https://cdnjson.com/images/2023/07/02/RLBHC1A79UNE645CHGQ.jpg)

# 三、一份CUDA代码如何在NVidia-GPU上跑起来？

首先要保证电脑内的CUDA环境，这里我们只说1个Linux下配置CUDA环境的细节：

- 下载CUDA的时候如果说有几个软件包无法下载，不妨加上`-- fix-missing`试一试，即`sudo apt install nvidia-cuda-toolkit --fix-missing`；

然后按照如图所示的方式就可以跑起来啦！可以看到明显快了很多（笔者是GTX1650，附上截图）。

![avatar](https://cdnjson.com/images/2023/07/02/WR4OPCZDJZXNSU6B.png)

![avatar](https://cdnjson.com/images/2023/07/02/imagea80bb0bd9e073ba0.png)

# 四、一份CUDA代码如何在ZLUDA上跑起来？

ZLUDA仓库：https://github.com/vosen/ZLUDA

这个比前面的复杂一些，网上的资料也比较少，但是经过一番探索还是摸索出来了方法：

我们需要的前置环境：

- Visual Studio 2019（对没错，必须是2017~2019的版本才行，笔者一开始下了一个2022的结果不行）
- Rust（下载最简单的版本就可以了，可以用`cargo --version`和`rustc --version`来检查是否安装成功）
- Visual Studio的`cl.exe`需要添加至环境变量，具体这个文件在哪可以使用Everything搜索
- Clone上述ZLUDA仓库到本地并编译，编译过程下文会展示

![avatar](https://cdnjson.com/images/2023/07/02/CST4_0EOPOSW8GYUUG6.png)

> 如上这种编译错误就是没有VS工具链导致的👆

## 1. ZLUDA前置环境如何编译？

首先要进入ZLUDA安装目录，打开终端后执行`cargo build --release`

## 2. 如何把.cu文件编译成.exe文件呢？

只需要进入我们的项目目录下执行：`nvcc -o my_app.exe my_app.cu`

例如笔者就是：`nvcc -o hello.exe windows-hello.cu`

> Linux：

`LD_LIBRARY_PATH=<ZLUDA_DIRECTORY> <APPLICATION> <APPLICATIONS_ARGUMENTS>`

## 3. 如何运行我们的代码？

按理来说随便进入一个目录就行，不过如果出错的话还是在ZLUDA目录下执行如下代码：

> Windows：

`<ZLUDA_DIRECTORY>\zluda_with.exe -- <APPLICATION> <APPLICATIONS_ARGUMENTS>`

最后一项参数实测**可以为空**，`<APPLICATION>`是你代码的`.exe`可执行文件，例如笔者就是：

`D:\My_Files\Coding-Project-2023\OSPP\OSPP-THU-CUDA\Start\ZLUDA\ZLUDA\target\release\deps\zluda_with.exe -- hello.exe`

![avatar](https://cdnjson.com/images/2023/07/02/image.png)

比单纯的GPU还快！虽然这里没有控制变量，要控制的话应该在Linux下去测试，但是还是肉眼可见的快。
