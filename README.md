# 零、前言

此demo的Github仓库（持续更新）：<https://github.com/Conqueror712/CUDA-Practice>

📕**欢迎访问**：

> 个人博客：[conqueror712.github.io/](https://link.juejin.cn?target=https%3A%2F%2Fconqueror712.github.io%2F)
>
> 知乎：[www.zhihu.com/people/soeu…](https://link.juejin.cn?target=https%3A%2F%2Fwww.zhihu.com%2Fpeople%2Fsoeur712%2Fposts)
>
> Bilibili：[space.bilibili.com/57089326](https://link.juejin.cn?target=https%3A%2F%2Fspace.bilibili.com%2F57089326)
>
> 掘金：[juejin.cn/user/129787…](https://juejin.cn/user/1297878069809725/posts)

有任何疏忽和错误欢迎各位读者指出！

欢迎Star本项目repo⭐：<https://github.com/Conqueror712/CUDA-Simulator/tree/main/cargo_demo>

# 一、Linux环境选择

首先准备一个带NVIDIA GPU的Linux环境；

笔者这里用的是WSL，N卡是原本Windows上带的。

*   Ubuntu 20.04 LTS
*   RTX 1650

> 尝试过的其他方法：
>
> *   Windows环境（兼容性问题）
> *   Vmware虚拟机环境（虚拟化过程繁琐）
> *   单主机双系统（切换不便，卡顿）
> *   云服务器（成本问题，有些驱动版本需要更新）
> *   Mac环境（不支持CUDA）

# 二、NVIDIA驱动安装

直接用Windows下载.exe文件安装；

另外，安装过程会伴随多次黑屏和重启，并且会需要一段时间，请耐心等待，会自动同步到WSL中；

> 如果没同步就请`apt install nvidia-utils-535-server`

值得一提的是，安装之前需要把旧版的带NVIDIA的软件全部卸载（除了NVIDIA Control Panel）

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b3d5beaf549e4132992979a706d3110f~tplv-k3u1fbpfcp-watermark.image?)

作为参考，笔者的驱动版本是536.67

> check: `nvidia-smi`
>
> ![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e1e603154d6e442e8d38a39790e2d95f~tplv-k3u1fbpfcp-watermark.image?)
# 三、更新依赖

    sudo apt update
    sudo apt-get update
    sudo apt-get install libclang-dev
    sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev

获取权限（可选）：`sudo su`

CUDA Toolkit下载：`sudo apt install nvidia-cuda-toolkit --fix-missing`

# 四、CUDA 12.2下载安装：

## 方式一：

推荐基于发行版的文档进行安装，这里是Ubuntu，参考文档如下：

<https://help.ubuntu.com/community/NvidiaDriversInstallation>

## 方式二：

直接安装：

      wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda

更新环境变量`nano ~/.bashrc` --> 以下操作 --> `source ~/.bashrc`

> 注意，这一步如果原本机器上没有旧版的CUDA就不用做

    export CUDA_HOME=/usr/local/cuda
    export PATH=$PATH:$CUDA_HOME/bin
    export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

注：退出nano的方法可以是`Ctrl + X, Y, Enter`（三步）

> check: `nvcc --version`
>
> ![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/75208dd1dbe7446196a243e56dc4e58c~tplv-k3u1fbpfcp-watermark.image?)
# 五、Rust & Cargo下载

`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`之后输入`1`；

随后重新加载`source $HOME/.cargo/env`；

> check: `cargo --version` && `rustc --version`
>
> ![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/47b0d93c12c1454b9d20fda8e583febd~tplv-k3u1fbpfcp-watermark.image?)
***

> （此步骤不必要）添加config文件配置镜像源，具体位置在/home/.cargo/config，配置文件e.g.如下：
>
>     [source.crates-io]
>     registry = "https://github.com/rust-lang/crates.io-index"
>     # 指定镜像
>     replace-with = 'tuna'
>     # 清华大学
>     [source.tuna]
>     registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git"

# 六、克隆仓库 + 环境变量

`git clone https://github.com/Conqueror712/CUDA-Simulator.git`

配置环境变量，用nano写入`nano ~/.bashrc`；

写入后保存`source ~/.bashrc`（这一步有时不做也可以，但推荐做）：

    export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda-12.2/bin:$PATH
    export LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu/
    export BINDGEN_EXTRA_CLANG_ARGS="-I /usr/local/cuda-12.2/include"

# 七、生成动态链接库

于项目根目录下`cargo build --release`，生成libcuda.so文件，默认在target/release/


![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/06e88351e7f24d39829cdadef7112cbf~tplv-k3u1fbpfcp-watermark.image?)

# 八、简单测试

1.  cd进入smoketest，用`nvcc smoketest.cu`编译得到`a.out`，再运行`./a.out`

    ![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/bf7c94df9cd948fa84fd41f95d14b921~tplv-k3u1fbpfcp-watermark.image?)

2.  无错误之后方可`LD_PRELOAD=/home/<username>/CUDA-Simulator/cargo_demo/target/release/libcuda.so ./a.out`查看trace（已经实现的会显示..... --> CUDA\_SUCCESS）

    ![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/304ecd2ad37e4ea79a58f14177393fab~tplv-k3u1fbpfcp-watermark.image?)

3.  `cargo doc --open`在浏览器中打开文档查看函数签名

    ![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/eadc237df6f9445999b22761abab2b88~tplv-k3u1fbpfcp-watermark.image?)

4.  之后往`lib.rs中添加自己的实现即可`（未完待续，敬请期待下篇）

***

FIN
