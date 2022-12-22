
1. 安装 Eigen 3.4.0（[下载链接](https://eigen.tuxfamily.org/index.php?title=Main_Page)），并将 Eigen 和 Unsupported 文件夹复制到编译器可以识别到的头文件目录中。
1. 下载 `microdnn.h` （[链接](https://raw.githubusercontent.com/ethan-enhe/ANN/master/include/microdnn.h)）并放到需要的目录中。
1. 在程序中引入 `microdnn.h`，以下将以一个示例程序说明 MicroDNN 库的用法。

## 例子

下面我们将实现如下一个简单的网络，来实现判断一个二维坐标（范围在 $0\sim1$）是否在外径为 $1$，内径为 $0.5$ 的圆环内。


```cpp title='demo.cpp'
#include <bits/stdc++.h>
#include "../include/microdnn.h"
using namespace std;
int main() {
    sequential net;
    net.add(make_shared<linear>(2, 10));
    net.add(make_shared<relu>());
    net.add(make_shared<linear>(10, 10));
    net.add(make_shared<relu>());
    net.add(make_shared<linear>(10, 1));
    net.add(make_shared<sigmoid>());

    batch dat;
    for (int i = 1; i <= 10000; i++) {
        float x = rd(0, 1), y = rd(0, 1);
        float label = 0;
        if (x * x + y * y >= 0.5 * 0.5 && x * x + y * y <= 1 * 1) label = 1;
        dat.first.push_back(make_vec({x, y}));
        dat.second.push_back(make_vec({label}));
    }
    data_set divided_dat(dat);

    adam opt;
    upd(opt, divided_dat, net, 32, 10000, chk_2, "model.txt");

    return 0;
}
```

如果配置正确，编译运行后你应该看到类似下文的输出：

```text
...

Time elapse: 0.183945
Epoch: 9900
Loss: -0.979377
-------------------------
Time elapse: 0.184759
Epoch: 9950
Loss: -0.973472
-------------------------
Time elapse: 0.18553
Epoch: 10000
Loss: -0.978143
!! Error: -0.988024
```

## 解释

接下来，我们将逐行解释这些代码的意思

```cpp
#include <bits/stdc++.h>
#include "../include/microdnn.h"
using namespace std;
int main() {
```

引入头文件，定义 `main` 函数，没有什么特别的。

```cpp
    sequential net;
    net.add(make_shared<linear>(2, 10));
    net.add(make_shared<relu>());
    net.add(make_shared<linear>(10, 10));
    net.add(make_shared<relu>());
    net.add(make_shared<linear>(10, 1));
    net.add(make_shared<sigmoid>());
```
这里定义了一个 `sequential` 类型的变量。这是 MicroDNN 库中定义的一个类，支持把不同的层首尾依次相接。比如代码里定义的就是一个一次由线性层 + Relu + 线性层 + Relu + 线性层 + Sigmoid 的网络。因为这是一个二分类任务，所以最后一层使用 Sigmoid 当激活函数。 

```cpp
    batch dat;
    for (int i = 1; i <= 10000; i++) {
        float x = rd(0, 1), y = rd(0, 1);
        float label = 0;
        if (x * x + y * y >= 0.5 * 0.5 && x * x + y * y <= 1 * 1) label = 1;
        dat.first.push_back(make_vec({x, y}));
        dat.second.push_back(make_vec({label}));
    }
    data_set divided_dat(dat);
```
接下来这些代码是用来生成数据的，`batch` 是库中定义的，等价于 `pair<VectorXf,VectorXf>`，`VectorXf` 是 Eigen 中定义的向量类。这里通过 MicroDNN 中定义的函数 `make_vec()` 将一个 `std::vector<float>` 转化为 `VectorXf`。

此后，又用 `dat` 初始化了一个 `data_set` 类型的变量。这是 MicroDNN 实现的一个建议数据库，会自动把初始化参数中的数据分成 train 和 valid 两个数据集，同时支持随机取出一个 Batch 的数据等操作。如果你不满足于这个数据库的功能，你也可以自己实现一个。

```cpp
    adam opt;
    upd(opt, divided_dat, net, 32, 10000, chk_2, "model.txt");

    return 0;
}
```

定义了一个 adam 优化器，MicroDNN 中实现了 adam，nesterov，sgd 等优化器，你也可以自己实现优化器，详见开发者文档。

随后调用 MicroDNN 定义的优化函数 `opt` 优化网络参数，这些参数分别是优化器、数据库、网络、batch size、训练轮数、输出中用到的误差函数、模型输出的文件名。
