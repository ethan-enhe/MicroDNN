
1. 安装 Eigen 3.4.0（[下载链接](https://eigen.tuxfamily.org/index.php?title=Main_Page)），并将 Eigen 和 Unsupported 文件夹复制到编译器可以识别到的头文件目录中。
1. 下载 `net2.h` （[链接](https://raw.githubusercontent.com/ethan-enhe/ANN/master/include/net2.h)）并放到需要的目录中。
1. 在程序中引入 `net2.h`，以下将以一个示例程序说明 MicroDNN 库的用法。

## 例子：

下面我们将实现如下一个简单的网络，来实现判断一个二维坐标（范围在 $0\sim1$）是否在外径为 $1$，内径为 $0.5$ 的圆环内。

[网络结构示意图]

```cpp title='demo.cpp'
#include "../include/net2.h"

#include <bits/stdc++.h>
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
