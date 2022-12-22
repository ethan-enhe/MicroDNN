
## 最后一层

对于一些特定的任务，最后一层一般会采用特定的网络层，才可以在使用 `sequntial` 时得到正确的对于误差函数的梯度。

### sigmoid

```cpp
net.add(make_shared<sigmoid>());
```

最经典的神经网络激活函数。MicroDNN 中初始化 sigmoid 层不需要参数。

对于二分类任务，要求最后一层采用 sigmoid 层，将输出映射到 $0\sim 1$，同时也可以获得正确的梯度。

### same

```cpp
net.add(make_shared<same>());
```

MicroDNN 中初始化 same 层不需要参数。

对于拟合任务，要求最后一层采用 same 层，同时才能获得正确的梯度。

### softmax

```cpp
net.add(make_shared<softmax>());
```

MicroDNN 中初始化 softmax 层不需要参数。目前，softmax 层不支持反向传播。

对于 k 分类任务，要求最后一层采用 softmax 层，同时才能获得正确的梯度。


## linear

```cpp
net.add(make_shared<linear>(in_sz,out_sz));
```

- `in_sz` 是上一层网络输出向量的维数
- `out_sz` 是这个线性层输出向量的维数

## batchnorm

```cpp
net.add(make_shared<batchnorm>(sz,momentum));
```

- `sz` 上一层输出向量的维数
- `momentum` 移动平均的动量参数

!!! note "batchnorm 层状态的切换"
    众所周知，在训练时，batchnorm 层会不断更新移动平均和移动方差。而推理时则不应该更新移动平均。为此，网络层的基类中有一个 `set_train_mode(0/1)` 函数，可以更改网络层的状态（0 代表推理阶段，1 代表训练阶段）


## convolution

```cpp
net.add(make_shared<conv>(in_ch, out_ch, in_row, in_col, kernel_row, kernel_col));
```

- `in_ch` 输入通道数
- `out_ch` 输出通道数
- `in_row` 输入图像的行数
- `in_col` 输入图像的列数
- `kernel_row` 卷积核的行数
- `kernel_col` 卷积核的列数

注意，卷积层输入输出仍然是 `VectorXf`，但是是把 3 维张量 flatten 之后得到的结果。因此，输出是 $32\times 28\times 28$ 的卷积层实际上输出的就是 $32\times 28\times 28$ 维度的向量


## maxpool2x2

```cpp
net.add(make_shared<maxpool2x2>(in_ch, in_row, in_col));
```
窗口为 $2\times 2$ 的最大池化，步长也为2.

- `in_ch` 输入通道数
- `in_row` 输入图像的行数
- `in_col` 输入图像的列数

## 激活函数

激活函数初始化时均没有参数。

### sigmoid

```cpp
net.add(make_shared<sigmoid>());
```


### tanh
```cpp
net.add(make_shared<th>());
```

### relu
```cpp
net.add(make_shared<relu>());
```

### hardswish
```cpp
net.add(make_shared<hardswish>());
```

### swish
```cpp
net.add(make_shared<swish>());
```

### mish
```cpp
net.add(make_shared<mish>());
```
