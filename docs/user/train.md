
## sequential

`sequential` 封装了把多个网络层首尾相接连在一起时的一些操作：

成员：

`vector<layerp> layers`
: 一个 `vector`，存放每层的智能指针。

`void add(const layerp &x)`
: 在当前网络的最后再加一层，参数是这一层的智能指针

`string shape()`
: 返回当前网络的形状

`void set_train_mode(const bool &new_train_mod)`
: 设置是训练还是推理模式，对于具有批归一化层的网络是必须要设置的。

`vec_batch forward(const vec_batch &input)`
: 传入一个 batch 的输入，进行正向传播，并返回一个 batch 的输出

`vec_batch backward(const vec_batch &label)`
: 传入一个 batch 的 label，进行反向传播，返回关于输入的梯度

`void upd(optimizer &opt, const batch &data)`
: 传入一个优化器和一个 batch 的训练数据，优化一轮网络参数

`void writef(const string &f)`
: 写入网络参数到文件

`void readf(const string &f)`
: 从文件读取网络参数 

`vec_batch &out()`
: 返回最后一层网络输出的引用

## data_set

初始化 `data_set(const batch &all_data)`
: 传入全部数据，按照 6:1 分为 train 和 validate

`batch get_train_batch(int batch_sz)`
: 返回一个 batch 的随机训练数据

`batch get_valid_batch(int batch_sz)`
: 返回一个 batch 的随机 validate 数据


## upd

```cpp
void upd(optimizer &opt, const data_set &data, net &net, int batch_sz, int epoch,
         function<float(const vec_batch &, const vec_batch &)> err_func, const string &save_file = "") 
```

- `opt` 优化器
- `data` 数据库
- `net` 要训练的神经网络
- `batch_sz` 顾名思义
- `epoch` 训练次数
- `err_func` 误差函数，用于训练过程的输出，以及用来判断是否存储权值系数。
- `save_file` 训练中会自动把在 valid 数据集上用 `err_func` 算出来的，误差最小的网络系数保存到这个文件中。如果不填这个参数，则不会自动保存.

函数输出：

每隔 50 次，函数会输出在训练过程中计算的 train loss。每隔 1000 次，函数会输出网络在整个

