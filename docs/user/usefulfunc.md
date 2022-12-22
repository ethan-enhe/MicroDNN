## 定义的常量，以及数据类型的别名

```cpp

using vec = VectorXf;
using mat = MatrixXf;
using ten0 = Tensor<float, 0>;
using ten1 = Tensor<float, 1>;
using ten2 = Tensor<float, 2>;
using ten3 = Tensor<float, 3>;
using ten4 = Tensor<float, 4>;
using vecmap = Map<vec>;
using matmap = Map<mat>;
using ten3map = TensorMap<ten3>;
using vec_batch = vector<vec>;
using batch = pair<vec_batch, vec_batch>;
const float INF = 1e8;
const float EPS = 1e-8;
```

## make_vec

```cpp
vec make_vec(const vector<double> &x) 
```
把 `std::vector<double>` 转换为 `VectorXf`

## to_vecmap

```cpp
vecmap to_vecmap(T &x)
```
返回任意一个 Eigen 对象对应的 `Map<vec>`

## 随机数相关

### rd

```cpp
float rd(float l, float r)
```
返回 $l\sim r$ 中的随机数。

### nd

```cpp
float nd(float x, float y)
```
按照平均值为 x，标准差为 y 的高斯分布生成随机数。
