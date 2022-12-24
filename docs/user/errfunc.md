## variance

```cpp
float variance(const vec_batch &out, const vec_batch &label) 
```

返回两个 batch 的数据之差的方差

## sqrtvariance

```cpp
float sqrtvariance(const vec_batch &out, const vec_batch &label) 
```

返回两个 batch 的数据之差的标准差

## crossentropy_2

```cpp
float crossentropy_2(const vec_batch &out, const vec_batch &label) 
```
返回二分类问题对应的交叉熵。

## crossentropy_k

```cpp
float crossentropy_k(const vec_batch &out, const vec_batch &label) 
```

返回 k 分类问题对应的交叉熵。


## chk_2

```cpp
float chk_2(const vec_batch &out, const vec_batch &label)
```

用于二分类问题，返回正确率（为了统一误差函数的结果都是越小越好，这里输出正确率的相反数）。

## chk_k

```cpp
float chk_k(const vec_batch &out, const vec_batch &label)
```
用于 k 分类问题，返回正确率（为了统一误差函数的结果都是越小越好，这里输出正确率的相反数）。
