## sgd

```cpp
sgd(float lr = 0.01, float lambda = 0)
```

- `lr` 学习率
- `lambda` L2 正则化中的参数，默认为 0，即不进行正则化

## nesterov

```cpp
nesterov(float alpha = 0.01, float mu = 0.9, float lambda = 0)
```

- `alpha` nesterov 中类似学习率的参数
- `mu` 与动量相关的参数
- `lambda` L2 正则化中的参数，默认为 0，即不进行正则化

## adam

```cpp
adam(float lr = 0.001, float rho1 = 0.9, float rho2 = 0.999, float eps = 1e-6, float lambda = 0)
```

- `lr` 学习率
- `rho1`, `rho2` adam 中的参数
- `eps` 平滑因子
- `lambda` L2 正则化中的参数，默认为 0，即不进行正则化
