<style>
.md-typeset hr {
    border-bottom: 0;
    display: flow-root;
    margin: 0;
}
.tabbed-set {
    FONT-VARIANT: JIS04;
    border-radius: .1rem;
    display: flex;
    flex-flow: column wrap;
    margin: 0px !important;
    position: relative;
}
.js .md-typeset .tabbed-labels {
    position: relative;
    line-height: 0.5;
}
</style>

此页部分伪代码图片来源于 <https://blog.csdn.net/qq_39332551/article/details/123440318>

## 基类

---
=== "代码"
    ```cpp
    struct optimizer {
        virtual void upd(vecmap, const vecmap &) = 0;
    };

    template <const int N>
    struct optimizer_holder : public optimizer {
        unordered_map<const float *, vec> V[N];
        template <const int I>
        vec &get(const vecmap &x) {
            auto it = V[I].find(x.data());
            if (it != V[I].end())
                return it->second;
            else
                return V[I][x.data()] = vec::Zero(x.size());
        }
    };
    ```

=== "注解"
    `optimizer`
    : 所有优化器的基类，包含一个虚函数 `void upd(vecmap, const vecmap &)`。这个函数传入两个 `Eigen::Map<VectorXf>` 类型变量，第一个是系数向量，第二个是系数的梯度向量。需要在这个函数里实现权值更新。

    `optimizer_holder<int N>`
    : 一个类，对每一个系数矩阵维护对应的 N 个额外矩阵，可以用于维护动量之类的东西

## sgd

=== "代码"

    ```cpp
    struct sgd : public optimizer_holder<0> {
        float lr, lambda;
        sgd(float lr = 0.01, float lambda = 0) : lr(lr), lambda(lambda) {}
        void upd(vecmap w, const vecmap &gw) { w -= (gw + w * lambda) * lr; }
    };
    ```
=== "注解"
    mini-batch sgd 伪代码：
    ![](https://pic.imgdb.cn/item/63a6fd6208b6830163a84110.jpg)

## nesterov

=== "代码"
    ```cpp
    struct nesterov : public optimizer_holder<1> {
        float alpha, mu, lambda;
        nesterov(float alpha = 0.01, float mu = 0.9, float lambda = 0) : alpha(alpha), mu(mu), lambda(lambda) {}
        void upd(vecmap w, const vecmap &gw) {
            vec &v = get<0>(w);
            w -= mu * v;
            v = mu * v - (gw + lambda * w) * alpha;
            w += (1. + mu) * v;
        }
    };
    ```
=== "注解"
    ![](https://img-blog.csdnimg.cn/5fc28c3fb6d84b40a88bdef99b3998d7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAWmh1b2p1bkNoZW4=,size_20,color_FFFFFF,t_70,g_se,x_16)
    
## Adam

=== "代码"
    ```cpp
    struct adam : public optimizer_holder<2> {
        float lr, rho1, rho2, eps, lambda;
        float mult1, mult2;
        adam(float lr = 0.001, float rho1 = 0.9, float rho2 = 0.999, float eps = 1e-6, float lambda = 0)
            : lr(lr), rho1(rho1), rho2(rho2), eps(eps), lambda(lambda), mult1(1), mult2(1) {}
        void upd(vecmap w, const vecmap &gw) {
            vec &s = get<0>(w), &r = get<1>(w);
            mult1 *= rho1, mult2 *= rho2;
            s = s * rho1 + (gw + w * lambda) * (1. - rho1);
            r = r * rho2 + (gw + w * lambda).cwiseAbs2() * (1. - rho2);
            w.array() -= lr * s.array() / (sqrt(r.array() / (1. - mult2) + eps)) / (1. - mult1);
        }
    };
    ```
=== "注解"
    adam 伪代码：

    ![](https://img-blog.csdnimg.cn/762f7856161045da803fdd237bca1a5e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAWmh1b2p1bkNoZW4=,size_20,color_FFFFFF,t_70,g_se,x_16)
