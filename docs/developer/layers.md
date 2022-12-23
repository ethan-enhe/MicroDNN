
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
## 基类

=== "代码"

    ```cpp
    struct layer {
        string name;
        vec_batch out, grad;
        int batch_sz;
        bool train_mode;

        layer(const string &name) : name(name), batch_sz(0), train_mode(1) {}
    ```

=== "注解"
    - `name` 是这层的类型和形状的信息。
    - `out` 和 `grad` 保存的是每一个 batch 的输出和梯度。
    - `batch_sz` 是当前训练时的 batch size
    - `train_mode` 是目前的模式，初始设为训练模式

---
=== "代码"

    ```cpp
        // 更改batch-size 和 trainmode 选项
        void set_train_mode(const bool &new_train_mode) { train_mode = new_train_mode; }
        virtual void _resize(int){};
        void resize(int new_batch_sz) {
            if (batch_sz != new_batch_sz) {
                batch_sz = new_batch_sz;
                out.resize(batch_sz);
                grad.resize(batch_sz);
                _resize(batch_sz);
            }
        }
    ```

=== "注解"
    然后就是一些用于更改这层网络属性的函数：

    - `set_train_mode` 用来修改目前的模式
    - `resize` 用来更改 `batch_sz`
    - `_resize` 是一个回调函数，对于自定义的网络层，如果需要根据 `batch_sz` 的大小而调整某些东西，可以重写这个函数。

---
=== "代码"

    ```cpp
        virtual void forward(const vec_batch &in) {}
        virtual void backward(const vec_batch &, const vec_batch &) = 0;

        virtual void clear_grad() {}
        virtual void write(ostream &io) {}
        virtual void read(istream &io) {}
        virtual void upd(optimizer &opt) {}
    };
    ```

=== "注解"
    - `forward` 用来前向传播
    - `backward` 用来反向传播，其中第一个参数是上一层网络的输出，第二个参数是下一层网络返回的梯度
    - `clear_grad` 里写每次反向传播，把梯度数组清零的数组
    - `read` 用于从一个输入流读取网络参数，`write` 用于把网络参数输出到一个流，注意传入的流都是二进制流
    - `upd` 传入一个优化器，用这个优化器来优化网络参数


## 线性层

---
=== "代码"
    ```cpp
    struct linear : public layer {
        const int in_sz, out_sz;
        // 系数，系数梯度
        mat weight, grad_weight;
        vec bias, grad_bias;
        linear(int in_sz, int out_sz)
            : layer("linear " + to_string(in_sz) + " -> " + to_string(out_sz))
            , in_sz(in_sz)
            , out_sz(out_sz)
            , weight(out_sz, in_sz)
            , grad_weight(out_sz, in_sz)
            , bias(out_sz)
            , grad_bias(out_sz) {
            bias.setZero();
            for (auto &i : weight.reshaped()) i = nd(0, sqrt(2. / in_sz));
        }
        void forward(const vec_batch &in) {
            for (int i = 0; i < batch_sz; i++) out[i] = weight * in[i] + bias;
        }
    ```
=== "注解"
    - `weight` 和 `grad_weight` 分别为权值矩阵和权值矩阵的梯度。
    - `bias` 和 `bias_weight` 分别为梯度向量和梯度向量的梯度。

    默认采用 Kaiming He 初始化初始化系数：
    $$
        W \sim \mathcal N(0,\sqrt {\frac 2{\text{输入向量长度}}})
    $$

    线性层的正向传播：
    $$
        O=W \cdot I + B
    $$


---
=== "代码"
    ```cpp
        void clear_grad() {
            grad_weight.setZero();
            grad_bias.setZero();
        }
        void backward(const vec_batch &in, const vec_batch &nxt_grad) {
            for (int i = 0; i < batch_sz; i++) {
                grad_bias += nxt_grad[i];
                grad_weight += nxt_grad[i] * in[i].transpose();
                grad[i] = weight.transpose() * nxt_grad[i];
            }
            grad_weight /= batch_sz;
            grad_bias /= batch_sz;
        }
        void upd(optimizer &opt) {
            opt.upd(to_vecmap(weight), to_vecmap(grad_weight));
            opt.upd(to_vecmap(bias), to_vecmap(grad_bias));
        }
        void write(ostream &io) {
            for (auto &i : weight.reshaped()) io.write((char *)&i, sizeof(i));
            for (auto &i : bias.reshaped()) io.write((char *)&i, sizeof(i));
        }
        void read(istream &io) {
            for (auto &i : weight.reshaped()) io.read((char *)&i, sizeof(i));
            for (auto &i : bias.reshaped()) io.read((char *)&i, sizeof(i));
        }
    };
    ```
=== "注解"

    线性层的反向传播：

