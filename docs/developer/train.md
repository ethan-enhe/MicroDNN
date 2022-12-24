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


## 网络结构

### 基类
```cpp
struct net {
    virtual string shape() = 0;
    virtual void set_train_mode(const bool &) = 0;
    virtual vec_batch forward(const vec_batch &) = 0;
    virtual vec_batch backward(const vec_batch &) = 0;
    virtual void upd(optimizer &, const batch &) = 0;
    virtual void writef(const string &f) = 0;
    virtual void readf(const string &f) = 0;
    virtual vec_batch &out() = 0;
};
```

### sequential

```cpp
struct sequential : public net {
    int batch_sz;
    vector<layerp> layers;
    sequential() : batch_sz(0) {
        layerp x = make_shared<same>();
        x->name = "input";
        layers.emplace_back(x);
    }
    void add(const layerp &x) { layers.push_back(x); }
    string shape() {
        string res = "";
        for (auto &it : layers) res += it->name + "\n";
        return res;
    }
    void set_train_mode(const bool &new_train_mod) {
        for (auto &l : layers) l->set_train_mode(new_train_mod);
    }
    vec_batch forward(const vec_batch &input) {
        if ((int)input.size() != batch_sz) {
            batch_sz = input.size();
            for (auto &l : layers) l->resize(batch_sz);
        }
        int layer_sz = layers.size();
        layers[0]->forward(input);
        for (int i = 1; i < layer_sz; i++) layers[i]->forward(layers[i - 1]->out);
        return layers.back()->out;
    }
    vec_batch backward(const vec_batch &label) {
        for (int i = 0; i < batch_sz; i++) layers.back()->grad[i] = layers.back()->out[i] - label[i];
        int layer_sz = layers.size();
        for (int i = layer_sz - 2; i >= 0; i--)
            layers[i]->backward(i ? layers[i - 1]->out : vec_batch(), layers[i + 1]->grad);
        return layers[0]->grad;
    }
    void upd(optimizer &opt, const batch &data) {
        int layer_sz = layers.size();
        for (int i = 0; i < layer_sz; i++) layers[i]->clear_grad();
        forward(data.first);
        backward(data.second);
        for (int i = 0; i < layer_sz; i++) layers[i]->upd(opt);
    }
    void writef(const string &f) {
        ofstream fout(f, ios::binary | ios::out);
        int layer_sz = layers.size();
        for (int i = 0; i < layer_sz; i++) layers[i]->write(fout);
        fout.close();
    }
    void readf(const string &f) {
        ifstream fin(f, ios::binary | ios::in);
        int layer_sz = layers.size();
        for (int i = 0; i < layer_sz; i++) layers[i]->read(fin);
        fin.close();
    }
    vec_batch &out() { return layers.back()->out; }
};
```

## 数据集对象


=== "代码"
    ```cpp
    struct data_set {
        batch train, valid;
        data_set() {}
        data_set(const batch &all_data) {
            for (int i = 0; i < (int)all_data.first.size(); i++) {
                int rnd = ri(0, 6);
                if (rnd == 0) {
                    valid.first.push_back(all_data.first[i]);
                    valid.second.push_back(all_data.second[i]);
                } else {
                    train.first.push_back(all_data.first[i]);
                    train.second.push_back(all_data.second[i]);
                }
            }
        }
        batch get_train_batch(int batch_sz) const {
            assert(train.first.size());
            batch res;
            for (int i = 0; i < batch_sz; i++) {
                int id = ri(0, train.first.size() - 1);
                res.first.push_back(train.first[id]);
                res.second.push_back(train.second[id]);
            }
            return res;
        }
        batch get_valid_batch(int batch_sz) const {
            assert(valid.first.size());
            batch res;
            for (int i = 0; i < batch_sz; i++) {
                int id = ri(0, valid.first.size() - 1);
                res.first.push_back(valid.first[id]);
                res.second.push_back(valid.second[id]);
            }
            return res;
        }
    };
    ```
=== "注解"
     构造函数中随机将大概 $\frac 17$ 的数据放到 validate 里。在数据集较小是可能出现 validate 或者 train 为空，此时也可以自己重写一个数据集对象。

     后面也是随机从数据集中取出一个 batch，如对这个机制不满意，可以重写。


## 默认的训练函数

=== "代码"
    ```cpp
    void upd(optimizer &opt, const data_set &data, net &net, int batch_sz, int epoch,
             function<float(const vec_batch &, const vec_batch &)> err_func, const string &save_file = "") {
        int t0 = clock();
        float tloss = 0, mult = 1, mn = INF;
        for (int i = 1; i <= epoch; i++) {
            auto tmp = data.get_train_batch(batch_sz);
            net.upd(opt, tmp);
    ```
=== "注解"
    获取一个 batch 的训练数据，并更新系数

---
=== "代码"
    ```cpp
            mult *= 0.9;
            tloss = tloss * 0.9 + err_func(net.out(), tmp.second) * 0.1;
            if (i % 50 == 0) {
                cerr << "-------------------------" << endl;
                cerr << "Time elapse: " << (float)(clock() - t0) / CLOCKS_PER_SEC << endl;
                cerr << "Epoch: " << i << endl;
                cerr << "Loss: " << tloss / (1. - mult) << endl;
                if (i % 1000 == 0) {
                    net.set_train_mode(0);
                    float sum = 0;
                    for (int j = 0; j < (int)data.valid.first.size(); j++) {
                        batch tmp = {{data.valid.first[j]}, {data.valid.second[j]}};
                        sum += err_func(net.forward(tmp.first), tmp.second);
                    }
                    net.set_train_mode(1);
                    sum /= data.valid.first.size();
                    cerr << "!! Error: " << sum << endl;
                    if (sum < mn && save_file != "") {
                        cerr << "Saved" << endl;
                        mn = sum;
                        net.writef(save_file);
                    }
                }
            }
        }
    }
    ```
=== "注解"
    计算平均 loss，并且保存 loss 最低的网络参数。


