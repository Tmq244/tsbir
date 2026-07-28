# TASK-former 多任务损失与梯度路径

三项损失不是分别执行三次 `optimizer.step()`，而是先合成一个标量总损失，反向传播时各条计算路径产生的梯度在同一参数的 `.grad` 中相加，最后统一更新一次。

## 1. 前向计算关系

先定义三个归一化后的 embedding：

$$
v=f_{\theta_v}(I),\qquad
s=f_{\theta_v}(S),\qquad
t=f_{\theta_t}(T)
$$

其中：

- \(I\)：目标图像；
- \(S\)：草图；
- \(T\)：文本；
- \(\theta_v\)：图像和草图共享的视觉编码器参数；
- \(\theta_t\)：文本编码器参数。

由于图像和草图权重共享，二者都调用同一个 \(f_{\theta_v}\)。

融合查询为：

$$
q=
\operatorname{Normalize}\left(\frac{t+s}{2}\right)
$$

### 对比损失

$$
L_e=\operatorname{Contrastive}(v,q)
$$

所以 \(L_e\) 依赖：

- 图像特征 \(v\)；
- 草图特征 \(s\)；
- 文本特征 \(t\)；
- logit scale。

### 分类损失

当前代码实际是：

$$
L_c
=
\frac{
L_c^{img}+L_c^{sketch}+L_c^{text}
}{3}
$$

其中：

$$
L_c^{img}=\operatorname{ASL}(C_{\theta_c}(v),y)
$$

$$
L_c^{sketch}=\operatorname{ASL}(C_{\theta_c}(s),y)
$$

$$
L_c^{text}=\operatorname{ASL}(C_{\theta_c}(t),y)
$$

三个模态共用一个分类头 \(C_{\theta_c}\)。

### Caption decoder 损失

构造视觉上下文：

$$
c=\frac{v+s}{2}
$$

然后：

$$
L_d=
\operatorname{CE}
\left(
D_{\theta_d}(w_{<k},c),w_k
\right)
$$

所以 \(L_d\) 依赖：

- 图像特征 \(v\)；
- 草图特征 \(s\)；
- decoder 参数 \(\theta_d\)；

但不依赖文本特征 \(t\)。

最终：

$$
L=100L_e+10L_c+L_d
$$

## 2. `loss.backward()` 做了什么

代码相当于：

```python
loss = 100 * embedding + 10 * classification + decoder
loss.backward()
```

根据微分的线性性质，对于任意参数 \(\theta\)：

$$
\nabla_\theta L
=
100\nabla_\theta L_e
+
10\nabla_\theta L_c
+
\nabla_\theta L_d
$$

但如果某项损失与该参数没有计算路径，那么对应梯度就是 0。

例如 decoder 参数没有参与 \(L_e\)：

$$
\nabla_{\theta_d}L_e=0
$$

所以 decoder 不会收到 \(L_e\) 的梯度。

参数与损失的依赖关系如下：

| 参数 | \(L_e\) | \(L_c\) | \(L_d\) |
|---|---:|---:|---:|
| 共享视觉编码器 \(\theta_v\) | 有 | 图像、草图分支 | 有 |
| 文本编码器 \(\theta_t\) | 有 | 文本分支 | 无 |
| 分类头 \(\theta_c\) | 无 | 有 | 无 |
| Caption decoder \(\theta_d\) | 无 | 无 | 有 |
| logit scale | 有 | 无 | 无 |

## 3. 共享视觉编码器的精确梯度

高层写法是：

$$
\nabla_{\theta_v}L
=
100\nabla_{\theta_v}L_e
+
10\nabla_{\theta_v}L_c
+
\nabla_{\theta_v}L_d
$$

展开 \(L_c\) 后，更精确地是：

$$
\nabla_{\theta_v}L
=
100\nabla_{\theta_v}L_e
+
\frac{10}{3}
\left(
\nabla_{\theta_v}L_c^{img}
+
\nabla_{\theta_v}L_c^{sketch}
\right)
+
\nabla_{\theta_v}L_d
$$

没有 \(L_c^{text}\) 项，是因为文本分类路径不经过视觉编码器：

$$
\nabla_{\theta_v}L_c^{text}=0
$$

### \(L_e\) 对视觉编码器的两条路径

视觉编码器在 \(L_e\) 中出现两次。

第一条是图库图像路径：

$$
\theta_v\rightarrow v\rightarrow L_e
$$

第二条是查询草图路径：

$$
\theta_v\rightarrow s\rightarrow q\rightarrow L_e
$$

因此：

$$
\nabla_{\theta_v}L_e
=
\left(
\frac{\partial L_e}{\partial v}
\frac{\partial v}{\partial\theta_v}
\right)
+
\left(
\frac{\partial L_e}{\partial q}
\frac{\partial q}{\partial s}
\frac{\partial s}{\partial\theta_v}
\right)
$$

融合后的重新归一化也是可微分的，所以梯度能够经过：

```text
Le → normalized fused query → sketch feature → visual encoder
```

### \(L_c\) 对视觉编码器的两条路径

$$
\theta_v\rightarrow v\rightarrow C(v)\rightarrow L_c^{img}
$$

以及：

$$
\theta_v\rightarrow s\rightarrow C(s)\rightarrow L_c^{sketch}
$$

因为图像和草图编码器共享参数，两条梯度最终累加到同一组 \(\theta_v\)。

### \(L_d\) 对视觉编码器的两条路径

$$
c=\frac{v+s}{2}
$$

因此：

$$
\frac{\partial L_d}{\partial\theta_v}
=
\frac{1}{2}
\frac{\partial L_d}{\partial c}
\frac{\partial v}{\partial\theta_v}
+
\frac{1}{2}
\frac{\partial L_d}{\partial c}
\frac{\partial s}{\partial\theta_v}
$$

同样包含图像和草图两条路径。

所以视觉编码器最终同时受到：

- 对比检索目标；
- 图像分类目标；
- 草图分类目标；
- caption 生成目标；

四类监督信号。

## 4. 文本编码器的精确梯度

高层写法：

$$
\nabla_{\theta_t}L
=
100\nabla_{\theta_t}L_e
+
10\nabla_{\theta_t}L_c
$$

展开后：

$$
\nabla_{\theta_t}L
=
100\nabla_{\theta_t}L_e
+
\frac{10}{3}\nabla_{\theta_t}L_c^{text}
$$

### 对比路径

$$
\theta_t\rightarrow t\rightarrow q\rightarrow L_e
$$

即：

$$
\nabla_{\theta_t}L_e
=
\frac{\partial L_e}{\partial q}
\frac{\partial q}{\partial t}
\frac{\partial t}{\partial\theta_t}
$$

### 分类路径

$$
\theta_t\rightarrow t\rightarrow C(t)\rightarrow L_c^{text}
$$

### 为什么没有 \(L_d\)？

Decoder context 是：

$$
c=\frac{v+s}{2}
$$

其中没有 \(t\)，所以：

$$
\frac{\partial L_d}{\partial t}=0
$$

进而：

$$
\nabla_{\theta_t}L_d=0
$$

caption 的真实 tokens 只是监督标签，不是由文本编码器生成的，因此也不会形成到文本编码器的梯度路径。

## 5. 分类头的精确梯度

分类头在三个模态上重复使用：

```python
module.classify(image_features)
module.classify(sketch_features)
module.classify(text_features)
```

所以分类头梯度是：

$$
\nabla_{\theta_c}L
=
\frac{10}{3}
\left(
\nabla_{\theta_c}L_c^{img}
+
\nabla_{\theta_c}L_c^{sketch}
+
\nabla_{\theta_c}L_c^{text}
\right)
$$

分类头不参与对比学习和 caption 生成：

$$
\nabla_{\theta_c}L_e=0,\qquad
\nabla_{\theta_c}L_d=0
$$

三个模态不是各自拥有一个分类头，而是共享同一个分类头，所以三个分类梯度会共同更新这两层全连接网络。

## 6. Caption decoder 的梯度

Decoder 只参与 \(L_d\)：

$$
\nabla_{\theta_d}L
=
\nabla_{\theta_d}L_d
$$

总损失中 \(L_d\) 的权重为 1，所以不再额外缩放。

这不代表 decoder 更新弱或不更新。因为 decoder 没有来自其他损失的梯度，它的全部更新信号就是 \(L_d\)。

decoder 参数包括：

- token embedding；
- positional embedding；
- causal self-attention；
- cross-attention；
- FFN；
- 输出词表投影。

这些参数都会收到 \(L_d\) 梯度。

## 7. 权重 100、10、1 不等于实际贡献比例

不能简单理解为：

```text
Le 占 100/111
Lc 占 10/111
Ld 占 1/111
```

真正影响参数更新的是梯度大小，而不是损失标量本身：

$$
100\|\nabla L_e\|,\qquad
10\|\nabla L_c\|,\qquad
\|\nabla L_d\|
$$

例如某个视觉参数上：

$$
\nabla L_e=0.002,\qquad
\nabla L_c=0.03,\qquad
\nabla L_d=0.4
$$

加权后：

$$
100(0.002)+10(0.03)+0.4
=
0.2+0.3+0.4
$$

尽管 \(L_d\) 的权重只有 1，其梯度贡献仍可能最大。

而且梯度是向量，不只是标量。不同任务的梯度可能：

- 方向一致：互相增强；
- 方向相反：部分抵消；
- 接近正交：分别改变不同参数方向。

例如：

$$
100\nabla L_e
\approx
-10\nabla L_c
$$

两项可能相互抵消。这就是多任务训练中的梯度冲突。

## 8. 一次参数更新的完整过程

代码实际执行顺序是：

```python
optimizer.zero_grad()

loss = 100 * Le + 10 * Lc + Ld
loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

具体过程：

1. 前向计算三个损失。
2. 构造一个总损失标量。
3. `backward()` 沿所有计算路径传播。
4. 同一参数来自不同损失、不同模态的梯度累加到 `.grad`。
5. 梯度裁剪对合并后的总梯度进行处理。
6. AdamW 根据合并梯度、历史一阶/二阶动量和 weight decay 更新参数。

优化器看到的只有最终合并后的：

```python
parameter.grad
```

它不知道这部分梯度来自 \(L_e\)、\(L_c\) 还是 \(L_d\)。

如果启用梯度累积，代码还会计算：

```python
(loss / accumulation).backward()
```

这会把每个 micro-batch 的三项梯度都除以累积次数，然后在多个 micro-batch 之间继续相加，最后再执行一次 `optimizer.step()`。
