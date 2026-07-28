# TASK-former 模型融合与草图数据处理优化实验报告

> 基线：逐像素 stroke dropout + 固定平均融合（Avg）
> 正式模型选择：10-epoch 实验统一使用 epoch 10；冻结 Gate 使用预先规定的 epoch 3
> 结论：四种优化中，只有 **Human-style 复合增强 + 双视图一致性（C2）** 在显著提高 Human sketch 检索性能的同时，基本保持了 Synthetic sketch 性能。

## 1. 实验目的

训练集只提供 synthetic sketch，而最终测试需要处理 human sketch。二者存在明显的域差异：synthetic sketch 通常轮廓完整、线宽稳定、结构规则；human sketch 则存在局部缺口、线宽变化、比例偏移、形变、简化以及语义细节遗漏。

本组实验从两个方向探究改进方法：

1. **模型融合优化**：让模型根据 text 和 sketch 的可靠性自适应决定融合权重；
2. **草图数据处理优化**：把 synthetic sketch 变换为更接近 human sketch 的训练样本。

在普通方法之外，共测试四种优化：


| 编号 | 类型             | 方法                                      |
| ---- | ---------------- | ----------------------------------------- |
| 1    | 模型融合         | 编码器与 Gate 联合训练                    |
| 2    | 模型融合         | 冻结编码器，只训练带可靠性监督的 Gate     |
| 3    | 数据处理         | Segment-only：按骨架连通笔画分段删除      |
| 4    | 数据处理与正则化 | Human-style 复合增强 + 双视图一致性（C2） |

## 2. 普通方法与统一评测协议

### 2.1 普通方法 A

普通方法使用 CLIP-pretrained TASK-former，训练 10 epoch，固定平均融合，并使用原论文式逐像素 stroke dropout。设归一化后的文本、草图和目标图像特征分别为

$$
t=f_t(x_t),\qquad s=f_s(x_s),\qquad v=f_i(x_i),
$$

则普通方法的 mixed query 为

$$
q_{\mathrm{avg}}
=\operatorname{norm}\!\left(\frac{t+s}{2}\right)
=\operatorname{norm}(t+s).
$$

其中 `norm` 表示 L2 归一化。固定平均等价于文本权重

$$
\alpha=0.5,
\qquad
q=\operatorname{norm}\bigl(\alpha t+(1-\alpha)s\bigr).
$$

### 2.2 普通逐像素 stroke dropout

设草图前景像素集合为 $F$。每次增强首先采样保留概率

$$
p\sim \mathcal U(0.6,1.0),
$$

然后对每个前景像素独立采样

$$
z_j\sim\operatorname{Bernoulli}(p),\qquad j\in F.
$$

增强后的前景为

$$
F'=\{j\in F\mid z_j=1\}.
$$

该方法能够制造不完整草图，但像素之间互相独立，容易形成均匀的“椒盐式”缺失，未必符合人工绘制时沿笔画或局部结构发生遗漏的规律。

### 2.3 公共训练目标

除第二阶段冻结 Gate 外，基线、联合 Gate、Segment-only 和 C2 都采用相同的主训练目标：

$$
\mathcal L_{\mathrm{base}}
=100\mathcal L_{\mathrm{ret}}
+10\mathcal L_{\mathrm{cls}}
+\mathcal L_{\mathrm{dec}}.
$$

其中检索损失为跨 GPU gather 后的双向对比损失。设 batch 内图像和查询特征矩阵分别为 $V,Q$，温度尺度为 $\gamma=\exp(\text{logit\_scale})$，则

$$
\mathcal L_{\mathrm{ret}}
=\frac12\left[
\operatorname{CE}(\gamma QV^\top,y)
+\operatorname{CE}(\gamma VQ^\top,y)
\right].
$$

$\mathcal L_{\mathrm{cls}}$ 是图像、文本和草图三种特征的非对称多标签分类损失，$\mathcal L_{\mathrm{dec}}$ 是以图像和草图特征为上下文的 caption 重建损失。

### 2.4 评测协议

- Gallery：5,000 张 COCO 测试图像；
- Sketch query：5,000 张 human sketch 或对应的 5,000 张 synthetic sketch；
- Text/Mixed query：25,010 条 caption；
- Human 和 Synthetic 评测只改变 sketch 来源，gallery 和 caption 保持相同；
- 正式结果不根据 Human test 选择中间 epoch。

若第 $i$ 个查询的正确图像排名为 $r_i$，则

$$
R@K=\frac1N\sum_{i=1}^{N}\mathbf 1[r_i\le K].
$$

下文所有指标均以百分比表示，格式为 **R@1 / R@5 / R@10**；$\Delta$ 表示相对普通方法 A 的百分点变化。

普通方法 A 的结果为：


| 测试域    |                Sketch |                  Text |                 Mixed |
| --------- | --------------------: | --------------------: | --------------------: |
| Human     | 12.60 / 24.38 / 31.66 | 43.63 / 72.26 / 81.92 | 49.27 / 75.86 / 85.07 |
| Synthetic | 89.16 / 95.78 / 97.40 | 43.63 / 72.26 / 81.92 | 97.45 / 99.67 / 99.86 |

---

## 3. 方法一：编码器与 Gate 联合训练

### 3.1 动机

固定平均融合假设 text 和 sketch 对所有查询同样可靠，但真实情况并非如此：

- 有的 caption 描述准确，sketch 信息较少，应提高文本权重；
- 有的 caption 含糊，sketch 提供了形状线索，应提高草图权重；
- 不同查询的最优融合系数可能不同。

因此，第一个想法是用可学习 Gate 为每个查询预测独立的 $\alpha$，替代固定的 $\alpha=0.5$。

### 3.2 方法实现

Gate 输入由文本特征、草图特征及其逐维乘积组成：

$$
g=[t;s;t\odot s]\in\mathbb R^{3d}.
$$

其中 $t\odot s$ 用于显式表示两种模态的一致程度。Gate 是一个隐藏维度为 256 的两层 MLP：

$$
\hat\alpha
=\sigma\!\left(
W_2\operatorname{GELU}(W_1g+b_1)+b_2
\right),
$$

$$
q_{\mathrm{gate}}
=\operatorname{norm}\bigl(
\hat\alpha t+(1-\hat\alpha)s
\bigr).
$$

这里 $\hat\alpha$ 是文本权重。输出层权重初始化为 0，偏置初始化为

$$
b_2=\operatorname{logit}(0.5)=0,
$$

所以训练开始时 $\hat\alpha=0.5$，与普通平均融合严格等价。随后编码器和 Gate 使用同一个主目标联合训练 10 epoch；编码器学习率为 $10^{-5}$，随机初始化 Gate 的学习率为 $10^{-4}$。

### 3.3 Human sketch 结果对比


| 模式   |            普通方法 A |             联合 Gate |                 $\Delta$ |
| ------ | --------------------: | --------------------: | -----------------------: |
| Sketch | 12.60 / 24.38 / 31.66 | 11.16 / 22.64 / 29.68 | −1.44 / −1.74 / −1.98 |
| Text   | 43.63 / 72.26 / 81.92 | 43.52 / 72.44 / 82.06 |   −0.12 / +0.18 / +0.14 |
| Mixed  | 49.27 / 75.86 / 85.07 | 46.96 / 73.75 / 83.05 | −2.31 / −2.11 / −2.02 |

### 3.4 Synthetic sketch 结果对比


| 模式   |            普通方法 A |             联合 Gate |               $\Delta$ |
| ------ | --------------------: | --------------------: | ---------------------: |
| Sketch | 89.16 / 95.78 / 97.40 | 89.34 / 95.88 / 97.30 | +0.18 / +0.10 / −0.10 |
| Text   | 43.63 / 72.26 / 81.92 | 43.52 / 72.44 / 82.06 | −0.12 / +0.18 / +0.14 |
| Mixed  | 97.45 / 99.67 / 99.86 | 97.41 / 99.68 / 99.88 | −0.03 / +0.01 / +0.02 |

### 3.5 结果解释

联合 Gate 没有产生预期的逐查询分化。Human mixed 模式下

$$
\operatorname{mean}(\hat\alpha)=0.485,
\qquad
\operatorname{std}(\hat\alpha)=0.023.
$$

Gate 基本退化为一个接近 0.5 的固定系数。Synthetic mixed 性能与基线近似相同，但 Human mixed R@1 下降 2.31 个百分点，说明增加参数并不保证结果优于固定平均：联合优化可以改变原本已经较好的特征空间，而弱且高度不平衡的 Gate 梯度不足以学出可靠的逐查询策略。

**结论：联合 Gate 无效，且显著降低 Human mixed 检索性能。**

---

## 4. 方法二：冻结编码器，只训练带可靠性监督的 Gate

### 4.1 动机

方法一同时更新编码器和 Gate，存在两个问题：

1. Gate 的梯度来自整体检索损失，监督过于间接；
2. 联合训练可能破坏已经训练好的平均融合特征空间。

因此，第二个方案直接加载普通方法 A 的最终权重，冻结图像、文本和草图编码器以及其他参数，只训练 Gate。这样可以把实验严格聚焦于“能否从固定特征中学到更好的自适应融合系数”。同时加入显式可靠性 teacher，让 Gate 知道每个训练样本更应该依赖文本还是草图。

### 4.2 冻结与 mixed-only 训练

编码器输出在 `no_grad` 下计算，只有 Gate 参数参与反向传播：

$$
\nabla_{\theta_i,\theta_t,\theta_s}\mathcal L=0,
\qquad
\nabla_{\theta_g}\mathcal L\ne0.
$$

训练数据仍然只使用 synthetic sketch，并关闭 query dropout，保证每个训练查询同时包含 text 和 sketch。该阶段训练 3 epoch，共 4,428 step，全局 batch 为 $192\times2=384$。

### 4.3 基于 α-grid 的可靠性 teacher

最终实现使用候选融合系数网格

$$
\mathcal A=\{0,0.1,0.2,\ldots,1.0\}.
$$

对第 $i$ 个样本和第 $j$ 个候选系数 $\alpha_j$，先构造

$$
q_{ij}=\operatorname{norm}\bigl(
\alpha_jt_i+(1-\alpha_j)s_i
\bigr).
$$

设 batch 内目标图像为 $v_i$，所有候选图像为 $\{v_k\}$，则该候选系数对应的单样本检索损失为

$$
\ell_{ij}
=-\log
\frac{\exp(\gamma q_{ij}^{\top}v_i)}
{\sum_k\exp(\gamma q_{ij}^{\top}v_k)}.
$$

使用温度 $\tau=0.1$ 将各候选损失转换为软 teacher 分布：

$$
p_{ij}
=\frac{\exp(-\ell_{ij}/\tau)}
{\sum_m\exp(-\ell_{im}/\tau)}.
$$

Gate 的连续软目标是候选 $\alpha$ 的期望：

$$
\alpha_i^*=\sum_jp_{ij}\alpha_j.
$$

如果所有候选 $\alpha$ 的损失近似相同，teacher 分布接近均匀，此时监督不可靠。为此使用归一化熵定义置信度：

$$
c_i
=1-\frac{H(p_i)}{\log|\mathcal A|},
\qquad
H(p_i)=-\sum_jp_{ij}\log p_{ij}.
$$

可靠性损失使用置信度加权 MSE：

$$
\mathcal L_{\mathrm{rel}}
=\frac{\sum_i(c_0+c_i)(\hat\alpha_i-\alpha_i^*)^2}
{\sum_i(c_0+c_i)},
\qquad c_0=0.1.
$$

最终 Gate 训练目标为

$$
\mathcal L_{\mathrm{gate}}
=\mathcal L_{\mathrm{ret}}(q_{\mathrm{gate}},v)
+\mathcal L_{\mathrm{rel}}.
$$

目标图像只用于在训练时生成 teacher，不是 Gate 的输入；推理时 Gate 只接收 $[t;s;t\odot s]$。

### 4.4 Human sketch 结果对比


| 模式   |            普通方法 A |       冻结可靠性 Gate |                 $\Delta$ |
| ------ | --------------------: | --------------------: | -----------------------: |
| Sketch | 12.60 / 24.38 / 31.66 | 11.52 / 23.68 / 30.06 | −1.08 / −0.70 / −1.60 |
| Text   | 43.63 / 72.26 / 81.92 | 37.97 / 65.57 / 76.07 | −5.66 / −6.69 / −5.85 |
| Mixed  | 49.27 / 75.86 / 85.07 | 48.02 / 74.25 / 83.68 | −1.25 / −1.61 / −1.39 |

### 4.5 Synthetic sketch 结果对比


| 模式   |            普通方法 A |       冻结可靠性 Gate |                 $\Delta$ |
| ------ | --------------------: | --------------------: | -----------------------: |
| Sketch | 89.16 / 95.78 / 97.40 | 84.26 / 93.92 / 95.96 | −4.90 / −1.86 / −1.44 |
| Text   | 43.63 / 72.26 / 81.92 | 37.97 / 65.57 / 76.07 | −5.66 / −6.69 / −5.85 |
| Mixed  | 97.45 / 99.67 / 99.86 | 97.52 / 99.69 / 99.86 |    +0.08 / +0.02 / +0.00 |

### 4.6 公平性说明与结果解释

该实验启用了 `gate_mixed_only`：

- mixed query 使用训练后的 Gate；
- text-only 直接使用 $t$，不再与 blank sketch 特征融合；
- sketch-only 直接使用 $s$，不再与 empty text 特征融合。

因此，该方法的 Text 和 Sketch 指标还包含“单模态路由改变”的影响，不能把它们的下降全部归因于 Gate。判断可靠性 Gate 本身是否有效，应主要比较 Mixed。

Mixed 结果仍然不理想：Synthetic R@1 仅提高 0.08 个百分点，而 Human R@1 下降 1.25 个百分点。其 Gate 输出仍接近常数：

$$
\begin{aligned}
\text{Human mixed: }&\hat\alpha=0.477\pm0.013,\\
\text{Synthetic mixed: }&\hat\alpha=0.480\pm0.015.
\end{aligned}
$$

这说明 synthetic 训练数据中的 teacher 信号不足以学习对 human sketch 有效的可靠性判断。即使冻结编码器避免了特征空间漂移，Synthetic-to-Human 域差异仍然存在。

**结论：可靠性监督 Gate 在 Synthetic mixed 上近似持平，但没有改善 Human mixed。**

---

## 5. 方法三：Segment-only 骨架笔画分段删除

### 5.1 动机

逐像素 dropout 独立删除前景像素，而人工草图的不完整性经常表现为一小段线条、一个局部轮廓或一个次要结构整体缺失。因此提出：先把草图骨架拆成连通笔画段，再以“段”为单位删除，从而产生更加连贯的结构缺失。

为了形成清晰的论文消融实验，Segment-only 只改变 dropout 类型；初始化、seed、batch、训练轮数、损失和固定平均融合均与普通方法 A 相同。

### 5.2 骨架分段

对前景集合 $F$ 使用 Zhang-Suen 算法得到单像素骨架 $S$。对骨架像素 $u$，定义其 8 邻域度数为

$$
d(u)=\sum_{w\in\mathcal N_8(u)}\mathbf 1[w\in S].
$$

满足 $d(u)\ne2$ 的像素被视为端点或交叉节点。移除这些节点后，把剩余无分叉路径划分为连通分量；过长路径进一步切成不超过 64 个骨架像素的段。最后使用最近骨架点标签把一像素骨架段扩展回原始抗锯齿/粗线前景。

### 5.3 按段删除

与普通方法相同，首先采样

$$
p\sim\mathcal U(0.6,1.0).
$$

若原始前景面积为 $|F|$，总删除预算为

$$
B=\left\lfloor|F|(1-p)\right\rfloor.
$$

设第 $j$ 个段覆盖的前景面积为 $a_j$，随机选择待删除段集合 $D$，满足

$$
\sum_{j\in D}a_j\le B,
\qquad
a_j\le0.2|F|,
\qquad
M-|D|\ge3,
$$

其中 $M$ 是总段数。第二、三个约束分别避免单次删除过大的主体结构，并保证至少保留三个分段。增强结果为

$$
F'=F\setminus\bigcup_{j\in D}F_j.
$$

### 5.4 Human sketch 结果对比


| 模式   |            普通方法 A |          Segment-only |                 $\Delta$ |
| ------ | --------------------: | --------------------: | -----------------------: |
| Sketch | 12.60 / 24.38 / 31.66 |  9.98 / 20.56 / 27.34 | −2.62 / −3.82 / −4.32 |
| Text   | 43.63 / 72.26 / 81.92 | 43.92 / 72.46 / 82.35 |    +0.29 / +0.20 / +0.43 |
| Mixed  | 49.27 / 75.86 / 85.07 | 46.55 / 73.41 / 82.86 | −2.72 / −2.45 / −2.21 |

### 5.5 Synthetic sketch 结果对比


| 模式   |            普通方法 A |          Segment-only |              $\Delta$ |
| ------ | --------------------: | --------------------: | --------------------: |
| Sketch | 89.16 / 95.78 / 97.40 | 90.34 / 96.70 / 97.92 | +1.18 / +0.92 / +0.52 |
| Text   | 43.63 / 72.26 / 81.92 | 43.92 / 72.46 / 82.35 | +0.29 / +0.20 / +0.43 |
| Mixed  | 97.45 / 99.67 / 99.86 | 97.73 / 99.75 / 99.92 | +0.28 / +0.08 / +0.06 |

### 5.6 结果解释

Segment-only 提高了 Synthetic sketch 指标，但明显降低了 Human sketch 指标：

$$
\Delta R@1_{\mathrm{synthetic\ sketch}}=+1.18,
\qquad
\Delta R@1_{\mathrm{human\ sketch}}=-2.62.
$$

这说明结构化删除本身并不能模拟完整的人工绘制分布。它可能让模型更擅长识别“由同一种 synthetic sketch 经规则分段删除得到的样本”，但人工草图还包含线宽、坐标、比例、局部形变和轮廓简化等差异。完全取消 pixel dropout 也降低了扰动的细粒度多样性。

**结论：Segment-only 是一个干净但失败的消融；它改善 Synthetic，却扩大了 Synthetic-to-Human 泛化差距。**

---

## 6. 方法四：Human-style 复合增强 + 双视图一致性（C2）

### 6.1 动机

Segment-only 的失败说明 synthetic-to-human 差异不是单一的“整段笔画缺失”。人工草图的变化是复合的：

- 线宽不稳定；
- 局部笔画断裂和不均匀缺口；
- 曲线和坐标发生抖动或非刚性形变；
- 轮廓被简化，小结构被忽略；
- 少量完整笔画段缺失；
- 同一个人即使画同一物体，两次结果也会不同，但语义应保持一致。

因此，C2 不再用单一 Segment-only 替换原方法，而是在同一个保留预算内组合多种 human-style 扰动，并通过双视图一致性损失约束草图编码器学习增强不变性。

### 6.2 Human-style 复合增强

每个原始 synthetic sketch 采样一个共享目标保留率

$$
p_{\mathrm{keep}}\sim\mathcal U(0.6,1.0).
$$

在不突破 60% 最低前景保留率的条件下，随机组合下列变换：


| 变换                | 概率 | 参数                                 |
| ------------------- | ---: | ------------------------------------ |
| 线宽随机变细/变粗   | 0.35 | 灰度膨胀或腐蚀，半径 1–2            |
| 非刚性 elastic warp | 0.30 | amplitude 1–4，Gaussian sigma 8–16 |
| 局部不均匀缺口      | 0.30 | 1–4 个缺口，长度 4–16，宽度 2–6   |
| 轮廓简化            | 0.20 | 缩小到 0.55–0.85 后再放大           |
| 小结构删除          | 0.15 | 优先删除骨架长度不超过 24 的小段     |
| 普通连通段删除      | 0.10 | 最大骨架段长度 64                    |

单次结构删除不超过参考前景面积的 10%。完成结构变换后，仍然使用逐像素删除补足剩余预算：

$$
N_{\mathrm{pixel\ delete}}
=\max\bigl(0,|F_{\mathrm{changed}}|-N_{\mathrm{target}}\bigr),
$$

$$
N_{\mathrm{target}}
=\max\left(
\left\lceil0.6|F_{\mathrm{ref}}|\right\rceil,
\left\lfloor p_{\mathrm{keep}}|F_{\mathrm{ref}}|\right\rfloor
\right).
$$

所以 C2 保留了普通方法的像素级随机性，但把局部结构变化、形变和线宽变化纳入同一个完整度预算，避免多种删除简单叠加后把草图破坏得过重。

### 6.3 双视图与一致性损失

对同一 synthetic sketch 独立采样两套 human-style 风格扰动，得到 $s_1,s_2$。两个视图共享相同的全局 affine 和 crop，从而让一致性损失主要约束风格扰动，而不是要求编码器忽略完全不同的观察区域。

设两个归一化草图特征为

$$
z_1=f_s(s_1),\qquad z_2=f_s(s_2),
$$

则一致性损失为

$$
\mathcal L_{\mathrm{cons}}
=\frac1{N_s}\sum_{i=1}^{N_s}
\left(
1-\frac{z_{1,i}^{\top}z_{2,i}}
{\|z_{1,i}\|_2\|z_{2,i}\|_2}
\right).
$$

只对 sketch 未被 query dropout 置空的样本计算该损失。C2 的总目标为

$$
\mathcal L_{\mathrm{C2}}
=100\mathcal L_{\mathrm{ret}}
+10\mathcal L_{\mathrm{cls}}
+\mathcal L_{\mathrm{dec}}
+\lambda_{\mathrm{cons}}\mathcal L_{\mathrm{cons}},
\qquad
\lambda_{\mathrm{cons}}=1.
$$

双视图增加了一次 sketch ViT 前向，因此训练开启 gradient checkpointing 降低显存；它只改变显存和计算开销，不改变目标函数。

### 6.4 Human sketch 结果对比


| 模式   |            普通方法 A |                        C2 |                  $\Delta$ |
| ------ | --------------------: | ------------------------: | ------------------------: |
| Sketch | 12.60 / 24.38 / 31.66 | **14.08 / 26.64 / 33.26** | **+1.48 / +2.26 / +1.60** |
| Text   | 43.63 / 72.26 / 81.92 | **43.69 / 72.62 / 82.22** |     +0.06 / +0.36 / +0.30 |
| Mixed  | 49.27 / 75.86 / 85.07 | **50.46 / 76.60 / 85.21** | **+1.19 / +0.74 / +0.14** |

按 `coco_id` 对 5,000 个测试图像进行 30,000 次配对 bootstrap，得到：


| 指标              | 提升（百分点） |          95% CI | 判断   |
| ----------------- | -------------: | --------------: | ------ |
| Human Sketch R@1  |          +1.48 |  [+0.74, +2.24] | 显著   |
| Human Sketch R@5  |          +2.26 |  [+1.42, +3.12] | 显著   |
| Human Sketch R@10 |          +1.60 |  [+0.72, +2.50] | 显著   |
| Human Mixed R@1   |          +1.19 |  [+0.50, +1.88] | 显著   |
| Human Mixed R@5   |          +0.74 |  [+0.18, +1.30] | 显著   |
| Human Mixed R@10  |          +0.14 | [−0.33, +0.61] | 不显著 |

### 6.5 Synthetic sketch 结果对比


| 模式   |            普通方法 A |                        C2 |                $\Delta$ |
| ------ | --------------------: | ------------------------: | ----------------------: |
| Sketch | 89.16 / 95.78 / 97.40 | **90.06 / 95.70 / 97.14** | +0.90 / −0.08 / −0.26 |
| Text   | 43.63 / 72.26 / 81.92 | **43.69 / 72.62 / 82.22** |   +0.06 / +0.36 / +0.30 |
| Mixed  | 97.45 / 99.67 / 99.86 | **97.56 / 99.68 / 99.84** |  +0.12 / +0.01 / −0.02 |

除 Synthetic Sketch R@1 提高 0.90 个百分点外，其余 Synthetic 变化都很小，主要置信区间包含 0，可视为基本保持原有 Synthetic 检索能力。

### 6.6 结果解释

C2 是四种优化中唯一在正式 epoch-10 checkpoint 上全面提高 Human Sketch R@1/5/10，并提高 Human Mixed R@1/5 的方法。与此同时 Synthetic mixed 基本不变，说明提升不是通过牺牲合成域性能获得的。

它与 Segment-only 的关键区别是：

1. 不把人工草图差异简化为一种结构删除；
2. 保留像素级随机性，并把多种扰动约束在同一个完整度预算内；
3. 用双视图一致性显式要求不同风格版本保持相同语义表示。

需要注意，当前只运行了完整 C2 组合实验，没有分别运行“仅复合增强”和“仅一致性损失”。因此可以声称 **完整 C2 方法有效**，但不能从现有实验中分别量化每个组成部分的独立贡献。

**结论：C2 是当前唯一得到 Human 域正向且有统计支持的优化方案。**

---

## 7. 四种方法综合对比

下表只列最能反映草图域泛化和多模态检索能力的 R@1：


| 方法            |       Human Sketch |        Human Mixed |   Synthetic Sketch |    Synthetic Mixed |
| --------------- | -----------------: | -----------------: | -----------------: | -----------------: |
| 普通方法 A      |              12.60 |              49.27 |              89.16 |              97.45 |
| 联合 Gate       |    11.16（−1.44） |    46.96（−2.31） |     89.34（+0.18） |    97.41（−0.03） |
| 冻结可靠性 Gate |    11.52（−1.08） |    48.02（−1.25） |    84.26（−4.90） |     97.52（+0.08） |
| Segment-only    |     9.98（−2.62） |    46.55（−2.72） | **90.34（+1.18）** | **97.73（+0.28）** |
| **C2**          | **14.08（+1.48）** | **50.46（+1.19）** |     90.06（+0.90） |     97.56（+0.12） |

综合实验支持以下结论：

1. **增加可学习参数不等于性能一定提高。** 两种 Gate 都没有学出有意义的逐查询可靠性分化；
2. **只优化 Synthetic 指标不能证明域泛化改善。** Segment-only 在 Synthetic 上最好，却在 Human 上下降最多；
3. **主要瓶颈是 Synthetic-to-Human 草图分布差异，而不是平均融合表达能力不足；**
4. **多因素 Human-style 扰动与特征一致性约束相结合，才产生了稳定的 Human 域收益。**

可用于论文的总结表述为：

> 固定平均融合已经能够获得大部分 text-sketch 互补收益，直接引入可学习 Gate 或单一连通笔画删除均未改善人工草图检索。相比之下，Human-style 复合增强在统一完整度预算下联合模拟线宽、局部缺口、轮廓简化、非刚性形变和低概率结构缺失，并利用双视图一致性约束保持语义表示稳定，最终在基本保持 Synthetic 性能的同时显著提高 Human Sketch 与 Human Mixed 检索性能。

## 8. 实验产物


| 方法            | 训练配置                                                                                                                                    | Human 指标                                                                                               | Synthetic 指标                                                                                                     |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 普通方法 A      | [`configs/train_coco_clip_10ep_gpu12.yaml`](../configs/train_coco_clip_10ep_gpu12.yaml)                                                     | [`test_clip_10ep_gpu12_last/metrics.json`](../outputs/test_clip_10ep_gpu12_last/metrics.json)                       | [`test_clip_10ep_gpu12_synthetic/metrics.json`](../outputs/test_clip_10ep_gpu12_synthetic/metrics.json)                       |
| 联合 Gate       | [`configs/train_coco_gate_joint_gpu12.yaml`](../configs/train_coco_gate_joint_gpu12.yaml)                                                   | [`test_gate_joint_10ep_gpu12_last/metrics.json`](../outputs/test_gate_joint_10ep_gpu12_last/metrics.json)           | [`test_gate_joint_10ep_gpu12_synthetic/metrics.json`](../outputs/test_gate_joint_10ep_gpu12_synthetic/metrics.json)           |
| 冻结可靠性 Gate | [`configs/train_coco_frozen_gate_grid_gpu45.yaml`](../configs/train_coco_frozen_gate_grid_gpu45.yaml)                                       | [`test_frozen_gate_grid_3ep_gpu45_last/metrics.json`](../outputs/test_frozen_gate_grid_3ep_gpu45_last/metrics.json) | [`test_frozen_gate_grid_3ep_gpu45_synthetic/metrics.json`](../outputs/test_frozen_gate_grid_3ep_gpu45_synthetic/metrics.json) |
| Segment-only    | [`configs/train_coco_clip_segment_10ep_gpu14.yaml`](../configs/train_coco_clip_segment_10ep_gpu14.yaml)                                     | [`test_clip_segment_10ep_gpu14_human/metrics.json`](../outputs/test_clip_segment_10ep_gpu14_human/metrics.json)     | [`test_clip_segment_10ep_gpu14_synthetic/metrics.json`](../outputs/test_clip_segment_10ep_gpu14_synthetic/metrics.json)       |
| C2              | [`configs/train_coco_clip_humanstyle_consistency_10ep_auto2gpu.yaml`](../configs/train_coco_clip_humanstyle_consistency_10ep_auto2gpu.yaml) | [`test_clip_humanstyle_epoch010_human/metrics.json`](../outputs/test_clip_humanstyle_epoch010_human/metrics.json)   | [`test_clip_humanstyle_epoch010_synthetic/metrics.json`](../outputs/test_clip_humanstyle_epoch010_synthetic/metrics.json)     |
