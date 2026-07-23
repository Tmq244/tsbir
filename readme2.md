这个项目目前已经形成了一个完整的 TASK-former 复现闭环：

> COCO 数据下载 → 人工/合成草图整理 → 三模态模型微调 → 草图/文本/混合检索 → 指标统计 → 成功失败案例输出

它对应课程任务书中的绝大多数核心要求，不再只是原仓库提供的 100 张图片 Demo。

## 1. 项目要解决什么问题

传统文本图像检索只能表达“语义”，例如：

> “一个人在骑摩托车。”

但不容易表达人物位置、物体布局和形状。草图恰好擅长表达这些空间信息，却可能画得不准确。

TASK-former 同时使用：

- 文本：描述类别、颜色、动作、语义；
- 草图：描述轮廓、布局、相对位置；
- 图像：作为被检索图库。

核心检索公式可以概括为：

$$
q=\operatorname{Normalize}\left(\frac{f_\text{text}(t)+f_\text{sketch}(s)}{2}\right)
$$

$$
\operatorname{score}(q,I_i)=q^\top f_\text{image}(I_i)
$$

对图库中所有图像计算余弦相似度，取最高的 Top-5。

这种设计属于“后期融合双编码器”：图库图像特征可以提前计算，查询时只编码文本和草图，因此适合大规模检索。

---

## 2. 当前仓库的两部分

### 原论文 Demo

主要包括：

- [README.md](/home/tangmingqiang/cir/tsbir/README.md)
- [Retrieval_Demo.ipynb](/home/tangmingqiang/cir/tsbir/notebooks/Retrieval_Demo.ipynb)
- `images/`：100 张示例图库图片
- `sketches/`：示例查询草图
- `model/tsbir_model_final.pt`：官方预训练模型
- `code/clip/`：修改过的 CLIP/TASK-former 实现

Notebook 的流程是预提取 100 张图片的特征，再用文本和草图的平均特征执行 KNN 检索。它只能算演示，不能满足课程要求中的“自行准备数据和真实训练”。

### 当前扩展的完整复现代码

新增的核心部分是：

- `src/tsbir/`：数据、训练、评测代码
- `configs/`：两卡训练配置
- `scripts/`：数据下载、训练流水线、曲线绘制
- `data/`：完整 COCO 数据和草图
- `outputs/`：训练权重、日志、指标和案例图

需要注意：从 Git 状态看，`src/`、`configs/`、`scripts/` 等目前仍是未提交文件，README 也还没有更新为完整训练版说明。

---

## 3. 数据处理流程

### 数据来源

项目使用：

- COCO 2014 train/val 图片；
- COCO instance annotations，提供 90 维物体类别标签；
- Karpathy split，提供图像检索领域常用的 train/val/test 划分；
- 论文发布的 5000 张人工草图；
- PhotoSketch 生成的合成训练草图。

下载入口是：

- [download_coco.sh](/home/tangmingqiang/cir/tsbir/scripts/download_coco.sh)
- [download_photosketch.sh](/home/tangmingqiang/cir/tsbir/scripts/download_photosketch.sh)

### 数据规模

当前生成的清单统计为：

| 划分 | 图像数 | 文本数 | 草图类型 |
|---|---:|---:|---|
| train | 113,287 | 566,747 | PhotoSketch 合成草图 |
| val | 5,000 | 25,010 | PhotoSketch 合成草图 |
| test | 5,000 | 25,010 | 人工手绘草图 |

共有：

- 123,287 张合成草图；
- 5,000 张人工测试草图；
- 90 维 COCO 多标签；
- `data/` 目录约 42 GB。

### Manifest 结构

[prepare_data.py](/home/tangmingqiang/cir/tsbir/src/tsbir/prepare_data.py) 将不同来源的数据统一成 JSONL。每条记录包含：

```json
{
  "coco_id": 391895,
  "image": "data/coco/val2014/...",
  "synthetic_sketch": "data/synthetic_sketches/...",
  "human_sketch": "data/human_sketches/...",
  "captions": ["caption 1", "..."],
  "category_ids": [1, 2, 4]
}
```

训练样本的基本单位不是一张图，而是“图像—某条 caption—草图”。因此 113,287 张图像展开后得到 566,747 个训练三元组。训练仍保留全部 caption 展开，但使用全局唯一图像批采样器，保证一次对比损失所见的跨 GPU 全局 batch 中同一图像最多出现一次。

---

## 4. 数据增强

数据集实现在 [data.py](/home/tangmingqiang/cir/tsbir/src/tsbir/data.py:102)。

图像和草图都会独立执行：

- ±30° 仿射旋转；
- 最大 30% 平移；
- 剪切和尺度变化；
- 随机裁剪；
- CLIP 均值方差归一化。

图像与草图分别调用各自的变换链，因此旋转、平移、缩放和裁剪参数彼此独立。这会主动打破合成草图与源图像的精确对齐，符合论文训练描述。图像端不再使用水平翻转。

草图在上述增强之前还会：

- 随机删除部分黑色像素，即 stroke dropout。

另外还有 20% 的 query dropout：

- 10% 概率把草图替换成白图；
- 10% 概率把文本替换成空字符串；
- 80% 同时使用文本和草图。

它的目的，是让同一个模型能够处理混合、纯文本、纯草图三种查询。

---

## 5. 模型结构

核心模型位于 [model.py](/home/tangmingqiang/cir/tsbir/code/clip/model.py:247)，基础骨干是 CLIP ViT-B/16：

- 输入分辨率：224×224；
- patch 大小：16×16；
- 图像 Transformer：12 层、宽度 768；
- 文本 Transformer：12 层、宽度 512；
- 文本最大长度：77；
- BPE 词表：49,408；
- 最终三种模态都映射到 512 维。

整体结构如下：

```text
目标图像 ── ViT-B/16 ───────────────┐
                                     ├─ 共享的 512 维检索空间
查询草图 ── 同一个 ViT-B/16 ────────┤
                                     │
查询文本 ── Text Transformer ────────┘
                  │
        文本特征 + 草图特征
                  │
          平均融合并重新归一化
                  │
         与所有图像计算余弦相似度
```

### 图像和草图权重共享

配置中 `weight_sharing: true`，所以：

```python
self.visual2 = self.visual
```

图像和草图由同一个 ViT 编码器处理。这样参数更少，同时强制照片和线稿进入相同视觉空间。

### 分类头

每种模态的 512 维特征都会经过：

```text
512 → 1024 → ReLU → 90
```

它要求图像、文本和草图特征都能预测 COCO 物体类别。

### Caption Decoder

模型还包含一个六层、八头的自回归 Transformer Decoder。它根据图像和草图的平均特征生成原始 caption。

这个 Decoder 不是检索阶段所必需的，而是训练时的辅助任务：迫使视觉特征保留足够的语义信息。

---

## 6. 三个训练损失

训练核心位于 [train.py](/home/tangmingqiang/cir/tsbir/src/tsbir/train.py:221)。

### 对比检索损失 \(L_e\)

文本和草图先融合，并对融合结果重新进行 L2 归一化：

$$
q_i=\operatorname{Normalize}\left(\frac{t_i+s_i}{2}\right)
$$

然后使用 CLIP 式双向对称交叉熵：

$$
L_e=\frac{1}{2}
\left[
CE(qI^\top,y)+CE(Iq^\top,y)
\right]
$$

既要求查询找到图像，也要求图像找到查询。文本、草图和图像特征会先分别归一化；融合 query 再次归一化后，训练 logits 是严格的 cosine similarity，并且与评测路径保持一致。

DDP 训练时会通过 `all_gather` 收集所有 GPU 的特征，使负样本范围扩展到跨全部 rank 的全局 batch，实现在 [losses.py](/home/tangmingqiang/cir/tsbir/src/tsbir/losses.py:43)。这不局限于两张 GPU，实际 GPU 数由 `torchrun` 的 `WORLD_SIZE` 决定。

### 全局唯一图像批采样

[DistributedUniqueImageBatchSampler](/home/tangmingqiang/cir/tsbir/src/tsbir/data.py:165) 保留 566,747 个 caption pair，并按照每张图剩余的 caption 数构造 batch。某张图只有在当前全局 batch 完成后才会重新进入候选队列，因此：

- 单卡局部 batch 内不会重复图像；
- 多卡 `all_gather` 后的全局 batch 内也不会重复图像；
- 不同 caption 不会在对角线式 CLIP loss 中互相成为假负样本；
- 支持任意 `world_size`，并非为两张 GPU 写死。

真实训练清单在两卡、每卡 batch 96 下，每个 epoch 调度 566,746/566,747 个 pair，共 2,952 个全局 batch。由于总 pair 数为奇数，而 DDP 要求各 rank 局部 batch 等长，每个 epoch 会随机舍弃 1 个 pair；不同 epoch 舍弃对象会变化。一般情况下最多舍弃 `world_size - 1` 个尾部 pair。

### 多标签分类损失 \(L_c\)

分别对图像、草图、文本预测 90 维 COCO 类别，再取平均：

$$
L_c=\frac{L_\text{img}+L_\text{sketch}+L_\text{text}}{3}
$$

项目使用 Asymmetric Loss，负类别采用更强的聚焦系数，缓解 COCO 多标签中负样本远多于正样本的问题。

### Caption Decoder 损失 \(L_d\)

Decoder 的上下文是：

$$
c=\frac{f_\text{image}+f_\text{sketch}}{2}
$$

使用 teacher forcing 逐 token 预测原 caption，并计算 token 交叉熵。

### 总损失

和论文一致：

$$
L=100L_e+10L_c+L_d
$$

虽然 \(L_e\) 数值最小，但乘以 100 后仍是主要优化目标。

---

## 7. 训练策略

10 epoch 主实验配置见 [finetune_coco_ddp_10ep_gpu23.yaml](/home/tangmingqiang/cir/tsbir/configs/finetune_coco_ddp_10ep_gpu23.yaml)。

主要参数：

- 初始化：官方 `tsbir_model_final.pt`；
- GPU：配置记录为物理 GPU 2、3；
- DDP 进程数：2；
- 每卡 batch：96；
- 全局 batch：192；
- epoch：10；
- 优化器：AdamW；
- 初始学习率：\(10^{-6}\)；
- 最低学习率：\(10^{-7}\)；
- warmup：100 step；
- cosine decay；
- weight decay：0.1；
- bf16 自动混合精度；
- gradient checkpointing；
- 梯度裁剪：1.0。

当前采样器下，10 epoch、两卡、每卡 batch 96 共执行约 29,520 次 optimizer update。此前已经完成的历史实验使用旧采样器，共记录约 29,510 次 update、22,636 秒，即约 6 小时 17 分钟；历史结果不会因代码更新而自动改变。

### 使用更多 GPU

采样器和对比损失都按运行时 `world_size` 工作。假设使用 4 张 GPU、每卡 batch 48，需要同步修改：

```yaml
train:
  batch_size_per_gpu: 48
  expected_world_size: 4
  global_batch_size: 192

distributed:
  nproc_per_node: 4
  devices: [0, 1, 2, 3]
```

启动示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc-per-node=4 \
  -m tsbir.train --config configs/finetune_coco_ddp.yaml
```

`expected_world_size` 是强校验，必须与 `--nproc-per-node` 一致。`distributed.devices` 只用于记录，实际可见设备仍由 `CUDA_VISIBLE_DEVICES` 决定。全局 batch 大小为：

$$
B_\text{global}=B_\text{per-GPU}\times\text{world\_size}
$$

### checkpoint 保存与模型选择

训练过程不再在每个 epoch 结束后运行合成草图验证。每个 epoch 固定保存：

- `epoch_001_weights.pt`、`epoch_002_weights.pt` 等纯模型权重；
- `last_weights.pt`，始终覆盖为最后一个 epoch 的模型；
- `last.ckpt`，包含最后模型、优化器和学习率调度器状态。

默认选择最后一个 epoch，而不是根据合成验证集选择 `best_weights.pt`。评测脚本默认读取 `outputs/taskformer_coco_official_finetune/last_weights.pt`。

完整流水线入口是 [run_training_pipeline_2gpu.sh](/home/tangmingqiang/cir/tsbir/scripts/run_training_pipeline_2gpu.sh)，依次执行：

1. 等待 COCO 数据准备完成；
2. 生成 manifest；
3. 用 PhotoSketch 生成合成草图；
4. 检查草图数量；
5. 等待目标 GPU 空闲；
6. 执行两步 smoke test；
7. 启动正式 DDP 微调。

---

## 8. 损失变化

10 个 epoch 的平均总损失从：

- epoch 1：38.17
- epoch 5：21.91
- epoch 8：20.99
- epoch 10：20.32

主要是对比损失从 0.230 降到约 0.070。

分类损失从 1.20 降至约 1.03，Decoder 损失则长期停留在 3.0 左右。这说明：

- 检索目标学习很快；
- 分类辅助任务略有改善；
- Caption Decoder 基本进入平台期，对后期训练的贡献有限。

生成的损失曲线位于：

[loss_curves_10epochs.png](/home/tangmingqiang/cir/tsbir/outputs/taskformer_coco_official_finetune_10ep_gpu23_bs96/loss_curves_10epochs.png)

---

## 9. 三种检索如何评测

评测代码位于 [evaluate.py](/home/tangmingqiang/cir/tsbir/src/tsbir/evaluate.py:155)。

测试图库固定为 5000 张 COCO 图像：

- 草图检索：5000 个查询；
- 文本检索：25,010 个 caption 查询；
- 混合检索：25,010 个 caption+对应草图查询。

这里的“单模态”仍然沿用模型训练形式：

- 草图检索 = 草图特征 + 空文本特征；
- 文本检索 = 文本特征 + 白色草图特征；
- 混合检索 = 文本特征 + 人工草图特征。

因此它不是完全绕过另一个模态，而是用空输入占位。这与论文的 query dropout 设计一致。

评测输出包括：

- R@1、R@5、R@10；
- MRR；
- median rank、mean rank；
- 每个查询的 Top-5 COCO ID；
- 三种模式各自的成功和失败案例图。

---

## 10. 历史实验结果

| 模型 | 模式 | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| 官方权重 | 草图 | 20.30% | 36.32% | 43.42% |
| 官方权重 | 文本 | 44.37% | 73.70% | 83.65% |
| 官方权重 | 混合 | 62.58% | 87.72% | 94.35% |
| 微调 1 epoch | 草图 | 15.44% | 28.70% | 35.88% |
| 微调 1 epoch | 文本 | 42.18% | 71.44% | 81.99% |
| 微调 1 epoch | 混合 | 55.46% | 81.45% | 89.53% |
| 微调 10 epoch | 草图 | 14.14% | 25.58% | 32.16% |
| 微调 10 epoch | 文本 | 43.71% | 73.16% | 83.33% |
| 微调 10 epoch | 混合 | 51.57% | 77.64% | 86.15% |

以下是修改训练策略之前已经完成的历史实验结果。最重要的历史结论是：

> 旧训练流程中损失不断下降，合成草图验证集的 mixed R@5 高达 99.1%，但人工草图测试性能反而下降。

这不是“模型没有训练成功”，而是明显的域偏移和过拟合：

- train/val 使用 PhotoSketch 合成线稿；
- test 使用人类凭记忆绘制的自由草图；
- 合成草图保留了目标图像大量边缘和空间细节；
- 人工草图更抽象、遗漏更多、风格差异更大；
- 验证集与测试集分布不同，所以 99% 验证 R@5 不能预测真实人工草图效果。

草图 R@5 从官方权重的 36.32% 降到 25.58%，进一步说明长时间合成草图微调损害了人工草图泛化能力。文本分支下降较少，是因为文本分布没有发生同等程度的变化。

---

## 11. 当前实现值得注意的问题

### 当前验证与选择策略

旧流程依据合成草图 val mixed R@5 选择最佳权重，但最终关注的是人工草图测试性能，两者存在明显域差异。当前代码已经取消逐 epoch 合成验证和 `best_weights.pt` 选择，默认使用最后一个 epoch 的 `last_weights.pt`。

这种策略避免被不可靠的合成验证指标误导，但也失去了 early stopping。若以后需要根据指标选择 checkpoint，应另外准备不与 test 重叠的人工草图验证集。

### 配置文件有部分字段没有真正生效

例如：

- `architecture` 和 `embed_dim`：训练代码固定读取 ViT-B-16 JSON；
- `decoder_depth`、`decoder_heads`：模型中硬编码为 6 和 8；
- `precision`：代码固定使用 bf16；
- `gather_contrastive_features`：代码始终 gather；
- `distributed.devices`：真正设备由 shell 的 `CUDA_VISIBLE_DEVICES` 决定；

因此 YAML 目前更像实验记录，而不是所有参数都可配置的唯一事实来源。

### Caption Decoder 收益可能有限

Decoder loss 长期在 3.0 附近，后期改善很小，却占用大量参数、显存和计算。可以做消融实验判断是否值得保留。

### 同图多 caption 的处理

数据仍以 caption 为样本单位，但当前全局批采样器保证同一图像不会在一次对比损失的全局 batch 中出现两次，因此不会再把同图不同 caption 当作负样本。如果未来绕过该采样器或允许同图重复，则必须改用基于 `coco_id` 的多正样本对比损失。

### 工程可复现性仍需收尾

- README 仍只介绍旧 Demo；
- 训练扩展代码尚未提交；
- `outputs/` 约 18 GB；
- 缺少自动化测试；
- 训练脚本绑定特定 GPU 编号；
- 新采样与增强代码已通过语法、配置和真实 manifest 不变量测试，但尚未重新执行完整 GPU 训练。

---

## 12. 建议的改进方向

最有针对性的方案是“缩小合成草图与人工草图的域差异”：

1. 加入更接近人类绘画的增强：

   - 连通笔画级 dropout，而不是独立像素删除；
   - 笔画粗细、抖动、断裂和简化；
   - 局部物体遗漏；
   - 位置和尺度的大范围扰动。

2. 用少量人工草图做验证或参数高效微调。

3. 训练时混入 official checkpoint 蒸馏损失，约束新模型不要偏离原有人工草图能力。

4. 对文本和草图使用可学习门控融合：

$$
q=\alpha(t,s)f_t+[1-\alpha(t,s)]f_s
$$

让模型根据草图质量动态选择依赖文本还是草图，而不是永远各占一半。

5. 如果未来需要在同一全局 batch 中放入同图的多个 caption，可将当前唯一图像采样改成基于 `coco_id` 的多正样本对比学习。

总体而言，这个项目的工程闭环已经完成。历史实验最值得写进报告的发现，不是简单宣称“微调提升了性能”，而是合成域损失和验证指标持续改善时，人工草图泛化仍可能显著下降。当前训练流程已经移除基于合成验证集的选模，但合成草图到真实自由手绘草图之间的域鸿沟仍然是核心问题。
