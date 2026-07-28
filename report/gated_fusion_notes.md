# Plan: 联合训练可学习门控融合（gated fusion）

## Context（为什么做这件事）

当前融合是固定等权平均 `q = (text + sketch)/2`（[model.py:468](../code/clip/model.py#L468)）。前序分析表明：mixed 相对 text 把 19.5% 的查询救回 R@1，但也把 13.9% 本已命中的 R@1 打掉（弱草图污染强文本），oracle 门控上界 R@1=0.632 vs 当前 0.493，有约 14 个点的"门控头室"被浪费。

目标：把标量 0.5 换成**逐查询、可学习的门控 α(text, sketch)**，`q = α·text + (1-α)·sketch` 再归一化；**encoder 与 gate 联合训练**（用户明确要求直接联合，不冻结）。

硬约束：**兼容原本的训练与测试方法**——现有 `avg` 配置/checkpoint 的训练与评估行为必须完全不变；新增 `gate` 模式可训练可评估；新旧 checkpoint 都能加载。

## 关键事实（已核实）

* 训练时 `query_dropout=0.2`（[data.py:143-148](../src/tsbir/data.py#L143-L148)）会注入"纯白草图图""空 caption"，与 eval 的 `blank_sketch_feature`/`empty_text_feature`（[evaluate.py:261-265](../src/tsbir/evaluate.py#L261-L265)）**同源**——空白特征 in-distribution，gate 能学会处理空白输入（α→1 / α→0）。**`query_dropout` 必须保持开启**。
* 训练里 `feature_fuse` 接收的是 **fp32** 特征（[train.py:276-278](../src/tsbir/train.py#L276-L278) 先 `.float()`），eval 同样 fp32（[evaluate.py:264-265](../src/tsbir/evaluate.py#L264-L265)）。gate 可在 fp32 下稳定计算。
* `feature_fusion` 已从 config 读入（[train.py:60](../src/tsbir/train.py#L60)）；`normalize_fused_query` 默认 True，融合后 `F.normalize` 已存在（[model.py:471-472](../code/clip/model.py#L471-L472)），无需改。
* train/eval 入口保持 gate 为 fp32；原 Notebook 会调用 `convert_weights`，因此 `build_model`/`convert_weights` 路径也必须保留 gate fp32 并支持从 checkpoint 推断 gate hidden dim。
* DDP 下 `module = ddp.module`（[train.py:222/225](../src/tsbir/train.py#L222)），`module.feature_fuse(...)`（[train.py:279](../src/tsbir/train.py#L279)）在 ddp.forward 之外调用——与现有 `classify`/`decoder` 头同模式，gate 梯度正常回传与同步，`find_unused_parameters=false` 不受影响。
* 训练**无任何冻结**（`freeze_nonfc` 从未被调用），gate 默认参与训练。

## 改动清单

### 1. model.py — 新增 gate 模块 + `gate` 融合分支

**(a) `CLIP.__init__`（[model.py:248-266](../code/clip/model.py#L248-L266) 签名；注册点在 [model.py:350-351](../code/clip/model.py#L350-L351) 之后、[model.py:353](../code/clip/model.py#L353) `initialize_parameters()` 之前）**

* 签名新增两个带默认值的 kwargs（不破坏现有调用，evaluate.py 用 `**config` + 默认值即可）：
  * `gate_hidden_dim: int = 256`（架构常量，**train/eval 必须一致**，故固定默认值，不做 per-config）
  * `gate_alpha_init: float = 0.5`（仅影响初始化偏置，不影响架构；eval 用默认即可）
* 在 classification 头之后注册：
  ```python
  self.gate = nn.Sequential(
      nn.Linear(embed_dim * 3, gate_hidden_dim),
      nn.GELU(),
      nn.Linear(gate_hidden_dim, 1),
  )
  # 初始化：末层权重小、偏置置 logit(alpha_init) → 起步 α≈常数(0.5)，冷启动≈avg，再逐查询分化
  nn.init.zeros_(self.gate[-1].weight)
  nn.init.constant_(self.gate[-1].bias, float(np.log(gate_alpha_init / (1 - gate_alpha_init))))
  ```

**(b) `feature_fuse`（[model.py:465-473](../code/clip/model.py#L465-L473)）新增 `gate` 分支**，输入用归一化特征 + Hadamard 积捕获两路一致性：

```python
elif self.feature_fusion == 'gate':
    with torch.autocast(text_features.device.type, enabled=False):  # sigmoid/α 在 fp32 算
        t = text_features.float(); s = sketch_features.float()
        gate_in = torch.cat([t, s, t * s], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_in))            # [B, 1]
    fused = alpha * text_features + (1.0 - alpha) * sketch_features
```

末尾 `F.normalize`（`normalize_fused_query`）保持不变。`else` 分支的报错保留。

### 2. train.py — gate 独立 LR 组 + 放宽 strict 加载

**(a) `parameter_groups`（[train.py:115-127](../src/tsbir/train.py#L115-L127)）** 新增第三个组（按参数名 `"gate" in name` 过滤），gate 组带自己的 `lr`：

```python
def parameter_groups(model, weight_decay, gate_lr=None):
    decay, no_decay, gate = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if "gate" in name:
            gate.append(p); continue
        (no_decay if p.ndim < 2 or name.endswith("bias") or name == "logit_scale" else decay).append(p)
    groups = [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
    if gate:
        groups.append({"params": gate, "lr": gate_lr, "weight_decay": weight_decay})
    return groups
```

* 调用处（[train.py:226-231](../src/tsbir/train.py#L226-L231)）：`gate_lr = float(config["train"].get("gate_learning_rate", config["train"]["learning_rate"]))`，传入 `parameter_groups(model, wd, gate_lr)`。
* 机制依赖：PyTorch 中**带 `lr` 的组用自己的 lr，不带 `lr` 的组退回 optimizer 级默认 lr**——故 encoder 两组行为不变。`LambdaLR`（[train.py:130-138](../src/tsbir/train.py#L130-L138)）按相同 λ 缩放所有组，gate 自动跟随同样的 warmup+cosine 形状。无需改 scheduler。

**(b) 受控兼容 checkpoint**：训练初始化时用 `strict=False` 取得键差异，但只允许所选初始化方式预期缺失的 `gate.*`/辅助头；任何 backbone 缺失或多余键立即报错。

* 对 `avg` 模式：无缺失键 → 行为与 strict=True **完全一致**（不掩盖任何错误）。
* 对 `gate` 模式：backbone checkpoint 缺 `gate.*` 键 → 容忍，gate 保持构造初始化。

**(c) 读取 alpha\_init**：`load_model`（[train.py:53-63](../src/tsbir/train.py#L53-L63)）从 `config["model"].get("gate_alpha_init", 0.5)` 读，传给 CLIP 构造（`gate_hidden_dim` 用默认 256 不传）。

> 不改动：decoder 的 `context = ((image_features + sketch_features)/2)`（[train.py:292](../src/tsbir/train.py#L292)）走 image+sketch，与检索 gate 无关；分类损失（[train.py:287-289](../src/tsbir/train.py#L287-L289)）不经融合。gate 仅由对比损失（[train.py:280-284](../src/tsbir/train.py#L280-L284)）驱动梯度——正是想要的。

### 3. evaluate.py — CLI 选融合模式 + 三模式走 feature\_fuse + 放宽 strict

**(a) CLI**（[evaluate.py:207-230](../src/tsbir/evaluate.py#L207-L230)）新增：

```python
parser.add_argument("--feature-fusion", default="auto", choices=["auto", "avg", "gate"],
                    help="默认从 checkpoint 自动识别，显式模式不匹配时直接报错")
```

**(b) `load_model`**：优先读取 checkpoint 内嵌配置，并以 `gate.*` 键交叉校验；自动构造匹配模型后使用 `strict=True`，避免把 gate checkpoint 静默当成 avg 评测。

**(c) 三模式统一走 `model.feature_fuse`**（替换 [evaluate.py:268](../src/tsbir/evaluate.py#L268) 与 [evaluate.py:299-302](../src/tsbir/evaluate.py#L299-L302) 的手算 `(a+b)/2`）：

```python
# sketch-only（循环外，targets=target_indices）
sketch_query = model.feature_fuse(empty_text_feature, sketch_features)
# 循环内
queries = {
    "text":  model.feature_fuse(text_features, blank_sketch_feature),
    "mixed": model.feature_fuse(text_features, current_sketches),
}
```

* **对 avg：** `feature_fuse` 内部就是 `(a+b)/2`+归一化，输入特征均已归一化 → 与原手算**逐位等价**，旧 checkpoint 指标不变。
* **对 gate：** mixed 走学到的 gate；text/sketch-only 走 `gate(., blank)`——因训练 query\_dropout 见过空白，gate 会输出 α≈1/0，等价于退回单模态。空白特征 in-distribution，无 OOD 问题。

### 4. 新增训练配置（不改现有 yaml）

新增 `configs/train_coco_gate_joint_gpu12.yaml`，复制自 `configs/train_coco_clip_10ep_gpu12.yaml`（**决定：从 OpenAI CLIP 起步**，与现有 clip\_10ep run 直接可比），改动：

* `experiment.name` / `output_dir` → `taskformer_coco_clip_gate_joint_10ep_gpu12`
* `model.feature_fusion: gate`
* `model.gate_alpha_init: 0.5`（可选；冷启动≈avg）
* `train.gate_learning_rate: 1.0e-4`（encoder 仍 1.0e-5，warmup\_steps: 1000 等沿用）
* `init: clip_pretrained` / `checkpoint: model/ViT-B-16.pt`（保持不变）
* 训练侧 strict 自动兼容：`clip_pretrained` 分支（[train.py:97](../src/tsbir/train.py#L97)）本就 `strict=False`，会容忍缺失的 `gate.*` 键并打印日志——2(b) 的 official 分支放宽仅为通用兼容，对本 run 非必需。

## 验证（端到端）

1. **兼容回归**（必须先过）：用现有 `avg` 配置 `--smoke` 跑 2 步，确认 loss 正常、gate 组不存在；用旧 checkpoint 跑 `evaluate.py`（默认 `--feature-fusion auto`），确认 mixed R@1/R@10 与基线 0.4927/0.8507 一致。
2. **gate smoke**：`--config configs/train_coco_gate_joint_gpu12.yaml --smoke`，确认 gate 参数 `.grad` 非零、α 统计写入 CSV/TensorBoard、loss 正常。
3. **正式训练**：10ep on gpu12（与现有 run 对齐）。
4. **评估**：`evaluate.py --feature-fusion gate --checkpoint <new>`，对比：
   * 下界：当前 mixed R@1=0.493（必须超过，否则 gate 退化）
   * 上界：oracle R@1=0.632 / R@10=0.929
   * 合格线：mixed R@1 ∈ (0.493, 0.632)，如 ≥0.55 即有意义提升。
   * 同时记录 α 直方图 + mixed-vs-text 的 rescue/hurt 计数（复用前序对齐脚本），确认救援保留、回退减少。

## 风险与注意

* **α 退化**：可能塌成常数（≈avg，白做）或 →1（永远信文本，丢救援）。靠相似度项 `t*s` 提供逐查询信号 + 监控 α 直方图；必要时后续加批内 α 方差正则（本期不加，先看基线）。
* **架构一致性**：`gate_hidden_dim` 固定 256，避免 train/eval 构造不一致导致 state\_dict 对不上。
* **strict=False 不掩盖真错误**：训练初始化仅容忍白名单内的预期缺失键；正式评测构造匹配架构后使用 `strict=True`。
* **梯度裁剪**：全局 norm（[train.py:308](../src/tsbir/train.py#L308)）覆盖 gate，小模块不会主导裁剪预算，无需改。
