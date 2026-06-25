我按你的意思，把“确保所有环节可落地、可复现、可调试、可投稿”作为目标，重新完善一版。后面我们就围绕这一版逐步实现；遇到实验困难，处理原则是**补强实现、不缩小问题；增加诊断、不降低主张；分层推进、不放弃主线**。

# 题目定稿：状态条件动作代数 Agent

建议把方向正式定为：

# **ActA: Learning Contextual Action Algebra from Environment Probes for Language Agents**

中文可以叫：

# **基于环境探测的状态条件动作代数学习**

一句话定义：

> 现有语言 Agent 通常把交互轨迹当作线性序列。我们认为，Agent 的行动本质上是对环境状态的变换，动作之间存在可学习的结构关系，包括可交换、幂等、逆元、冲突、依赖、吸收等。本文通过黑盒环境 replay 自动探测这些关系，学习状态条件动作代数，并用于轨迹规范化、历史压缩和 Agent 控制。

这个题目的重点不再是“怎么规划下一步”，而是：

> **哪些动作序列在环境中其实等价，哪些顺序是必要的，哪些动作是冗余的，哪些动作会破坏目标状态。**

这和 plan-then-execute 的区别很明确：

| 对比项           | Plan-then-execute | ActA                             |
| ------------- | ----------------- | -------------------------------- |
| 核心对象          | 未来计划序列            | 动作之间的结构关系                        |
| 主要问题          | 下一串动作怎么做          | 哪些动作序列等价或冲突                      |
| 是否要求生成完整计划    | 通常需要              | 不需要                              |
| 是否依赖 LLM 预测未来 | 常见                | 不依赖                              |
| 监督信号          | 成功率、奖励、judge      | 环境 replay 自动生成                   |
| 表示形式          | sequence / tree   | algebra / relation / normal form |
| 主要收益          | 找到可行路径            | 消除冗余、识别依赖、规范轨迹、约束错误              |

---

# 一、把边界划清：避开已有工作

我刚检索了一下，发现如果只做“从轨迹里恢复偏序结构”，会有明显撞题风险。已有 BPOP 方向把 agent traces 看作 latent partial order 的线性扩展，并用贝叶斯方法从成功轨迹中恢复依赖图，目标是生成可复用 SOP 和降低执行成本。这个工作直接说明，**单纯 partial order trace de-linearization 不够安全**。([ar5iv][1])

因此我们的边界必须升级：

| 维度          | BPOP / 偏序轨迹恢复         | ActA / 状态条件动作代数              |
| ----------- | --------------------- | ---------------------------- |
| 数据来源        | 已有成功轨迹                | 黑盒环境主动探测                     |
| 结构类型        | 主要是依赖偏序               | 可交换、幂等、逆元、冲突、依赖、吸收           |
| 是否状态相关      | 倾向 state-free SOP     | 明确建模 (r(s,a,b))              |
| 是否只看成功轨迹    | 是                     | 成功、失败、无效、反事实动作对都用            |
| 是否需要动作序列多样性 | 需要多条线性扩展              | 通过 replay 主动构造 `a;b` 与 `b;a` |
| 控制方式        | 编译成 frontier executor | 轨迹规范化 + 动作过滤 + 候选重排          |
| 论文主张        | 从线性轨迹恢复 SOP           | 从环境变换中学习动作代数                 |

这就是后续论文里必须守住的核心边界：

> **ActA 不是从日志里推偏序，而是把动作看作状态变换，通过环境探测自动学习上下文相关的动作代数。**

---

# 二、把问题讲准：从“轨迹序列”到“动作变换”

形式化地说，环境状态为 (s)，动作 (a) 可以看作一个状态变换：

[
a: s \rightarrow s'
]

两个动作 (a,b) 的关系不是全局固定的，而是状态条件的：

[
R(s,a,b)
]

我们关心六类主要关系。

## 1. 可交换关系

如果：

[
b(a(s)) \equiv a(b(s))
]

则说明在状态 (s) 下，动作 (a) 和 (b) 可交换。

例子：

```text
examine table; look around
```

和：

```text
look around; examine table
```

在某些状态下可能等价。

---

## 2. 幂等关系

如果：

[
a(a(s)) \equiv a(s)
]

则说明动作 (a) 是幂等的。

例子：

```text
open fridge; open fridge
```

第二次 `open fridge` 多数情况下是重复动作。

---

## 3. 逆元关系

如果：

[
b(a(s)) \equiv s
]

则说明 (b) 在该状态附近是 (a) 的逆动作。

例子：

```text
take apple; put apple back
```

在部分环境中可能抵消。

---

## 4. 依赖关系

如果 (b) 在 (s) 下无效，但在 (a(s)) 下有效，则说明：

[
a \prec_s b
]

例子：

```text
open fridge → take apple from fridge
```

`take apple` 依赖 `open fridge` 或 `find apple`。

---

## 5. 冲突关系

如果两个动作都会改变关键状态，但顺序不同导致最终目标状态不同，则说明二者冲突。

例子：

```text
put apple in fridge
put apple in sink
```

同一个 apple 不能同时在两个位置。

---

## 6. 吸收关系

如果：

[
b(a(s)) \equiv a(s)
]

则说明在动作 (a) 执行后，动作 (b) 对关键状态没有新增效果。

例子：

```text
buy item; continue browsing
```

在 WebShop 类任务里，购买后继续浏览通常没有意义。

---

# 三、把方法做实：四个模块闭环

最终系统分为四个模块。

# **模块一：环境适配器**

统一封装 TextWorld、ALFWorld、ScienceWorld、MiniGrid。

接口固定为：

```python
class EnvAdapter:
    def reset(self, task_id: str, seed: int): ...
    def step(self, action: str): ...
    def replay(self, task_id: str, prefix_actions: list[str]): ...
    def admissible_actions(self, state): ...
    def signature(self, state): ...
    def is_equivalent(self, sig1, sig2, mode: str): ...
```

这里最关键的是 `replay` 和 `signature`。

TextWorld 本身就是用于训练和评估文本游戏 RL agents 的 sandbox，环境接口包含 `step` 和 `reset` 这类基本交互能力，适合做第一阶段的自动 replay 和动作关系探测。([GitHub][2])

ALFWorld 包含与 ALFRED 对齐的交互式 TextWorld 环境，目标是让 agent 在抽象文本空间中学习高层策略，再迁移到 embodied household tasks，因此它适合作为主实验环境。([GitHub][3])

ScienceWorld 是面向小学科学课程任务的文本虚拟环境，任务天然包含观察、测量、加热、混合、移动等动作，适合验证动作依赖、不可交换和实验顺序关系。([GitHub][4])

MiniGrid 遵循 Gymnasium API，轻量、快速、可定制，适合做 sanity check 和状态签名验证。([GitHub][5])

---

# **模块二：动作关系探测器**

给定状态 (s) 和动作对 (a,b)，自动执行：

```text
s → a → b → s_ab
s → b → a → s_ba
```

然后比较：

```text
signature(s_ab) == signature(s_ba)
```

由此生成可交换标签。

其他关系同理：

| 关系  | 探测方式              |
| --- | ----------------- |
| 可交换 | 比较 `a;b` 与 `b;a`  |
| 幂等  | 比较 `a` 与 `a;a`    |
| 逆元  | 比较 `a;b` 与原状态     |
| 依赖  | `b` 原本无效，`a` 后变有效 |
| 冲突  | 两种顺序都有效，但关键状态不同   |
| 吸收  | `a;b` 与 `a` 等价    |

这一步不依赖人工标注，不依赖 LLM judge。标签来自环境 replay。

---

# **模块三：动作代数模型**

训练一个小模型：

[
C_\theta(s,a,b,g) \rightarrow r
]

输入：

```text
task goal
current observation
inventory
recent history
action a
action b
```

输出：

```text
commute / idempotent / inverse / dependency / conflict / absorb / unknown
```

第一版可以先不训练，直接用 replay 标签做分析和规则控制。

第二版再训练小模型：

| 模型                          | 用途          |
| --------------------------- | ----------- |
| DeBERTa / BGE cross-encoder | 关系分类器       |
| Qwen 1.5B / 3B LoRA         | 文本动作关系预测    |
| 小型 MLP + embedding          | 快速 ablation |

你的 3090 或 5090 足够做这部分，不需要微调主 Agent。

---

# **模块四：动作代数控制器**

测试时，基础 Agent 仍然生成候选动作：

```text
a_1, a_2, ..., a_K
```

ActA 控制器不直接替代 Agent，而是在执行前检查：

| 检查项        | 处理方式     |
| ---------- | -------- |
| 是否重复幂等动作   | 降权或拒绝    |
| 是否形成可逆空转   | 降权或删除片段  |
| 是否缺少前置依赖   | 要求先做依赖动作 |
| 是否和已完成目标冲突 | 拦截       |
| 是否可交换但顺序低效 | 规范化      |
| 是否对关键状态无贡献 | 降权       |

最终动作选择：

[
\text{score}(a) = \text{LLMScore}(a) - \lambda_1 \text{RepeatRisk}(a) - \lambda_2 \text{ConflictRisk}(a) - \lambda_3 \text{DependencyViolation}(a)
]

这不是 reward model。
它不是问“这个动作好不好”，而是问：

> **这个动作与已有动作和当前状态之间的结构关系是否合理。**

---

# 四、把 benchmark 排好：先稳后强，先通后扩

后续实验不从 WebArena、OSWorld、AppWorld 这类重环境起步。主线按四层推进。

# **第一层：TextWorld，保证现象成立**

TextWorld 是第一阶段主阵地。

目标不是追榜，而是证明：

1. 文本交互环境中确实存在大量动作代数关系；
2. 这些关系可以由环境 replay 自动标注；
3. 小模型可以学习这些关系；
4. 轨迹规范化能保持最终状态不变；
5. 控制器能减少重复、无效、循环动作。

TextWorld 的优势是环境轻、可生成、可 reset、可 replay，最适合做动作关系探测。([GitHub][2])

---

# **第二层：ALFWorld，作为主结果**

ALFWorld 是论文主实验。

理由：

1. 它是标准 LLM Agent 决策环境；
2. 动作是自然语言形式；
3. 任务有明确 success；
4. 轨迹中重复、循环、依赖错误都很常见；
5. ReAct、Reflexion 等经典 Agent 方法都用过 ALFWorld。

ReAct 原论文就在 ALFWorld 和 WebShop 上验证 reasoning + acting 的效果，因此 ALFWorld 作为主结果有认可度。([arXiv][6])

---

# **第三层：ScienceWorld，作为能力扩展**

ScienceWorld 放在第三层。

它比 ALFWorld 更难，但更能体现动作代数的价值，特别是：

```text
mix / pour / heat / cool / measure / focus / connect / disconnect
```

这些动作天然有依赖、冲突和不可交换关系。

ScienceWorld 用于证明 ActA 不只是“家务操作去重”，还可以处理科学实验式动作结构。([GitHub][4])

---

# **第四层：MiniGrid，作为算法验证**

MiniGrid 不一定作为主论文大表，但要保留。

它的作用是：

1. 验证 relation probing 是否正确；
2. 验证状态签名是否可靠；
3. 验证 trajectory normal form 是否保持状态；
4. 验证控制器是否真的减少搜索空间。

MiniGrid 轻量、快、可控，适合做方法 sanity check。([GitHub][5])

---

# 五、把实验设计齐：六组主实验

# **实验一：关系存在性**

问题：

> 动作代数关系在交互环境中是否普遍存在？

做法：

在 TextWorld / ALFWorld 中采样状态和动作对，统计：

```text
commute ratio
idempotent ratio
inverse ratio
dependency ratio
conflict ratio
absorb ratio
unknown ratio
```

预期结果：

```text
大量动作对不是普通序列关系，而是有稳定结构。
```

这组实验支撑论文问题定义。

---

# **实验二：关系可学习性**

问题：

> 这些关系能否被小模型从文本状态和动作中学到？

比较：

```text
Random
Name-similarity baseline
Rule baseline
Embedding baseline
Cross-encoder classifier
Qwen-LoRA classifier
LLM zero-shot classifier
```

指标：

```text
macro-F1
per-relation F1
calibration error
unseen-game F1
unseen-task F1
cross-env transfer F1
```

这里要特别看：

```text
train on TextWorld → test on ALFWorld
train on seen task types → test on unseen task types
```

这组实验支撑“动作代数不是死规则，而是可学习结构”。

---

# **实验三：轨迹规范化**

问题：

> 利用动作代数能否把原始轨迹转为更短、更稳定的 canonical trace？

做法：

对成功轨迹进行重写：

```text
删除幂等重复
删除 no-op
压缩可逆片段
交换可交换动作
保留依赖顺序
阻止冲突合并
```

指标：

```text
compression ratio
state-preservation rate
goal-preservation rate
canonical trace diversity
equivalent trajectory merge rate
```

关键指标是：

```text
规范化后最终状态仍然等价。
```

这组实验支撑“ActA 是结构压缩，不是普通摘要”。

---

# **实验四：Agent 控制效果**

问题：

> ActA 能否提升 Agent 表现？

主表放 ALFWorld，扩展到 TextWorld 和 ScienceWorld。

比较：

```text
ReAct
ReAct + admissible-action filtering
ReAct + history summary
ReAct + no-op cache
Reflexion
LATS
RAP / WKM-style planning baseline
ReAct + ActA rule controller
ReAct + ActA learned controller
```

ReAct 是 reasoning + acting 的经典范式；Reflexion 是通过语言反馈和 episodic memory 强化 Agent，而不是更新模型权重；LATS 是把 reasoning、acting、planning 与 tree search 结合；RAP 则把 LLM 同时作为 world model 和 reasoning agent，并使用 MCTS 风格规划。这些基线必须纳入，才能挡住“你只是过滤无效动作”或“你只是弱化版 planning”的质疑。([arXiv][6])

主指标：

```text
success rate
average steps
invalid action rate
repeated action rate
loop rate
token cost
environment calls
```

优先看三项：

```text
success rate 不下降并提升
average steps 明显下降
loop / repeated / invalid 明显下降
```

---

# **实验五：模块消融**

问题：

> 哪些动作关系真正有效？

消融：

```text
ActA full
w/o commutativity
w/o idempotence
w/o inverse
w/o dependency
w/o conflict
w/o absorb
rule-only
learned-only
no normal form
no controller
```

如果只靠某一类关系有效，论文会显得窄；如果多类关系都有贡献，论文就更稳。

---

# **实验六：失败诊断**

问题：

> ActA 到底修了哪些 Agent 错误？

错误分类：

```text
重复动作
原地循环
无效动作
依赖缺失
目标破坏
错误对象
错误容器
过早提交
无贡献探索
```

输出形式：

```text
before / after error count
case study
trajectory diff
normal form visualization
```

这组实验对写论文非常重要。它能把方法从“提升几个点”讲成“修复了一类结构性错误”。

---

# 六、把实现路线排清：四个阶段推进

# **阶段一：本地环境与 replay 打通**

目标：

```text
TextWorld adapter 跑通
状态 signature 稳定
action pair probing 可批量运行
```

产物：

```text
envs/textworld_adapter.py
probe/relation_probe.py
data/relation_records.jsonl
analysis/relation_stats.ipynb
```

最低验收：

```text
100 个 episode deterministic replay 一致
1000 个 action pair 可自动标注
signature 比较无明显异常
```

---

# **阶段二：动作关系数据集构建**

目标：

```text
构建 ActA-TextWorld 数据
构建 ActA-ALFWorld 数据
形成 train / valid / test split
```

数据格式：

```json
{
  "env": "textworld",
  "task_id": "...",
  "seed": 42,
  "goal": "...",
  "prefix_actions": ["...", "..."],
  "observation": "...",
  "inventory": "...",
  "action_a": "...",
  "action_b": "...",
  "sig_s": "...",
  "sig_ab": "...",
  "sig_ba": "...",
  "relations": {
    "commute": true,
    "dependency": false,
    "conflict": false,
    "idempotent_a": true,
    "idempotent_b": false
  }
}
```

最低验收：

```text
每类关系都有足够样本
标签噪声可估计
不同任务 split 无泄漏
```

---

# **阶段三：关系模型与轨迹规范化**

目标：

```text
训练 Cθ(s,a,b)
实现 normal_form(trace)
验证状态保持
```

产物：

```text
models/relation_classifier.py
rewrite/normalizer.py
eval/rewrite_eval.py
```

最低验收：

```text
关系分类器显著优于 random / embedding baseline
规范化后 state-preservation rate 高
轨迹压缩率有明显收益
```

---

# **阶段四：接入 ReAct 控制器**

目标：

```text
ActA controller 接入 Agent loop
在 ALFWorld 上跑完整对比
```

控制器策略分三档：

```text
soft rerank：只调候选动作分数
hard guard：拦截高置信错误动作
repair request：要求 LLM 换一个动作
```

最低验收：

```text
invalid / repeated / loop 下降
success rate 不低于 ReAct
在部分任务类型上有明确提升
```

---

# 七、把调试原则定死：困难来了不缩口

后面实验一定会遇到问题。处理原则如下。

# **问题一：环境不能 clone**

不缩小方向。

处理：

```text
用 reset + prefix replay 替代 clone
缓存 prefix 状态
对长 prefix 做 checkpoint
```

只要能从初始状态重放到 (s)，就能做 `a;b` 和 `b;a` 探测。

---

# **问题二：状态 signature 不稳定**

不改成“只看文本 observation”。

处理：

建立三层等价：

```text
full-state equivalence
observable-state equivalence
goal-relevant equivalence
```

不同环境用不同层级，但都保留“动作作为状态变换”的主张。

---

# **问题三：动作关系标签噪声大**

不删关系类别。

处理：

```text
重复 replay 验证
加入 unknown 类
用置信度分桶
按关系类型分别调阈值
对冲突和依赖优先保证 precision
```

宁可先让模型保守，也不能把问题缩小成“只做去重”。

---

# **问题四：小模型学不动**

不放弃 learned ActA。

处理：

```text
先做 binary relation，再做 multi-label
先做 TextWorld，再做 ALFWorld
先用 cross-encoder，再上 Qwen-LoRA
先用规则标签训练，再用 LLM 解释作为辅助输入
```

同时保留 rule controller 作为可运行版本。

---

# **问题五：Agent success 提升不明显**

不把论文改成“只做分析”。

处理顺序：

```text
先看 invalid / loop / repeat 是否下降
再看不同任务类型的局部收益
再调整 controller 强度
再从单动作过滤升级到候选重排
再引入 k=3 候选动作
再加入 dependency repair
```

目标是保持主张：

```text
动作代数能改善 Agent 控制。
```

而不是退成：

```text
动作代数只是一个分析工具。
```

---

# **问题六：ScienceWorld 太难**

不取消 ScienceWorld。

处理：

```text
全量结果继续报告
再增加 relation-rich subset 分析
主表给全量，子表解释关系密集任务
```

这不是缩窄，而是增加诊断层次。

---

# 八、把论文结构立住：ICML 叙事框架

论文题目建议：

# **Actions as Transformations: Learning Contextual Action Algebra for Language Agents**

摘要主线：

> Language-agent trajectories are usually represented as chronological action sequences. We argue that this representation misses algebraic structure induced by environment dynamics: actions may commute, repeat idempotently, invert each other, conflict, depend on each other, or become absorbed under specific states. We propose ActA, a framework that probes black-box environments to automatically learn contextual action algebra, rewrites trajectories into canonical forms, and controls LLM agents by filtering and reranking structurally invalid actions. Experiments on TextWorld, ALFWorld, and ScienceWorld show that ActA reduces redundant actions, preserves goal-relevant state under trace normalization, and improves agent efficiency and success without human annotation or fine-tuning the base LLM.

论文贡献写四条：

1. **提出问题**：语言 Agent 轨迹不应只作为线性序列，而应作为状态条件动作代数来建模。
2. **提出方法**：通过黑盒环境 replay 自动探测动作关系，无需人工标注。
3. **提出模型**：学习 (C_\theta(s,a,b))，预测上下文相关动作关系。
4. **提出应用**：用于 trajectory normal form、history compression、action guard、candidate reranking。

---

# 九、把最低可行版本定好

第一步直接做下面这些。

## 代码结构

```text
acta/
  envs/
    base.py
    textworld_adapter.py
    alfworld_adapter.py
    scienceworld_adapter.py
    minigrid_adapter.py

  probe/
    pair_probe.py
    relation_labeler.py
    sampler.py

  signatures/
    text_signature.py
    fact_signature.py
    goal_signature.py

  data/
    build_relation_dataset.py
    split.py

  models/
    relation_classifier.py
    train_classifier.py
    eval_classifier.py

  rewrite/
    normal_form.py
    trace_compressor.py

  agents/
    react_agent.py
    acta_controller.py
    acta_react.py

  eval/
    eval_relations.py
    eval_rewrite.py
    eval_agent.py
    error_analysis.py

  configs/
    textworld.yaml
    alfworld.yaml
    scienceworld.yaml
```

## 第一批只做两个关系

为了跑通闭环，第一批实现：

```text
commute
idempotent
```

但论文主张不缩小。代码结构从一开始支持：

```text
dependency
inverse
conflict
absorb
```

第一批先把框架跑通，后续把关系补齐。

## 第一批只跑 TextWorld

第一批只跑 TextWorld，不是因为题目缩小，而是因为 TextWorld 是最适合验证 replay、signature、pair probing 的基础环境。TextWorld 跑通后，接口无缝迁移到 ALFWorld。

---

# 十、最终定版

接下来我们按这个路线推进：

# **问题名称**

**状态条件动作代数学习**

# **方法名称**

**ActA**

# **核心机制**

```text
environment replay → relation probing → contextual action algebra → trace normal form → algebra-guided agent control
```

# **主 benchmark**

```text
TextWorld：现象与数据构建
ALFWorld：主实验
ScienceWorld：能力扩展
MiniGrid：sanity check
```

# **主基线**

```text
ReAct
Reflexion
LATS
RAP / WKM-style planning
history summarization
no-op cache
admissible-action filtering
```

# **主指标**

```text
success rate
average steps
invalid action rate
loop rate
repeated action rate
trace compression ratio
state-preservation rate
relation F1
token cost
```

# **最重要的判断标准**

这个方向能不能立住，不只看最终成功率，还看四个信号：

```text
动作关系是否普遍存在
动作关系是否可自动学习
轨迹规范化是否保持状态
Agent 控制是否减少结构性错误
```

我建议下一步直接从 **TextWorld adapter + pair probing** 开始。只要第一批 `commute / idempotent` 标签稳定生成，整个项目就有了可执行基础。

[1]: https://ar5iv.labs.arxiv.org/html/2602.02806 "[2602.02806] De-Linearizing Agent Traces: Bayesian Inference of Latent Partial Orders for Efficient Execution"
[2]: https://github.com/microsoft/TextWorld?utm_source=chatgpt.com "TextWorld is a sandbox learning environment ..."
[3]: https://github.com/alfworld/alfworld?utm_source=chatgpt.com "ALFWorld: Aligning Text and Embodied Environments ..."
[4]: https://github.com/allenai/scienceworld?utm_source=chatgpt.com "ScienceWorld is a text-based virtual environment centered ..."
[5]: https://github.com/Farama-Foundation/Minigrid?utm_source=chatgpt.com "Farama-Foundation/Minigrid: Simple and easily ..."
[6]: https://arxiv.org/abs/2210.03629?utm_source=chatgpt.com "Synergizing Reasoning and Acting in Language Models"
