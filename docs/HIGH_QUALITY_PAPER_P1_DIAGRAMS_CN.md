# 通用方法图与显式控制流

结构化稿件现在从冻结的 MethodSpec 生成 DiagramSpec，并通过 publication_assets 接入正文、导出、打包和最终验收。没有医学领域预设节点，也不从方法名称猜测算法模块。

## 图的来源和语义

`publication_assets.json` 的 `spec.diagrams` 保存完整规范：节点、输入/输出端口与变量 ID、公式引用、训练/推理阶段分组、有类型的边、MethodSpec 版本、图 ID。

- architecture 图逐一保留方法步骤。箭头只表示 `depends_on` 声明的依赖，不解释成张量流或执行顺序。
- execution 图仅在方法显式声明 `control_flow` 时产生。入口、判断条件、真假分支、循环和停止节点均来自该声明，不由文字模型补造。
- 两类图均保留 `implementation_equivalence=unresolved` 和 `termination=unproved`。可达性及存在停止路径不等于每次执行都终止。

当前不自动生成 introduction 概览或实验切分图；也不推断变量的唯一生产者。多阶段共享变量需要更具体的数据流契约，不能仅凭变量重名补画边。

## 可选控制流契约

在原有 MethodSpec 中添加下面的字段，示例引用既有 `predict` 步骤。`stopping_rule` 应同步描述批次耗尽的停止条件：

```yaml
control_flow:
  entry: run
  nodes:
    - {id: run, kind: step, step: predict}
    - {id: check, kind: decision, phase: inference, condition: '是否还有批次？'}
    - {id: done, kind: stop, phase: inference}
  edges:
    - {source: run, target: check, branch: next, loop: false}
    - {source: check, target: run, branch: 'true', loop: true}
    - {source: check, target: done, branch: 'false', loop: false}
```

每个方法步骤必须恰好映射一次；step 只有一条 next 出边，decision 必须同时有 true/false，stop 没有出边。所有节点必须从入口可达且有通向 stop 的路径。移除循环边后必须无环，循环边必须确实闭合一条反馈路径。必要的步骤依赖必须支配其使用步骤，即每条到达该步骤的路径都先经过依赖步骤。步骤依赖 DAG 仍不允许环，迭代在独立控制流中表达。

条件是受长度限制的文本，不作为 Python/TeX 执行。校验器不自动证明条件对应源码、停止描述与判断等价或运行时条件满足。

## 渲染、失效与复现

当前没有接入视觉拓扑评审器，因此明确选择受控矢量后备路径，不把未经检查的图像模型输出放入正式稿件。SVG 使用文本节点，PDF 为矢量输出，PNG 供 Markdown 预览；复现入口与其他图表一致，为 `publication_assets/reproduce.py`。

每条边使用独立路径和端点位置，循环用虚线，文字不经过数学表达式解析。节点文本进行实际边界检查；字体按 DejaVu Sans 和可用中文字体选择，字体文件摘要绑定渲染环境。缺字、超大图、过长标签均报错，不静默裁剪。单图上限 24 节点、48 边；需要进一步拆图时必须修改研究规范，当前不自动选择子集。

来源版本、渲染代码、Matplotlib、字体摘要、输出文件摘要均进入资产版本。校验从当前 MethodSpec 重新派生图规范并核对确定性重渲染摘要；改标签/边/端口后重新计算输出哈希仍不能通过。更改控制流使图表及 ManuscriptIR 缓存失效。

已用实际 SVG/PNG/PDF 测试确定性、字符保留、真假分支及回环；已人工查看一个生成的流程图。全稿缩放后的字号、复杂交叉边布局和全文视觉质量仍为 `unavailable`，这次单图检查不替代整稿视觉验收。

图像模型主生成、保留模型/提示词/原图的有界视觉修复、专业数据流图及大图分页仍待后续接入。
