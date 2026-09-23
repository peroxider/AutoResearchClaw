# 通用方法图与显式控制流

结构化稿件现在从冻结的 MethodSpec 生成 DiagramSpec，并通过 publication_assets 接入正文、导出、打包和最终验收。没有医学领域预设节点，也不从方法名称猜测算法模块。

## 图的来源和语义

`publication_assets.json` 的 `spec.diagrams` 保存完整规范：节点、输入/输出端口与变量 ID、公式引用、训练/推理阶段分组、有类型的边、MethodSpec 版本、图 ID。

- architecture 图逐一保留方法步骤。箭头只表示 `depends_on` 声明的依赖，不解释成张量流或执行顺序。
- execution 图仅在方法显式声明 `control_flow` 时产生。入口、判断条件、真假分支、循环和停止节点均来自该声明，不由文字模型补造。
- data_flow 图是变量与步骤的二部图。变量到步骤的 `declared_read` 边逐条来自步骤 `inputs`，步骤到变量的 `declared_write` 边逐条来自步骤 `outputs`；同一变量可以保留多个写入者，系统不会猜测唯一生产者。它表达 MethodSpec 声明的读写关系，不冒充观测到的运行时张量轨迹、形状执行或源码等价性。
- dataset_split 图逐个数据集保留冻结版本、任务/指标、split strategy、train/validation/test 行数和 split IDs 哈希。模型可见的 train、validation、test features 与宿主私有 test labels 分组显示；该边界来自 ResearchContract 的文件分离，不冒充操作系统权限隔离。
- research_overview 图把冻结数据集、带角色的方法、带类型的研究问题和宿主评估/稿件绑定压缩为四个完整节点。边只表示协议声明的输入、比较与证据生成路径；图中明确拒绝把这些边解释成因果、实现一致性或科学有效性证明。
- 两类方法图均保留 `implementation_equivalence=unresolved` 和 `termination=unproved`。可达性及存在停止路径不等于每次执行都终止。

当前不推断变量的唯一生产者，也不把某个写入者自动连接到后续读取者。若要表达值版本、跨步骤传递或多阶段共享变量的动态含义，仍需更具体的数据流契约，不能仅凭变量重名补造运行路径。

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

当前受控矢量路径带机器几何评审，不把未经检查的图像模型输出放入正式稿件。检查器在实际 Matplotlib canvas 上核对节点文字位于框内、标题/脚注/边标签位于画布内、节点框不重叠、边路由位于节点框外，并确认全部声明边进入渲染。它最多按 46、40、34 字符三档确定性换行宽度做有界修复；每次失败原因和最终尝试次数写入 `publication_assets.json`，非换行类错误立即失败，预算耗尽仍失败。SVG 使用文本节点，PDF 为矢量输出，PNG 供 Markdown 预览；复现入口与其他图表一致，为 `publication_assets/reproduce.py`。

每条边使用独立路径和端点位置，循环用虚线，文字不经过数学表达式解析。节点文本进行实际边界检查；字体按 DejaVu Sans 和可用中文字体选择，字体文件摘要绑定渲染环境。缺字和过长标签均报错，不静默裁剪。第四十一轮为超过 8 个节点的方法图增加确定性分页，第四十二轮生成的 data_flow 图沿用同一机制：真实节点按声明/拓扑顺序分组，每个节点恰好出现在一页；页内边照常绘制，跨页边在两个端点页各保存完整 source/target/type/label/loop 与页码，并写入相关节点的 continuation 文本。每页拥有独立 DiagramSpec ID 和 SVG/PNG/PDF，稿件逐页引用。原始方法契约仍限制 24 节点、48 边；分页不绕过源规范上限，也不把跨页边画成同页直接连线。

来源版本、渲染代码、Matplotlib、字体摘要、输出文件摘要均进入资产版本。方法图从当前 MethodSpec 重新派生，切分图从可移植复核后的 ResearchContract 重新派生，再核对确定性重渲染摘要；改标签/边/端口/样本数后重新计算输出哈希仍不能通过。更改控制流或冻结数据切分会使图表及 ManuscriptIR 缓存失效。

已用实际 SVG/PNG/PDF 测试确定性、字符保留、真假分支、回环和跨页 continuation；已人工查看一个生成的流程图。第四十三轮的 `matplotlib_geometry/v1` 是机器几何检查，不是人类审美、图意理解或整稿视觉认证；全稿缩放后的字号、复杂交叉边可读性和全文视觉质量仍需生产稿逐页验收。

第四十五轮已为既有 Stage 22 direct/hybrid 图像模型路径补齐 API 实际 prompt、逐提供方尝试、选定 provider/model、未修改原图/候选图和最终图的哈希证据链；direct 原图单独保存，fallback 明确没有原图。该路径尚未成为共源 DiagramSpec 的正式出版主路径；模型原始输出的独立语义/审美评审与局部修复仍待后续接入。
