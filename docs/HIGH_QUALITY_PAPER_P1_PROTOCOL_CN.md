# P1 第二轮：预声明实验矩阵与证据覆盖

后续已增加每个 RQ 的可选 `analysis_plan`：分析类型、图表、条件区间及计算预算在执行前冻结，详见 [AnalysisSpec 说明](HIGH_QUALITY_PAPER_P1_ANALYSIS_CN.md)。未声明时只做配对种子的描述性汇总，不自动加入显著性检验。

本说明记录第二轮实现范围。第三轮已补充宿主机逐单元调度、正式矩阵墙钟预算账本、失败记录和测试冻结；执行方式及当前边界以 [P1 宿主机执行说明](HIGH_QUALITY_PAPER_P1_EXECUTION_CN.md) 为准。

本轮在 ResearchBrief 和固定数据切分之上增加实验协议。目的在于执行前声明必须完成的比较，并在分析和最终验收时逐项检查，避免模型临时删掉难跑的基线、随机种子或不利结果。

支持范围：本地 CSV 输入流程、按 seed 的逐次独立评估、主比较、消融、敏感性和泛化问题。协议层预算是**预声明的等额分配**；后续宿主执行层已记录正式矩阵墙钟用量，并在第三十四轮增加 validation-only 调参披露、逐单元实际 trial 计数和最终冻结参数绑定。该披露仍不自动证明公平调参，具体边界见执行说明。

## 启用

沿用 [P1 数据接入说明](HIGH_QUALITY_PAPER_P1_INPUTS_CN.md) 的配置与 DatasetManifest，在 brief 中增加：

```yaml
protocol_path: protocol.yaml
```

路径相对 brief 文件目录解析，支持 YAML/JSON。该文件与原始数据、manifest 一起冻结；修改它必须新建运行。旧 brief 不配置此字段时维持既有流程，但不会自动获得本轮矩阵保证。

以下示例使用数据接入说明中的 `user_table` 数据集：

```yaml
schema_version: 1
seeds: [42, 7, 123]
budget:
  per_cell_seconds: 20
  max_total_seconds: 240
  tuning_trials: 2
methods:
  - id: baseline
    role: baseline
    description: 固定深度的树基线
    parameters: {max_depth: 3}
  - id: proposed
    role: proposed
    description: 加入新组件的方法
    parameters: {component_enabled: true}
  - id: without_component
    role: ablation
    description: 关闭新组件
    parameters: {component_enabled: false}
    parent: proposed
    disabled_components: [new_component]
questions:
  - id: rq_main
    kind: main
    question: 新方法能否改善同一测试集上的表现？
    datasets: [user_table]
    methods: [baseline, proposed]
    baseline: baseline
    analysis: 比较匹配种子的结果，报告零提升和负结果，不将种子数当作独立样本数。
  - id: rq_ablation
    kind: ablation
    question: 新组件对结果的贡献是什么？
    datasets: [user_table]
    methods: [proposed, without_component]
    baseline: proposed
    analysis: 报告关闭组件相对完整方法的差值；效果为零不等同于消融实现错误。
```

示例形成 12 个必需单元：两个问题，各两个方法、三个种子。相同方法出现在不同问题中会得到不同 `regime`，本版本要求分别绑定执行记录，不自动把一个结果复用到多个问题。

每个问题必须声明比较基准，程序不按方法名称排序选基准。所有声明的方法及输入数据集必须被问题覆盖；不能通过“声明但不运行”跳过它们。主比较至少存在一项。消融必须指定完整方法作为 parent，并声明不同配置与关闭组件；这只是结构校验，仍需实现和执行轨迹验证组件确实关闭。

`parameters` 必须是有限 JSON 值。完整方法声明的哈希成为 EvidenceKey 的 `config`；方法、描述、参数、消融声明变化都会改变身份。

### 预声明复现容差

未配置时，两个独立正式矩阵只能用数值和单位精确相等获得复现匹配。若预期不同硬件或数学库会产生可接受的浮点差异，可以在协议中执行前声明逐指标规则：

```yaml
reproduction:
  metrics:
    accuracy:
      absolute_tolerance: 0.001
      relative_tolerance: 0
      rationale: Allow documented floating-point variation across accelerators.
```

规则必须恰好覆盖所有数据集声明的指标，不允许遗漏或增加未使用指标。绝对/相对容差须为非布尔有限数，分别限制在 `[0, 1000000]` 与 `[0, 1]`，且不能同时为零；每项必须给出不超过 1000 字符的非空理由。容差随原始 `spec` 进入协议版本哈希，正式执行后修改会改变 protocol version 并使既有回执失效。双运行比较使用对称阈值 `absolute_tolerance + relative_tolerance × max(|left|, |right|)`，仍要求完整 EvidenceKey、单位、输入、协议和实验源码身份一致。阈值只是预声明的复现判据，不说明该偏差在科学上一定可忽略；理由仍需领域审查。

## 分配与实际用量的区别

- 所有单元使用相同 `per_cell_seconds` 和 `tuning_trials` 配额。每个单元的时间配额包含其训练、验证调参和推理；调参次数不是额外乘数。
- 必需单元数乘以单元秒数不能超过 `max_total_seconds`，单元秒数不能超过 brief 的 `max_experiment_seconds`。不符合时预检失败，不自动裁剪实验。
- `max_total_seconds` 在本轮只校验协议分配，没有接入跨进程、重试、CodeAgent 试跑或 API 费用的全局账本。实际 sandbox 执行仍服从现有 `experiment.time_budget_sec`。
- 执行器、硬件、未披露的隐藏调参、失败重试和实际耗时的公平性仍需完整实验审核。覆盖报告不会自动将实验质量维度判为 passed。

## 输出与运行链路

根目录新增冻结的 `experiment_protocol.json`（编译产物 schema_version=2），包含：

- 原始结构化协议 `spec`、内容版本 `version`。
- 完整 `cells`、`required_keys`、固定 split IDs 哈希。
- 逐 seed 的显式基准/候选配对 `comparisons`。
- 单元数、预留秒数和 `actual_usage_status: unmeasured`。

Stage 9 将它写入实验计划；模型改写和 HITL 计划更新之后重新应用冻结约束。显式协议不受原有条件数量裁剪影响。上下文包含每个单元的完整身份，代码生成者需要据此实现并提供预测。

独立评估沿用 `trusted_evaluation.json` 接口。每条 run 的 `key` 必须原样采用 `cells[].key`，包括 `regime`、配置哈希和 seed。执行记录除 P0 要求的 success、returncode、code_commit、environment 外，必须增加：

```text
protocol_version = experiment_protocol.json 中的 version
cell_id          = 对应 cells[].cell_id
key              = 对应 cells[].key 完整对象
```

预测、测试标签与执行记录均通过独立评估器归档哈希。评估器拒绝未声明单元、把同一执行记录改名为其他方法/seed/RQ、以及来自其他协议版本的记录。这里检查执行记录的身份一致性，不提供数字签名或可信执行环境；执行记录必须由可信调用方采集，不能将模型自写的 success 作为运行真实性证明。自动生成宿主机逐单元执行记录仍是后续工作。

Stage 14 输出 `experiment_coverage.json`，包括缺失、无效、未声明结果，以及各 RQ/数据集/方法对的逐 seed 差值。差值为 candidate minus baseline，mse/mae 等越小越好的指标须按负向改善解释；单位不一致时拒绝比较。每个差值绑定双方结果 ID，缺任一必需种子时不计算其余种子的总体均值。本轮不从多个 seeds 自动推出样本或总体层面的显著性。

该报告同时进入 `experiment_summary.json` 和分析上下文。正式模式下矩阵不完整会失败；探索模式可以保留诊断，但最终验收仍检查缺项。最终打包携带协议和覆盖报告，验收重新从协议、EvidenceStore 与执行记录计算覆盖，不信任手写的 `status: complete`。

## 验证与后续工作

测试覆盖严格 schema、预算不足、配置与源文件变更、删除/缩减矩阵、多数据集分组、显式基准、漏 seed、错误 regime、执行记录归属、单位混用、零提升、Stage 9 超过原有裁剪上限、Stage 14 和最终验收集成。

2026-09-22 验证记录：包含 executor/runner 的较大范围回归 407 passed；补入配对差值与多数据集验证后，相关回归 153 passed。两组覆盖重叠，不应相加。`git diff --check` 和改动模块编译检查通过。

```powershell
& .venv/Scripts/python.exe -m pytest tests/test_experiment_protocol.py tests/test_research_inputs.py tests/test_final_acceptance.py tests/test_independent_evaluator.py tests/test_submission_gates.py tests/test_statistical_evidence.py tests/test_evidence_store.py -q --disable-warnings --maxfail=3
```

未调用付费模型、未完整生成论文，也没有执行真实科研矩阵。本轮测试的预测/执行记录是明确标记的 fixture，不构成真实实验结果。

第七十六轮补充执行前复现容差策略：实验协议与双运行比较联合回归 59 项通过。反例覆盖空指标映射、双零阈值、布尔伪装、NaN、相对容差越界、空理由和多余指标；真实双矩阵集成覆盖同源码、同协议身份下的有界数值差。测试仍是工程 fixture，不构成跨硬件科研复现实验。

后续需要实现宿主机逐单元调度和预算账本、验证集选择与最终测试解锁、失败尝试不可丢弃的执行记录、资源与环境公平性审核、消融实现轨迹、外部数据适配以及 MethodSpec/TheoryBundle。
