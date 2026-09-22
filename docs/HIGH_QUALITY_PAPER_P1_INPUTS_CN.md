# P1 第一轮：结构化研究输入与本地表格数据接入

在 P0 的质量门禁和 EvidenceStore 基础上，本轮打通了 ResearchBrief → DatasetManifest → 数据预检 → 固定切分 → 沙箱数据接入 → 独立评估身份检查 → 最终交付复核。

当前支持本地 CSV 的分类与回归任务，执行后端为 `sandbox` 和 `docker`。没有配置 brief 的既有流程保持兼容。图像、时序模型专用数据加载、外部数据库下载适配器、操作系统级测试集权限隔离不在本轮范围内。

## 配置与输入文件

在现有运行配置中增加：

```yaml
research:
  topic: 比较两种分类方法的泛化效果
  brief_path: study/brief.yaml
  target_status: submission_candidate
experiment:
  mode: sandbox
  metric_key: accuracy
  metric_direction: maximize
  time_budget_sec: 300
```

`research.brief_path` 相对项目根目录解析；brief 内的 manifest 路径相对 brief 文件目录；manifest 内 CSV 路径相对 manifest 文件目录。Stage 1 首次创建输入快照；从后续阶段恢复前必须已有该快照。

`study/brief.yaml` 示例：

```yaml
schema_version: 1
question: 新方法在相同数据和计算预算下能否优于基线？
ideas:
  - 比较基线与新方法，并报告零提升或负结果
constraints:
  - 不添加外部训练样本
  - 只使用预先声明的特征
hypotheses:
  - 新方法可能降低泛化误差
paper_type: empirical
allow_external_data: false
max_experiment_seconds: 300
datasets:
  - dataset.yaml
```

`study/dataset.yaml` 示例：

```yaml
schema_version: 1
dataset: user_table
version: v1
path: records.csv
task: classification
id_column: row_id
label_column: outcome
features:
  - age
  - measurement
forbidden_features:
  - post_outcome_note
metric: accuracy
source: 用户提供的研究数据
license: 仅限本地研究
split:
  strategy: stratified
  seed: 42
  train_fraction: 0.6
  validation_fraction: 0.2
```

`features` 为显式白名单，ID、标签、分组/时间列及 forbidden_features 不能被混入。source/license 是用户声明的来源与使用条件，程序记录但不自动验证授权。manifest 中声明的列必须存在；未知字段会报错，避免拼错约束后被静默忽略。

分类指标支持 accuracy 和二分类 auroc，方向为 maximize；回归指标支持 mse、mae，方向为 minimize。回归应选择 random/group/time 切分。`primary_metric` 可作为配置层的别名，但 DatasetManifest 中仍需真实指标名。同一次运行的指标方向必须一致。

`max_experiment_seconds` 是单次实验执行时限上界，不是整个研究任务的总时长或全部 API 费用预算。初始化后还会冻结 topic、metric_key、metric_direction、time_budget_sec；修改这些值需要新运行，不能无声改变已冻结的研究协议。

## 四种切分

| strategy | 规则 |
|---|---|
| random | 依据 seed 与稳定行 ID 的哈希排序，划分 train/validation/test；原始行顺序不决定分区 |
| stratified | 分类任务按标签分层，每层至少三条记录，保证三个分区非空 |
| group | 必须指定 group_column，同一实体始终位于同一分区；至少三个独立组 |
| time | 必须指定 time_column，按 ISO-8601 时间排序；相同时间戳不拆开，至少三个独立时间点 |

test_fraction 为剩余比例。小样本、按组或时间切分时，实际行数比例可能偏离配置比例；实际大小写入 dataset card。时间戳统一转成 UTC，无时区时间按 UTC 解释。

声明了 group_column 就不能使用普通随机/分层切分。声明了 time_column 必须使用时间切分；同时声明 group/time 时，如果实体跨越时间分区，预检失败，需要另行设计协议，而不是自动允许交叉泄漏。

重复行 ID、缺标签、CSV 行列错位、回归非有限标签等会失败。训练集中发现直接复制目标的特征也会失败。不同 ID 但特征/标签相同可能是有效独立观测，因此只记录重复数量并要求来源审核，不据此自动删除样本或判定实现错误。

## 输出与模型上下文

```text
<run>/
  research_contract.json
  data_preflight.json
  research_inputs/<dataset>/
    train.csv
    validation.csv
    test_features.csv
    split_ids.json
    dataset_card.json
  evidence_artifacts/<dataset>/
    test_labels.csv
```

`research_contract.json` 记录规范化 brief、manifest、原始文件哈希、输出哈希、切分身份以及内容版本。重复初始化读取同一快照，不重新抽样。

dataset card 提供训练集的缺失数、不同值数量、数值范围/均值和标签统计。它不提供测试集标签分布。分类标签按训练集类别生成稳定 `label_encoding`，测试集不能出现训练集未知类别；AUROC 要求每个分区都有两个类别。

代码与写作上下文得到声明的约束、训练统计、标签编码和项目相对路径。测试标签路径、原始完整 CSV 路径不会由这个上下文接口注入。训练/验证 CSV 保留原标签，生成分类预测时必须按 label_encoding 转成数值类别；AUROC 输出正类得分。

每次沙箱 `run_project()` 会把训练集、验证集和测试特征复制到：

```text
<experiment-project>/research_data/<dataset>/
  train.csv
  validation.csv
  test_features.csv
```

因此 CodeAgent 试跑、首次实验、迭代修复和自动 repair 都使用相同相对路径。该目录是保留命名空间：出现未知文件或符号链接会报错。测试标签不复制到实验项目；调用者仍需保证 sandbox/docker 的宿主目录挂载、网络和权限符合研究隔离要求。本实现不宣称阻止具有宿主机读权限的进程主动搜索测试标签。

显式 manifest 的数据集优先于 BenchmarkAgent 自动选数；Stage 9 固定 datasets 与 dataset_metrics，Stage 10 不注入旧 benchmark_plan 来覆盖用户数据。`allow_external_data` 作为不可漂移的声明传入模型；即使它为 true，当前接入层仍要求所有实际评估数据集显式声明为 manifest，并不自动下载额外数据。

## 运行恢复和最终验收

runner 在阶段执行前后复核原始文件和准备好的切分文件。任何缺失、改写、移除 brief 配置或运行约束变化都会产生 `input_contract_invalid`，并写入失败的 data_preflight；探索模式和 skip_noncritical 也不能绕过此错误。

P0 的独立评估器现在会在存在 research_contract 时校验 DatasetManifest 的 dataset/version/metric，以及固定 test split 的标签文件路径和哈希，阻止把验证集、其他数据版本或自建标签冒充正式测试结果。`trusted_evaluation.json` 的 predictions/execution 记录仍按 P0 接口提供；本轮不会把训练代码自报的指标自动标成可信证据。

最终打包包含输入快照、dataset card、split IDs 和准备好的分区文件。最终验收重新验证这些文件和 EvidenceStore 的数据身份绑定，不依赖原始源文件仍在原机器上。预检通过只证明这些结构性检查通过，`data` 质量维度仍需要针对采样、泄漏风险、数据适用性等的完整审核。

本地文件接口不会自动上传数据。最终交付包包含数据分区和测试标签，分享交付包前应按数据实际使用条件选择可分享的文件；隐私数据的脱敏/受控分发不是这里的自动预检功能。

## 验证范围

新增测试覆盖四种切分、组/时间边界、未知类别、指标与预算约束、输入/输出篡改、路径解析、数据恢复、评估身份绑定、最终验收以及真实 Python 子进程对准备数据的读取。Docker 使用既有 mock 回归，未启动真实容器；没有调用付费模型或完整生成论文。

2026-09-22 相关回归共 **515 passed**（82.56 秒），运行命令如下；`git diff --check` 也通过。

```powershell
& .venv/Scripts/python.exe -m pytest tests/test_research_inputs.py tests/test_rc_executor.py tests/test_rc_runner.py tests/test_rc_config.py tests/test_rc_docker_sandbox.py tests/test_universal_codegen_integration.py tests/test_benchmark_agent.py tests/test_experiment_repair.py tests/test_final_acceptance.py tests/test_independent_evaluator.py tests/test_submission_gates.py -q --disable-warnings --maxfail=6
```

后续仍需继续实现通用数据下载适配器、公平实验矩阵与预算核算、MethodSpec/TheoryBundle、引用句支持工作流、ManuscriptIR 和固定模型端到端评测。

第二轮已增加可选 `protocol_path`，用于预声明实验矩阵、等额预算分配和结果覆盖检查；见 [P1 实验协议说明](HIGH_QUALITY_PAPER_P1_PROTOCOL_CN.md)。实际资源账本与自动调度仍待实现。
