# P1 AnalysisSpec 与配对效应图

本轮将“分析什么、使用哪些结果、计算什么、画什么、能解释到哪里”保存为 `analysis_spec.json`，并接入冻结协议、Stage 14、出版资产、ManuscriptIR 和最终验收。它核验计算和来源，不认证科研假设或模型判断。

## 冻结方案

在输入协议的 `questions` 项中可以声明 `analysis_plan`，与方法、数据集、种子和预算一同冻结。例如：

```yaml
analysis_plan:
  schema_version: 1
  summary: paired_seed_difference
  figures:
    - paired_seed
    - effect_summary
    - calibration        # 可选：逐训练种子 ECE，见下文
    - efficiency_pareto  # 可选：逐训练种子墙钟秒，须同时声明 metric_direction
    - learning_curve     # 可选：执行进程逐步遥测，须同时声明 learning_curve
  learning_curve:
    metric: validation_loss
    split: validation
    direction: minimize
    max_points: 200
  interval:
    method: paired_seed_percentile_bootstrap
    confidence: 0.95
    replicates: 2000
    random_seed: 42
    exchangeable_training_seeds: true
```

不声明时使用描述性默认方案：汇总配对种子差值、生成 `paired_seed` 图，`interval.method=none`。默认方案不会加入显著性检验，也不把普通文字 `analysis` 解释成已经预注册的统计方法。显式方案必须完整且字段受限，未知图表/检验、空值、重复图类型和无效参数在冻结时拒绝。

可选重采样要求置信水平在 0.8–0.99、次数为 1000–20000、随机种子为无符号 32 位整数，并显式声明训练种子的可交换性。全协议最多预留 5,000,000 次样本抽取，冻结前和计算前均检查；超预算要求修改研究方案，不删种子或只保留有利比较。这里记录的是计算预留量，不冒充实测资源消耗。

第三十一轮增加两个按测试样本结构预声明的区间方法：

- `cluster_percentile_bootstrap` 要求每个问题数据集声明 `group_column`，区间增加 `resampling_unit: group`。每次有放回抽取与原组数相同的完整实体簇，簇内样本始终一起出现。
- `moving_block_percentile_bootstrap` 要求 `time_column`，区间增加 `resampling_unit: time_point`、`block_length`（2–1000）与 `chronological_order_preserved: true`。实现从冻结的连续时间点序列随机选择起点并循环取完整块，直到获得原时间点数；同一时间戳的全部样本始终一起出现。

两种方法都在每个 replicate 内用重采样样本重新计算每个训练 seed 的基线/候选 accuracy、MSE、MAE 或 AUROC，再取 seed 间平均配对差值；不会把训练 seed 当患者、实体或时间点。计划冻结时核对数据结构、块长和按“测试样本 × seed × replicate”计算的 5,000,000 次抽取预算，计算时再次核对。少于三个结构单位、单位与预测 ID 不完全一致、AUROC replicate 缺一类或时间序不连续时失败关闭，不静默删 replicate。区间仍是冻结测试样本结构下的条件描述，`assumption_verified=false`、`simultaneous_coverage=false`，不认证抽样设计代表目标总体。

原始方案变更不能静默改变已经冻结的运行。交付包继续依赖冻结快照；尝试使用改过的原始输入恢复同一运行时，会被输入契约拒绝。

## 结果绑定与计算

每个比较保存：RQ、数据集及版本、split、metric、aggregation、regime、单位、基线和候选方法；每个配对种子的两个结果 ID、观测值及候选减基线差值；全部源结果 ID、协议与证据版本、输入摘要、计算器源码摘要及 Python 版本；方案来源、均值、差值标准差、范围、正/零/负差值数、区间状态和解释范围。

先重新审计整个必需实验矩阵和原始产物，再计算。缺失、失败、额外结果或原始产物变化都不能退化成“可用子集”的分析。非有限输入、算术溢出或非有限派生值会失败。不同数据集、版本或 RQ 不合并，零提升和负结果正常保留。

区间在配对差值上重采样，相当于始终共同选择同一个种子的基线与候选观测，不分别打乱两组。使用固定随机种子、样本均值的百分位区间和线性分位点插值；方法定义可参见 [SciPy bootstrap 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) 中的 paired 与 percentile 说明。当前实现使用 Python 标准库，不依赖 SciPy 的默认 BCa 算法。

少于三个种子返回 `unavailable_insufficient_seeds`；差值恒定返回 `unavailable_constant_differences`，不会把退化区间包装成有把握的结论。可计算时为 `computed_conditional`，同时保留 `assumption_verified=false`、`simultaneous_coverage=false`。三个种子只是本实现的最低计算门槛，不表示样本充分或覆盖率已经验证。

训练种子刻画同一冻结测试切分下的训练随机性，不是独立受试者或数据集。这里不产生总体推广、因果、显著性、多重比较联合覆盖率或方法优越性的认证。对于误差等越小越好的指标，正的候选减基线差值可能更差；图中不会自动把正差值称为提升。

## 图表与写作

`paired_seed` 保留逐种子连线与差值；`effect_summary` 在同一条件下汇总预声明比较，显示原始配对差值、观测均值及可用的条件区间。顺序来自协议，不按结果排序或只显示最佳方法。

第二十一轮新增两种可选图类型。`calibration`：对 `trusted_evaluation.json` 绑定的冻结逐样本预测计算逐训练种子期望校准误差（ECE，等宽 10 桶、首桶含下缘、空桶不计）；预测与标签文件必须出现在记录工件表中且文件哈希一致，分数须在 [0,1]、标签须恰为 0/1，否则失败关闭。图数据（逐种子 ECE、差值、样本数、桶数与范围声明）冻结进分析报告，重算校验覆盖。它描述该次冻结测试切分上的观测失准，不作总体校准或可靠性声明。`efficiency_pareto`：逐训练种子墙钟秒取自 `protocol_budget.json` 的 `per_cell_seconds`，要求 status=complete、协议版本一致、每个分析单元格的秒数为有限非负数；请求该图必须同时声明 `metric_direction`，且其他图类型不得携带该字段。它是单宿主机冻结账本的观测记录，不作跨硬件、成本或方法排名声明。

第三十轮新增 `learning_curve`。方案必须同时声明指标安全标识、`train`/`validation` split、优化方向和 2–2000 的最大点数。只有请求该图的问题才向每个协议单元传入固定 `protocol_learning_curve.csv` 输出契约；CSV 必须恰为 `step,value`，步号是规范非负整数、严格递增且基线/候选在同一种子上完全对齐，值必须有限。宿主机将文件复制到该次 attempt、限制为 1 MB，并把路径和 SHA-256 写进执行收据；账本复核、AnalysisSpec 重算与最终资产验证都会重新核对。图保留每条种子轨迹并叠加观测种子均值，不按最好 seed 或局部点裁剪。该指标由冻结实验进程报告，宿主机只验证结构、完整性和身份，不能独立证明它确为所声明的训练/验证指标；它不是最终测试表现，也不是收敛证明。

效应图按比较数量和完整方法名的换行量分图，每图最多六项比较，后续比较继续生成。图例、横轴说明与范围声明分别留出空间；缩放同时约束宽度和高度。没有区间时明确显示描述性或不可用状态。所有图有完整源 ID 的 CSV、确定性 PNG/PDF、渲染器记录和可保存的复现脚本。

Stage 14 的冻结协议路径直接生成分析契约和声明的图表，产出 `analysis.md`、`experiment_summary.json`、阶段内 `analysis_spec.json`；不再从自由指标摘要额外运行未声明的 t 检验或 FigureAgent 图表选择。研究解释留给之后的证据包写作与评审。无冻结协议的旧探索路径保持兼容。

ManuscriptIR 的每个匹配 RQ 结果任务获得对应分析契约，Discussion 等概括章节也可访问它。模型仍不得直接重复实验数字；正式数值表和图表由工具生成。分析变化会使依赖它的写作评审和导出失效。

打包复制分析契约到私有审计目录；最小投稿 ZIP 仍只包含实际编译所需材料。最终验收重新计算并逐字段比较 AnalysisSpec，即使人为重新计算外层 JSON 摘要也不能改统计量、计划、结果 ID、单位或推断范围。错误归属 `numeric` 维度，修复方为 `analysis`。

## 验证与当前边界

测试包含有已知百分位端点的离散样本、成对平移不改变区间、参数与预算拒绝、样本不足/恒定差值、零/负结果、源 ID 覆盖、源数据变化、重摘要篡改、缺契约的最终验收、阶段集成、长标签分图，以及实际 PNG/PDF 和双引擎 TeX 编译。

本轮较大范围回归 278 passed；最后完成标签边界和图例布局修复后的相关回归 129 passed，含 6 组真实 TeX 用例，集合重叠不累加。六份最终样稿共 42 页已 Poppler 渲染检查；效应图样稿最小提取字号约 7.96 磅，编译日志无横向溢出。比较标签使用实际字体边界分配横向空间与行距，并拒绝重叠/越界；这些检查仍不是任意复杂稿件的完整视觉审核。

第二十一轮校准/效率图集成后，分析契约套件 56 passed；资产、稿件、协议、执行、独立评估消费方回归 109 passed，账本/验收/工作台/方法验证/语义回归 186 passed，端到端与真实编译 8 passed, 6 skipped（集合重叠，不累加）。fixture 用真实宿主机 `run_matrix` 产出完整执行证据（源码清单、账本、预算、trusted_evaluation），常数 0.5 基线分数在均衡切分上 ECE=0，倒置过置信候选分数 ECE≈0.99，按精确值断言；PNG/PDF 确定性重渲染与 CSV 字段有测试覆盖。实现中发现并修复一处 fail-open：trusted_evaluation 缺少绑定分析记录的运行时曾以 AttributeError 崩溃，现失败关闭为 AnalysisError。

分组实体簇和循环移动时间块已有从冻结预测/标签重新计算指标的条件 percentile bootstrap；尚不支持层级混合模型、不规则时间间隔权重、生存/删失、空间相关、敏感性参数轴或联合多重比较区间。不得把当前结构化区间外推到这些设计。学习曲线已有预声明与哈希绑定的逐步遥测，但宿主机尚不能独立复算任意用户定义的训练指标。后续仍需根据真实研究设计验证抽样假设并进行科研基准评估。工程 fixture 和有限反例测试不能证明任意论文的统计有效性。
