# 高质量论文可靠性底座：P0 第一轮实现

依据：`HIGH_QUALITY_PAPER_GAP_ANALYSIS_CN.md`。本轮实现确定性的反例拦截、证据接口与正式交付门禁，保留原来的 23 阶段。它不是 P1/P2 全部完成，也不代表普通模型已通过真实科研端到端评测。

## 1. 已落实的行为

| 对应问题 | 实现 | 边界 |
|---|---|---|
| P0-A 降级冒充完成 | 独立的 `artifact_status`、`target_met` 和分维度状态；正式模式低分失败、预算耗尽暂停、保留 checkpoint；失败阶段不允许 `skip_noncritical` 绕过 | 默认仍允许探索运行；探索产物不自动获得正式状态 |
| P0-B 数值归属 | `EvidenceStore` 完整实验键、内容 ID、原始文件哈希、单位、执行状态；按 ID 渲染数字/审计表；旧 verifier 补充明确方法归属检测 | 自由文本的完整语义归属仍需逐项审核，旧全局白名单不再被视为正式证明 |
| P0-B 独立复算 | 主机侧 CSV 评估器计算 accuracy、binary AUROC、MSE、MAE，不导入生成的实验代码；Stage 14 接入可选冻结清单 | 不是全部任务指标的通用评估器；数据来源、标签冻结和测试集访问隔离仍需数据契约 |
| P0-C 统计 | 保留 method/stratum/seed/metric；显式基线；逐 stratum 配对；bootstrap 区间、Holm 校正；常量差值显式标记 | 当前协议是训练种子层面的 paired t / percentile bootstrap；不能解释为受试者或跨数据集独立重复 |
| P0-C 负结果 | 数值接近或相同只作为观察；修复器必须有组件移除的失败执行检查才诊断消融故障 | 组件执行检查仍需要实验实现提供，不能从结果反推 |
| P0-D 引用 | 无引用或全部跳过时分数为 null；区分 missing/unavailable/verified/contradicted；不复活已知虚构引用；取消全局 60 条上限 | 元数据 verified 不代表支持论断；正式验收另需全文片段和支持审核 |
| P0-E 最终版 | 完成打包和扩展钩子后重新验收；绑定 MD/TeX/PDF/Bib/图表/证据/审核哈希；编译、缺图、断引用、占位符、过期审核均阻止正式候选稿 | 尚未重构统一 ManuscriptIR，也未自动逐页视觉检查 PDF；缺少这些审核时状态保持 unknown |
| 空新颖性候选 | `novelty_score=null`、`assessment=unknown`；记录 search_status，建议继续核查 | 没有找到近似论文不能推断创新已成立 |

## 2. 配置

```yaml
research:
  target_status: submission_candidate  # exploratory / research_complete / submission_candidate
  quality_threshold: 7.0
  graceful_degradation: true          # 仅 exploratory 允许降级继续
experiment:
  comparison_baseline: Baseline       # 必须是输出中的真实方法 ID；空值不自动挑基线
export:
  max_citations: null                 # 按具体版式需要设置正整数
```

默认 `target_status=exploratory` 保持已有工作流可运行。请求正式产物后，未满足目标会在 `pipeline_summary.json` 中写入 `target_met=false`，并返回失败结果；诊断文件仍然保存。

`research_complete` 要求 data、experiments、numeric、citations、theory 全部通过，实证研究的 theory 可经明确审核标为 not_applicable。`submission_candidate` 还要求 quality、consistency、figures、layout 通过。未知项不能用另一维度的高分抵消。

## 3. 可选的独立数值评估

在 run 根目录准备 `trusted_evaluation.json`，标签及其哈希应在模型选择前冻结，不应交由生成实验的模型重新定义。被引用文件统一放在 `evidence_artifacts/` 下，打包时保留相对路径。

```json
{
  "schema_version": 1,
  "runs": [{
    "key": {
      "dataset": "example",
      "dataset_version": "v1",
      "split": "test",
      "method": "Baseline",
      "config": "configuration-sha256",
      "seed": "42",
      "metric": "accuracy",
      "aggregation": "per_run",
      "regime": "default"
    },
    "labels": "evidence_artifacts/labels.csv",
    "predictions": "evidence_artifacts/baseline_predictions.csv",
    "expected_labels_sha256": "填写冻结标签文件的真实SHA256",
    "execution": "evidence_artifacts/execution.json"
  }]
}
```

标签 CSV 为 `id,label`，预测 CSV 为 `id,prediction`，ID 必须唯一且集合完全相同。accuracy 接受离散数值类别；AUROC 接受二分类标签和连续得分，并处理并列得分。执行记录必须包含 `status: success`、整数 `returncode: 0`、`code_commit`、`environment`。标签变化、缺样本、重复 ID、非有限值、失败运行均不能生成成功证据。

Stage 14 生成 run 根目录的 `evidence_store.json`，以及 `stage-14/independent_evaluation.json`、`evidence_results.md`、`evidence_results.tex`。后两个文件是可核查的完整身份审计表，不是已完成版面排版的论文表格。Stage 17 会重新验证 store 和原始文件，再把确定性表格提供给写作上下文。

每条 `EvidenceRecord` 的 key 包含 dataset/version/split/method/config/seed/metric/aggregation/regime，记录包括单位、成功状态、代码版本、环境、执行记录、评估器标识及文件哈希。ID 与 store version 从内容计算；同一完整实验键不能静默覆盖成不同结果。`derive()` 提供带来源 ID 的差值、百分点差值和相对变化，禁止跨版本、split、metric 混算。

## 4. 最终验收资料

最终打包器读取 run 根目录中的下列附加资料。缺资料会生成可定位的缺项，不会自动创建虚假的通过审核。

| 文件 | 必需内容 |
|---|---|
| `experiment_protocol.json` | `required_keys`：预声明的完整实验键列表。每个必需键都必须有成功且可追溯的结果 |
| `numeric_claims.json` | `evidence_version`、`manuscript_hashes`、`claims`。每项 claim 带 result_id、完整 key、value、unit、decimals、rendered，以及 MD 和 TeX 中数值的字符跨度 `spans` |
| `citation_support.json` | `manuscript_hashes`；`citations` 中逐项记录 cite_key、status、checker、claim、source、source_sha256、locator、excerpt。source 使用已保存的全文文本摘录，excerpt 必须确实存在于文件中 |
| `theory_bundle.json` | `obligations`。关键证明只能为 machine_checked 或 reviewed_informal，且有 checker/evidence；unresolved 和纯数值检查不等价于证明 |
| `final_reviews.json` | `input_version` 与分维度 `dimensions`，每项含 status、checker、evidence；包括全文数值覆盖、引用支持、方法/实现一致性、图像语义和 PDF 全页布局审核 |

`manuscript_hashes` 是 `paper.tex` 与 `paper_final.md` 的 SHA-256 字典。`spans` 使用 Python 字符串的零基、右侧不包含索引；每种格式均须定位到准确的 `rendered` 数字。

引用删除后还需 `citation_support.json.removed_claim_resolutions`：以已删除 cite key 为键，提供 `status=claim_removed` 或 `rewritten_with_evidence`、checker、evidence。只删引用标记而未核查论断不能通过。

审核者/检查器必须真正检查相应证据后生成 final_reviews，不能把所有维度机械填为 passed。当前接口不提供密码学签名或抵御拥有整个工作目录写权限的恶意操作者；哈希用于发现版本变更与过期审核，不证明审核者身份或研究结论正确。

验收结果写入 `deliverables/final_acceptance.json`，每个 issue 带 dimension、artifact、repair_owner。`manifest.json` 绑定交付文件与审核哈希。`compilation.json` 保存本次编译状态、输入哈希、PDF 哈希、警告和修复记录；存在旧 PDF 不算编译成功。编译成功也不能替代布局/图像语义审核。

完成全部内容修订和编译后取得审核输入版本：

```powershell
.venv/Scripts/python.exe -m researchclaw.pipeline.final_acceptance artifacts/<run>/deliverables --print-input-version
```

在最终审核完成后重新验收，成功退出码为 0，目标未达到为 2：

```powershell
.venv/Scripts/python.exe -m researchclaw.pipeline.final_acceptance artifacts/<run>/deliverables --target-status submission_candidate
.venv/Scripts/python.exe -m researchclaw.pipeline.final_acceptance artifacts/<run>/deliverables --check-seal
```

后续改写任一交付文件会使 seal 失效。修改 TeX、Bib、图像后需要重新编译和重新审核；该验收命令不会自动重新编译。MD/TeX 的核心内容是否语义一致、所有数字是否被 claim 清单覆盖，由绑定完整文件版本的审核负责，不能仅凭数值字符串相同宣称完成一致性证明。

## 5. 回归验证与后续范围

新增测试覆盖方法/数据集/版本/split/metric/seed/config/regime 错配、百分比与百分点、失败运行、原始文件篡改、常量差值、跨 regime 复用 seed、显式基线、独立指标复算、空引用、虚构引用复活、无全局引用上限、预算耗尽 checkpoint、关键实验缺失、未决证明、缺图、占位符以及审核后文件变化。

验证记录：23 个相关测试文件的较大范围回归为 **853 passed、15 skipped**；随后新增表格渲染、删除引用后的论断处理、伪造全文片段、畸形审核、agent 重试耗尽和旧 store 失效测试后，13 个相关文件为 **288 passed、9 skipped**。两组有重叠，不应相加。测试使用本地 `.venv/Scripts/python.exe`，未调用付费模型，也未进行真实论文的端到端生成与视觉验收。

本轮未完成的文档路线包括：通用 ResearchBrief/DatasetManifest 与测试集权限隔离、完整文献覆盖矩阵和自动句级支持判断、通用 Theory Workbench/MethodSpec、统一 ManuscriptIR、图像语义自动验收、用户模板导入、小节级写作 DAG，以及固定普通模型预算的真实 ARC-Bench 端到端评测。现有确定性检查和单元/集成测试不能替代这些工作。
