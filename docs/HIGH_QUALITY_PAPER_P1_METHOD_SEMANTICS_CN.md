# 方法语义审核（structured semantic review）

MethodSpec 的静态映射只证明"名字存在"，预声明执行检查只证明"实例数值符合预期"；两者都不能宣称代码与规范语义等价。本轮增加一个有边界的人工审核通道：由具名审核者检查可重算的证据包，并给出分步结构化结论。它把**报告状态**从 `unresolved` 最多升级到 `reviewed_informal`，永远不是机器证明；`method_spec.json` 内冻结的 `semantic_equivalence` 不被改写。

## 证据包与绑定

`python -m researchclaw.pipeline.method_semantics <run目录> --write-packet` 从冻结产物重算并写出 `method_semantic_review_packet.json`：

- 冻结 MethodSpec 全文与 `method_version`；
- 每个已映射步骤的归档源码片段（按 AST 行号从 `evidence_artifacts/protocol_source/` 提取）、文件哈希与符号行号；
- 映射检查的问题清单与 `method_validation.json` 的摘要（含报告哈希，如存在）；
- 代码清单哈希 `code_sha256` 与 `packet_version`。

证据包只能由冻结产物重算得到；归档源码被改动、代码清单哈希不符或包文件被改写都会直接失败。审核者拿到的是只读证据，不是结论。

## 审核文档契约

审核者写出 `method_semantic_review.json`：

```json
{
  "schema_version": 1,
  "method_version": "<method_spec.json 的 version>",
  "code_sha256": "<protocol_code.json 的 code_sha256>",
  "packet_version": "<证据包哈希>",
  "reviewer": {"checker": "具名审核者标识", "evidence": "审核背景与独立性说明"},
  "steps": [{"step": "predict", "verdict": "consistent",
             "spec_evidence": "规范中的逐字引用",
             "code_evidence": "归档源码中的逐字引用", "notes": ""}],
  "equations": [{"equation": "projection", "verdict": "consistent",
                 "spec_evidence": "Y = X W", "notes": ""}],
  "conclusion": {"verdict": "consistent", "evidence": "结论依据"}
}
```

硬性规则：

- 绑定：`method_version`/`code_sha256`/`packet_version` 必须与重算证据包一致，规范重编译必须逐字节还原；否则审核文档被视为过期。
- 逐字引用：`spec_evidence` 必须在冻结规范中出现，`code_evidence` 必须为非空并出现在该步骤归档源码片段中；凭空描述与空字符串都不被接受。任何没有归档源码的步骤（映射失败或源码提取失败）使整份审核无法成立，必须先修复代码映射。
- 全覆盖：每个步骤和每条公式各有一条结论，词汇为 `consistent`/`inconsistent`/`unverifiable`；非 consistent 结论必须附 notes。
- 结论重算：总结论必须等于从全部记录结论推导的最坏结果（inconsistent > unverifiable > consistent），不接受审核者自行宣布的总结论。
- 版本防篡改：文档 `version` 绑定全部字段；持久化形式允许携带派生字段 `semantic_equivalence`/`version`，但核验时会重算并比对，改写后重新签名无效。

## 状态语义

| 记录结论 | 报告状态 |
|---|---|
| 全部 consistent | `reviewed_informal` |
| 存在 unverifiable | `unresolved` |
| 存在 inconsistent | `contradicted` |

`reviewed_informal` 表示"一位具名审核者核对证据后认为一致"，不认证审核者独立性，也不是机器证明；`contradicted` 表示审核者发现代码与规范矛盾，会在最终验收中产生 `method_semantic_review_contradicted` 归属 experiments 的问题。审核文件无效、过期或被篡改时，最终验收产生 `invalid_or_stale_method_semantic_review`，一律失败关闭。

## 集成点

- `reported_equivalence`：对外展示状态；证据缺失或核验失败时回退为冻结的 `unresolved`，不猜测。
- `manuscript.py`：写作证据包加入 `method_semantics` 条目（scope 明确 "never a machine proof"），论文算法块的语义状态与审核者标识随之更新。
- `final_acceptance.py`：交付时重算证据包并重查全部结论绑定，而非信任状态字符串。
- `method_semantic_review_packet.json` 进入导出依赖清单。

## 范围与验证记录

真实子进程测试使用小型合成函数；审核文档由 fixture 审核（"fixture independent reviewer"）给出，不经过真人。没有接入 LLM 审核者、多人交叉审核或审核者资质认证；这些都不能由数值样例冒充。

定向回归 9 项（`tests/test_method_semantics.py`）：接受审核、包与源码绑定、逐字引用反例（含空引用与无归档源码步骤失败关闭）、结论推导与覆盖、过期/篡改回退、CLI、最终验收三种结果、证据包条目。同轮工作台/验证/验收/写作等较大范围回归见[开发状态](HIGH_QUALITY_PAPER_DEVELOPMENT_STATUS_CN.md)。

```powershell
& .venv/Scripts/python.exe -m pytest tests/test_method_semantics.py tests/test_research_workbench.py tests/test_method_validation.py tests/test_final_acceptance.py tests/test_manuscript.py -q --disable-warnings
```
