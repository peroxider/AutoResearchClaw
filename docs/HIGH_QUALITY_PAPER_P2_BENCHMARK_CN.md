# P2 固定预算普通模型基准套件

现有 `benchmark_harness.py` 用七种故障注入测量最终验收门的错误放行；它不等于普通模型端到端基准。`benchmark_suite.py` 为后者增加公开任务覆盖、统一预算和私有金标准的冻结载体，但不会自行调用模型或生成真实分数。

## 公开计划

公开根目录内的 `benchmark_plan.json` 使用 schema v1：

```json
{
  "schema_version": 1,
  "min_repeats": 2,
  "budget": {
    "wall_seconds": 300,
    "model_calls": 20,
    "total_tokens": 100000
  },
  "cases": [
    {
      "case_id": "classification-leak-00",
      "task_family": "tabular_classification",
      "scenario": "leakage_trap",
      "repeat_index": 0,
      "input_bundle": "inputs/classification-leak-00.json",
      "input_sha256": "<64 lowercase hex characters>"
    }
  ]
}
```

四个必需任务族为 `tabular_classification`、`tabular_regression`、`temporal` 和 `image`。八个必需压力场景为 `leakage_trap`、`invalid_idea`、`null_or_negative_result`、`external_download_failure`、`incorrect_proof`、`image_api_failure`、`missing_template` 和 `long_manuscript`。每个任务族和每个场景都必须至少出现 `min_repeats` 次；该值限制在 2–20，总用例不超过 512。`(task_family, scenario, repeat_index)` 不得重复。

所有公开输入必须是公开根目录内、非符号链接、最大 10 MB 的现有文件，并与声明 SHA-256 一致。全部用例共享同一预算；墙钟、模型调用和 token 上限均为正整数并有硬上界。冻结器不根据运行结果删除困难用例或改变预算。

## 私有 gold

私有 gold 必须位于公开根目录之外，并恰好覆盖全部公开 case ID：

```json
{
  "schema_version": 1,
  "cases": [
    {
      "case_id": "classification-leak-00",
      "expected_disposition": "reject",
      "rubric": "Private evaluator rubric, up to 2000 characters."
    }
  ]
}
```

`expected_disposition` 只能是 `accept`、`reject` 或 `honest_negative`。公开冻结产物只保存 gold 文件 SHA-256，不复制路径、rubric 或期望结局。文件系统分离只是布局契约；真实盲评仍应在模型无权读取的账户、容器或远程评估服务中保存 gold。

## 冻结与复验

```powershell
& .venv/Scripts/python.exe -m researchclaw.pipeline.benchmark_suite <public-root> --plan <public-root>/benchmark_plan.json --private-gold <private>/gold.json --output <public-root>/benchmark_suite.json
```

报告冻结规范化公开计划、原始 plan/gold 摘要、逐类别计数、scope、限制和内容版本。输出不得覆盖 plan 或任一公开输入。`verify_benchmark_suite` 从当前公开输入和私有 gold 完整重建报告，自洽重哈希后的字段修改仍会被拒绝。

## 结论边界

`status: ready` 只表示用例覆盖、重复次数、预算、公开输入和私有 gold 形态满足契约。它不表示已经运行普通模型，不提供端到端成功率、错误接受率、成本或人工修订量，也不证明 gold 没有通过其他渠道泄漏。后续真实服务 runner 必须按 case 执行、保存实际模型/端点/调用账本和输出摘要，再由独立评估器读取私有 gold；有限基准上的结果不能外推为任意论文质量保证。
