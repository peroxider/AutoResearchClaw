# P2 固定预算普通模型基准套件

现有 `benchmark_harness.py` 用七种故障注入测量最终验收门的错误放行；它不等于普通模型端到端基准。`benchmark_suite.py` 为后者增加公开任务覆盖、统一预算和私有金标准的冻结载体；`benchmark_runner.py` 按冻结适配器逐 case 执行公开输入；真实分数仍只由独立评估器结合私有 gold 与盲评记录产生。

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
  "runner": {
    "adapter": "runner_adapter.py",
    "adapter_sha256": "<64 lowercase hex characters>",
    "environment_allowlist": ["OPENAI_API_KEY"]
  },
  "attestation": {
    "curator": {
      "algorithm": "ed25519",
      "key_id": "curator-1",
      "public_key": "<base64 raw Ed25519 public key>"
    },
    "runner": {
      "algorithm": "ed25519",
      "key_id": "runner-1",
      "public_key": "<base64 raw Ed25519 public key>"
    },
    "assessor": {
      "algorithm": "ed25519",
      "key_id": "assessor-1",
      "public_key": "<different base64 raw Ed25519 public key>"
    }
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

所有公开输入与可选 runner adapter 必须是公开根目录内、非符号链接、最大 10 MB 的现有文件，并与声明 SHA-256 一致。环境白名单最多 32 个、名称限大写标识符，不得占用 runner 控制的变量；它只决定额外继承哪些宿主变量。全部用例共享同一预算；墙钟、模型调用和 token 上限均为正整数并有硬上界。冻结器不根据运行结果删除困难用例或改变预算。

`attestation` 可选，用于冻结 curator、runner 与 assessor 的 Ed25519 公钥。算法、最长 64 字符的 key ID 与 32 字节原始公钥均严格校验；任何角色不能复用 key ID 或公钥。为兼容第八十轮的双角色 suite，curator 可省略；新生产 suite 应声明 curator，使任务计划、gold 摘要和其余角色公钥也由独立密钥签名。旧套件可继续无签名运行；一旦声明 runner/assessor attestation，结果清单和 private assessment 都必须由对应私钥签名，不能局部降级成自报身份。

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
& .venv/Scripts/python.exe -m researchclaw.pipeline.benchmark_suite <public-root> --plan <public-root>/benchmark_plan.json --private-gold <private>/gold.json --output <public-root>/benchmark_suite.json --signing-key <private>/curator.pem
```

报告冻结规范化公开计划、原始 plan/gold 摘要、逐类别计数、scope、限制和内容版本。输出不得覆盖 plan 或任一公开输入。`verify_benchmark_suite` 从当前公开输入和私有 gold 完整重建报告，自洽重哈希后的字段修改仍会被拒绝。

若计划声明 curator，冻结器要求匹配私钥并对完整 suite report 签名；`verify_public_benchmark_suite` 在无需读取 gold 的 runner 侧先重建全部公开字段，再验证 curator 签名。因此同时替换 plan、gold 摘要、runner/assessor 公钥和普通内容版本也不能通过。未声明 curator 的兼容套件没有这层来源认证。

## 结论边界

`status: ready` 只表示用例覆盖、重复次数、预算、公开输入、可选适配器和私有 gold 形态满足契约。它不表示已经运行普通模型，不提供端到端成功率、错误接受率、成本或人工修订量，也不证明 gold 没有通过其他渠道泄漏。有限基准上的结果不能外推为任意论文质量保证。

## 公开盲测执行

冻结 runner 后可运行：

```powershell
& .venv/Scripts/python.exe -m researchclaw.pipeline.benchmark_runner --suite-report <public>/benchmark_suite.json --public-root <public> --plan <public>/benchmark_plan.json --results-root <results> --signing-key <private>/runner.pem
```

runner 仅调用 `verify_public_benchmark_suite`，不接收、定位或读取私有 gold。结果根必须为空；每个 case 恰启动一次 `python -I <adapter>`，工作目录为独立 case 目录。适配器通过 `ARC_BENCHMARK_CASE`、`ARC_BENCHMARK_INPUT`、`ARC_BENCHMARK_RUN_DIR` 和 `ARC_BENCHMARK_BUDGET` 获得冻结输入；除此之外只继承计划白名单与 Python/Windows 启动所需的固定环境名。每次尝试冻结 stdout/stderr、退出状态、环境名称集合、最终验收、资源账本和摘要；失败、启动错误或超时均保留，runner 继续后续 case。墙钟上限由宿主强制，超时会终止进程树。

该宿主 runner 不提供操作系统级文件或网络隔离。适配器仍可访问其账户本来可见的主机资源，白名单中的秘密也会传入；真实盲测应在无权读取 gold 的独立账户、容器或远程执行服务中运行。当前通用适配器协议无法拦截任意内部模型调用，因此 `model_calls` 和 `total_tokens` 从运行产物资源账本复算为 `audited_lower_bound`，评估器对冻结了适配器的新套件拒绝把它改写成强制计量。下界超过预算可确定为 `exceeded`；下界未超过只能记为 `unverified`，不能声称预算合规。未来若接入受信任执行后端，须先在公开计划中冻结新的计量策略及其证明契约。

## 签名密钥与盲评记录

密钥工具生成未加密 PKCS#8 私钥，并只在标准输出返回可公开的 signer JSON：

```powershell
& .venv/Scripts/python.exe -m researchclaw.pipeline.evidence_signature generate --private-key <private>/runner.pem --key-id runner-1
& .venv/Scripts/python.exe -m researchclaw.pipeline.evidence_signature generate --private-key <private>/assessor.pem --key-id assessor-1
& .venv/Scripts/python.exe -m researchclaw.pipeline.evidence_signature generate --private-key <private>/curator.pem --key-id curator-1
```

把三份公开 signer JSON 写入 plan 的 `attestation` 后，由 curator 签名冻结 suite。私钥不得放入公开根、结果目录或版本库；工具拒绝覆盖已有密钥，POSIX 上以 0600 创建，Windows 上仍须由操作者设置账户 ACL。runner 在启动第一个 case 前先验证 suite curator 签名，再检查自己的私钥与冻结公钥一致，最终对包含所有 case 证据摘要和内容版本的结果 manifest 签名。

盲评者完成未签名 assessment 后，以冻结的 assessor signer JSON 签名到新文件：

```powershell
& .venv/Scripts/python.exe -m researchclaw.pipeline.evidence_signature sign-json --input <private>/assessment.unsigned.json --output <private>/assessment.json --private-key <private>/assessor.pem --signer <private>/assessor_signer.json --purpose benchmark_private_assessment/v1
```

评估器先验证 suite curator 签名，在读取逐 case 内容前验证 runner manifest 签名，并在使用盲评分数前验证 assessment 签名与 assessor key ID。签名覆盖规范 JSON、内容版本和全部摘要；修改内容后重算普通 SHA-256 仍不能通过。签名证明持有相应私钥的一方签过这份字节语义，不证明密钥保管完善、持钥者的现实身份、runner 位于远程主机，或 assessor 事实上没有见过 gold；这些仍需组织访问控制和独立审计。

## 完成运行后的评估输入

`benchmark_evaluator.py` 消费完整的逐 case 结果，不允许只提交成功子集。公开结果清单必须绑定 suite version，并为每个 case 记录安全相对运行目录及以下三个 SHA-256：

- `benchmark_case_receipt.json`：`arc-benchmark-runner/v1` 回执，绑定 suite/case/公开输入摘要/统一预算、适配器摘要、开始结束时间、墙钟秒数、退出状态、环境名称、stdout/stderr 摘要、用量范围、模型调用和 token 计数。
- `final_acceptance.json`：正式流水线最终验收；评估层核对 schema、status/target 一致性与摘要。完整源运行仍须保留以接受普通可移植验收。
- `resource_ledger.json`：按源产物重新构建验证；其可见模型调用/token 是审计下界，runner 总量不得小于该下界。

私有 assessment 同样位于公开根之外，声明唯一 assessor ID、`blinded: true`，并逐 case 绑定最终验收摘要，记录 `observed_disposition`、critical error 数、人工修订分钟数、编辑次数和理由。三种 disposition 与 gold 相同：`accept`、`reject`、`honest_negative`。这些人工字段是盲评者的结构化记录，不是 UI 自动计时。

```powershell
& .venv/Scripts/python.exe -m researchclaw.pipeline.benchmark_evaluator --suite-report <public>/benchmark_suite.json --public-root <public> --plan <public>/benchmark_plan.json --private-gold <private>/gold.json --results-root <public>/results --result-manifest <public>/results/benchmark_result_manifest.json --private-assessment <private>/assessment.json --output <evaluation>/benchmark_evaluation.json
```

报告逐 case 保存期望/观察结局、错误接受、误拒、诚实负结果、critical errors、修订负担、runner 用量范围、资源账本可见下界、预算状态及三类证据摘要。总体、任务族和压力场景分别聚合 disposition accuracy；错误接受率只以 gold 中应拒绝用例为分母，误拒率只以应接受用例为分母，诚实负结果单独报告命中率，并分别统计 `budget_exceeded` 与 `budget_unverified`。缺类或错误用例不能通过缩小分母隐藏，因为 suite、结果和 assessment 都要求精确全覆盖。

无 attestation 的旧套件中 runner 和 assessor 身份仍为自声明；启用后，评估报告明确列出 curator/runner/assessor key ID，以及 suite 与结果/assessment 的签名验证状态。兼容的双角色 suite 会明确显示 suite 签名未验证。所有模式都没有远程证明。当前仓库提供公开逐 case 执行、签名与评估契约及 fixture 集成，尚未使用真实外部模型凭据运行生产套件，也未进行真人盲评，因此没有可发布的生产基准数值。
