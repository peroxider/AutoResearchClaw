# P1 第三轮：宿主机实验调度、预算账本与测试冻结

本轮把第二轮的实验协议接到真实执行链路。配置了 `ResearchBrief.protocol_path` 的本地 CSV 研究，在 Stage 12 由宿主机逐单元调用 sandbox/docker，生成执行回执和预测证据；无协议的旧流程保持兼容。

## 生成代码的接口

`main.py` 每次只执行一个矩阵单元。从 `os.environ["ARC_PROTOCOL_REQUEST"]` 读取 JSON：

| 字段 | 含义 |
|---|---|
| phase | 固定为 `frozen_test` |
| protocol_version / cell_id / key | 冻结协议与完整实验身份 |
| method | 本次方法、参数、角色和消融声明 |
| dataset | 数据卡、标签编码和项目相对路径 |
| dataset.paths | `train`、`validation`、`test_features` 的 CSV 路径 |
| output | 固定为 `protocol_predictions.csv` |
| budget_seconds | 本次剩余额度内的超时秒数 |
| tuning_trials | 协议声明的调参上限，当前仍需实现审核 |

输出必须是 `id,prediction` CSV，完整覆盖该单元测试 ID。分类使用数据卡中的数值标签编码，AUROC 使用正类得分。生成程序不需要也不能自行生成正式执行回执；stdout 中的自报分数不用于独立评估。

没有该环境变量的 CodeAgent 试跑应只使用训练/验证数据。宿主机正式执行前冻结项目文件清单、内容哈希、后端及配置；测试开始后修改代码或执行配置会拒绝恢复，必须新建研究运行。

## 执行与恢复

宿主机为每次尝试创建独立沙箱目录，并在启动前向账本追加带预算预留的 start 事件。进程结束后记录实际墙钟时间、退出码、超时、预测、stdout/stderr 和内容哈希，再追加 finish 事件。

- 成功单元恢复时直接复核并复用，不重新执行，不挑选更高的测试成绩。
- 失败尝试保留全部执行回执和日志。相同代码可以在剩余单元额度内重试；本轮不会自动修改冻结代码。
- 超时至少计费整个预留时长；宿主机中断且没有 finish 的尝试，也按全额预留计费。
- 单元额度和全局正式矩阵额度都会扣除已用时间，剩余不足一秒时不再启动。
- 操作系统文件锁防止两个宿主进程并发调度同一矩阵；进程退出后锁自动释放。
- JSONL 事件包含序号、前一事件哈希和当前内容哈希。预算检查点防止恢复时接受被截短的合法账本前缀。它是完整性机制，不是抵抗拥有整个目录写权限的攻击者的签名系统。
- 显式 Docker 后端不可降级为宿主机 subprocess；Docker 不可用时失败。

所有单元完成之前，不输出正式测试分数，也不创建完整评估 manifest。已有旧评估产物会被作废，避免失败重跑继承成功结果。全部单元完成后，宿主机一次性生成 `trusted_evaluation.json`，独立计算指标、生成 EvidenceStore 和覆盖报告，并将真实指标交给 Stage 14。

完整矩阵之后，Stage 13 的通用编辑/测试循环跳过；Stage 14 的通用自动 repair 不再运行；Stage 15 若要求基于测试结果 REFINE/PIVOT，会暂停并标记 `new_protocol_required`。零提升或负结果仍可作为诚实研究结论进入后续写作，不能为提升测试成绩继续改动本次配置。

## 新增产物

```text
<run>/
  protocol_code.json
  protocol_execution.jsonl
  protocol_budget.json
  trusted_evaluation.json
  evidence_store.json
  experiment_coverage.json
  evidence_artifacts/
    protocol_source/                 # 冻结项目源文件，数据分区另行打包
    protocol_runs/attempt-000001/
      execution.json
      predictions.csv               # 成功获得预测时存在
      stdout.txt
      stderr.txt
```

执行记录中的 `code_commit` 使用 `source-sha256:<hash>` 明确表示源文件快照，而不是虚构 Git 提交。环境记录包括实际调用的后端类、宿主机 Python/平台和冻结执行配置。完整运行时依赖锁、容器镜像摘要与硬件探针仍需后续补充，不能把这些字段视为已经记录了所有可复现环境信息。

最终交付包携带账本、预算、代码清单和完整尝试目录。最终验收重新检查源码、预测、执行回执和 stdout/stderr 哈希，重算时长与成功单元；篡改预算报告或删除失败回执不能依靠手写 `complete` 通过。

## 预算与隔离边界

测量范围明确为 `formal_matrix_host_wall_time`：包括正式单元调用和结果采集，保留失败消耗；不含此前 CodeAgent 试跑、模型调用、宿主机包安装与所有 API 费用。`tuning_trials` 是传给生成程序的约束，尚未将每次内部训练变成宿主机独立调用，所以实际调参次数仍需审核。

MethodSpec 声明 `validation` 时，另有[正式测试前的方法执行检查](HIGH_QUALITY_PAPER_P1_METHOD_VALIDATION_CN.md)。其合成实例、梯度和开关调用单独预约并记录耗时，不混入正式矩阵计量。缺失、失败或中断的检查阻止正式调用；测试开始后不允许补跑缺失的检查证据。可移植验收核对检查完成时间先于所有正式调用。

墙钟超时不等于 GPU 秒数、CPU 核时或费用计量。宿主机 subprocess 不提供测试标签的操作系统级权限隔离，也不宣称其超时能约束恶意创建的所有后代进程。Docker 的挂载和网络权限仍按现有后端配置。已隔离测试标签于实验项目之外，测试指标也只在固定矩阵结束后释放，但不能据此宣称具备对恶意代码的完整隔离保证。

## 验证

真实 Python 子进程集成测试运行 8 个单元：读取准备好的训练/测试特征，用训练集多数类输出预测，宿主机独立计算 accuracy=0.5。这验证执行路径，不代表测试代码实现了任何科学上的新方法。

定向测试还包括：成功恢复不重跑、失败保留和重试计费、超时耗尽、中断预留、并发锁、源码/预测/回执/日志/账本/预算篡改、合法前缀截断、旧指标作废、Docker 降级拒绝及 Stage 12/13/15 集成。Docker 用 mock 验证后端约束，未启动真实容器；未调用付费模型。

较大范围回归包含 executor、runner、Docker、repair、输入协议和验收，结果为 463 passed（部分最后补充检查另有定向回归）。后续继续方法/证明契约、依赖与资源计量、文献证据、写作与图像验收及端到端基准，不能将本轮完成等同于整份差距分析完成。
