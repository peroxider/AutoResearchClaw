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
| tuning_trials | 协议声明的调参上限 |
| tuning | 固定输出名、validation 指标、上限和最终冻结参数的结构化调参披露契约 |

输出必须是 `id,prediction` CSV，完整覆盖该单元测试 ID。分类使用数据卡中的数值标签编码，AUROC 使用正类得分。每个单元还必须写出 `protocol_tuning_trials.json`：固定 schema 记录 validation 指标名、每个 trial 的唯一 ID、参数和有限 validation 指标，以及最终选中的 trial；没有调参的固定配置写空 trials 和 null 选项。trial 数不得超过协议上限，选中 trial 的参数必须逐字等于冻结 Method 配置。生成程序不需要也不能自行生成正式执行回执；stdout 中的自报分数不用于独立评估。

没有该环境变量的 CodeAgent 试跑应只使用训练/验证数据。宿主机正式执行前冻结项目文件清单、内容哈希、后端及配置；测试开始后修改代码或执行配置会拒绝恢复，必须新建研究运行。

## 执行与恢复

宿主机为每次尝试创建独立沙箱目录，并在启动前向账本追加带预算预留的 start 事件。进程结束后记录实际墙钟时间、退出码、超时、预测、stdout/stderr 和内容哈希，再追加 finish 事件。

- 成功单元恢复时直接复核并复用，不重新执行，不挑选更高的测试成绩。
- 失败尝试保留全部执行回执和日志。相同代码可以在剩余单元额度内重试；本轮不会自动修改冻结代码。
- 超时至少计费整个预留时长；宿主机中断且没有 finish 的尝试，也按全额预留计费。
- 单元额度和全局正式矩阵额度都会扣除已用时间，剩余不足一秒时不再启动。
- 操作系统文件锁防止两个宿主进程并发调度同一矩阵；进程退出后锁自动释放。
- JSONL 事件包含序号、前一事件哈希和当前内容哈希。预算检查点防止恢复时接受被截短的合法账本前缀。它是完整性机制，不是抵抗拥有整个目录写权限的攻击者的签名系统。
- 显式 Docker 后端不可降级为宿主机 subprocess；Docker 不可用时失败。正式矩阵额外要求 `network_policy=none` 和 `keep_containers=false`，不接受 setup-only/full 网络或保留容器。

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
      tuning_trials.json            # validation-only 调参披露，固定配置也必须存在
      stdout.txt
      stderr.txt
```

执行记录中的 `code_commit` 使用 `source-sha256:<hash>` 明确表示源文件快照，而不是虚构 Git 提交。环境记录包括实际调用的后端类、宿主机 Python/平台和冻结执行配置。依赖清单与容器镜像摘要已在后续轮次加入；硬件状态由方法验证设备清单覆盖其声明范围，不能把这些字段视为记录了未运行设备或跨主机复现。

最终交付包携带账本、预算、代码清单和完整尝试目录。最终验收重新检查源码、预测、执行回执和 stdout/stderr 哈希，重算时长与成功单元；篡改预算报告或删除失败回执不能依靠手写 `complete` 通过。

## 双运行独立复现比较

两个正式矩阵分别完成后，可生成一个严格的双运行比较报告：

```powershell
& .venv/Scripts/python.exe -m researchclaw.experiment.reproduction_comparison <left-run> <right-run> --left-site site-a --right-site site-b --output reproduction_comparison.json
```

比较器先对两侧分别执行完整的可移植执行束复验，重新独立计算全部指标，并要求冻结 EvidenceStore 除“复验器所在 Python 版本”之外的科学身份、数值、单位、执行回执、评估器代码哈希和原始产物哈希与重算结果一致；覆盖报告也必须与重新推导结果逐字段相同。输入身份不直接比较包含绝对源路径的本地 `research_contract.version`，而是比较 brief、runtime、数据描述、冻结输出哈希及按“源文件名+内容哈希”归一化的可移植身份。协议版本和实验源码哈希必须相同。

在没有预声明复现容差的 protocol v2 中，每个完整 EvidenceKey 的数值与单位必须精确相等。两侧记录集合缺项、源码/输入身份不同或任一数值不同都会得到 `mismatched`。报告还要求两条账本哈希不同，且两侧所有执行回执哈希不相交，因而直接复制一个完成目录不能获得 `matched`。依赖清单相同或不同都如实记录为 `environment_relation`；相同容器镜像在不同机器上运行仍可能显示 `same`。

`--left-site`/`--right-site` 只是有界的操作者自声明标签，不是物理主机认证。不同收据可排除逐字复制，但拥有全部目录写权限的人仍能重造哈希链；报告因此明确不声称数字签名、远程证明或真实跨主机身份认证。它证明两个完整、可复验且记录不同的执行束对同一冻结研究身份得到精确一致指标，不证明科研结论正确或可泛化。

## 预算与隔离边界

测量范围明确为 `formal_matrix_host_wall_time`：包括正式单元调用和结果采集，保留失败消耗；不含此前 CodeAgent 试跑、模型调用、宿主机包安装与所有 API 费用。第三十四轮开始，宿主机强制归档每个成功单元的 validation-only 调参披露，将实际 trial 数及逐单元计数写入 `protocol_budget.json`，并由运行级资源账本以 `validation_tuning_trials` 汇总；文件、哈希、收据计数或最终选型任一不一致都会失败关闭。

这些 trial 仍由实验程序自报，宿主机不会把每个内部训练拆成独立进程，也无法发现程序未披露的隐藏搜索；因此计数是可审计披露，不是对恶意或错误实现的完整监控。正式 test 标签仍不提供给程序，披露只允许 validation 指标，但宿主无法从一个标量证明实现确实只读取 validation 数据。

MethodSpec 声明 `validation` 时，另有[正式测试前的方法执行检查](HIGH_QUALITY_PAPER_P1_METHOD_VALIDATION_CN.md)。其合成实例、梯度和开关调用单独预约并记录耗时，不混入正式矩阵计量。缺失、失败或中断的检查阻止正式调用；测试开始后不允许补跑缺失的检查证据。可移植验收核对检查完成时间先于所有正式调用。

墙钟超时不等于 GPU 秒数、CPU 核时或费用计量。宿主机 subprocess 不提供测试标签的操作系统级权限隔离，也不宣称其超时能约束恶意创建的所有后代进程。第三十九轮收紧正式 Docker 路径：容器强制 `--network none`、只读根文件系统、丢弃全部 Linux capabilities、`no-new-privileges`、PID 上限和受限 `/tmp`；只挂载本单元 staging workspace，不挂载宿主 dataset/Hugging Face 缓存，也不转发 HF token。冻结 `isolation` 声明同时进入 `protocol_code.json`、start 事件和执行回执，可移植验证逐项核对。

这仍不是对任意 Docker/内核漏洞或 GPU 驱动侧信道的证明；workspace 必须可写以保存预测和遥测，容器镜像内部仍包含其自身文件。当前测试以命令构造和故障注入验证门禁，本轮没有在真实 Docker daemon 中执行恶意逃逸测试。宿主机模式明确记录 `host_guarded/v1` 与 `os_filesystem_isolation: unavailable`。

## 验证

真实 Python 子进程集成测试运行 8 个单元：读取准备好的训练/测试特征，用训练集多数类输出预测，宿主机独立计算 accuracy=0.5。这验证执行路径，不代表测试代码实现了任何科学上的新方法。

定向测试还包括：成功恢复不重跑、失败保留和重试计费、超时耗尽、中断预留、并发锁、源码/预测/调参披露/回执/日志/账本/预算篡改、错误选型、重复 trial ID、非有限 validation 指标、合法前缀截断、旧指标作废、Docker 降级拒绝、正式隔离参数/挂载/token 检查及 Stage 12/13/15 集成。Docker 用 mock 验证后端约束，未启动真实容器；未调用付费模型。

较大范围回归包含 executor、runner、Docker、repair、输入协议和验收，结果为 463 passed（部分最后补充检查另有定向回归）。后续继续方法/证明契约、依赖与资源计量、文献证据、写作与图像验收及端到端基准，不能将本轮完成等同于整份差距分析完成。

第七十五轮双运行比较定向回归 8 项通过；协议执行、实验协议、独立评估、依赖清单与最终验收扩大回归 115 项通过。测试实际建立不同目录、分别执行两套 8 单元正式矩阵并重算指标；覆盖有效匹配、复制目录拒绝、源码身份变化、报告篡改和站点标签边界。两套执行仍位于同一物理 Windows 主机，没有把自声明标签写成真实跨主机认证。
