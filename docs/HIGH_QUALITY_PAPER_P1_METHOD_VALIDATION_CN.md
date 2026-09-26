# 方法实现的预声明执行检查

MethodSpec 现在支持可选 `validation`。它随研究输入冻结，在正式测试矩阵的第一条调用之前执行。旧规范没有这个字段时仍只有静态映射，运行检查状态为 `not_declared`，不会被补写成通过。

## 契约与范围

一个验证计划有 `schema_version: 1`、`timeout_seconds`（1–120 秒）和 `cases`（1–32 项）。每项需要唯一 `id` 和 `kind`。除 `device_inventory` 外还需要 `call`、`atol`、`rtol`；两个容差必须在 0–0.01 之间。

`call` 指向已映射的步骤，例如：

```yaml
call:
  step: predict
  kwargs:
    X: [[1, 2], [3, 4]]
    W: [[2], [1]]
  constructor_kwargs: {}
```

除设备清单外，当前只支持顶层函数或 `Class.method`，每个步骤只有一个返回张量，调用使用关键字参数；`kwargs` 的键必须与步骤的 `inputs` 一致。因此对应 Python 实现需要保留这些参数名，例如 `Linear.predict(self, X, W)`。类方法调用会新建实例，可以用 `constructor_kwargs` 指定有界数值/布尔构造参数。不执行字符串表达式，也不从路径加载验证输入；数组是显式给定的合成实例。

数值输入须有限，数组须非空且矩形，每个张量最多 256 个元素、最多六维；命名维度按一次调用的一组输入统一约束。预期输出也校验声明形状。计划整体最多 256 KiB，总调用最多 256 次。探针输出支持普通数值、列表/元组及可转换的 NumPy/PyTorch 张量；异步函数及多返回值映射暂不支持。

七种检查：

- `microinstance`：增加 `expected`，逐元素比较返回值和预声明答案，同时比较完整形状。
- `gradient`：增加 `gradient_call`、`argument`、`epsilon`。`call` 返回标量损失，`gradient_call` 返回指定输入的梯度。两者的输入与构造参数必须相同；对最多 64 个坐标逐一做中心有限差分。扰动范围为 `1.0e-8`–`0.01`，过小到浮点无法表示的扰动在执行前拒绝。比较的是局部数值导数，不认证自动微分实现来源。
- `component_toggle`：增加 `parameter`、`expected_enabled`、`expected_disabled`。参数必须是已声明的布尔输入，分别设置为 true/false 后调用；两次输出都要符合预期。只检查这些实例的输出行为，不能证明模块已从内部计算图中移除，也不能证明普遍有效性。
- `autodiff`：增加 `argument`、`epsilon`。`call` 返回标量可微损失；探针进程在冻结前向代码上用 PyTorch 自动微分（`torch.float64`、`requires_grad`、`backward()`），宿主机对同一实例的最多 64 个坐标做中心有限差分并交叉比较，同时要求梯度形状与被微分的输入一致。未安装 PyTorch 时该检查失败，不降级为"仅有限差分"。这仍不是导数证明或后端认证：只覆盖一个进程内、声明实例上的交叉数值核对。
- `determinism`：无额外字段。同一调用先执行一次，在计划的全部其他调用执行完之后追加重复调用并逐元素比较；两次不一致说明存在跨调用状态污染。每个预声明调用前按冻结策略重新播种，以免调用顺序消耗随机状态造成误报。它只覆盖单个探针进程内、两次调用之间的重复性，不认证跨设备或分布式确定性。
- `process_determinism`：字段与 `determinism` 相同，但两次调用分别由两个新的 Python 解释器执行，并要求两个子进程 PID 互异且不同于控制进程。每个解释器在项目源码进入导入路径前应用同一冻结随机策略。该检查可发现进程身份、未固定外部熵等跨进程差异，也避免把一个解释器内遗留的模块全局状态误当成跨进程状态；计划时限至少为 2 秒，每个子进程最多使用总时限的一半。结论只适用于同一已配置主机/容器，不是跨主机、跨设备或分布式复现认证。
- `device_inventory`：不调用方法，改用 `required_devices`（`cpu`/`cuda`/`mps` 的非空无重复列表）、`minimum_cuda_devices`（0–64）和 `require_torch_determinism`。驱动在加入项目源码路径之前记录 Python/平台、三个 CUDA 环境变量，以及真实环境中 PyTorch 的版本、CUDA/MPS 可用性、CUDA 设备名称/计算能力/显存、确定性算法和 cuDNN 设置。CPU 总是可用；请求 CUDA/MPS、最小设备数或 PyTorch 确定性时按快照失败关闭。它只证明运行时状态已记录并满足声明，不证明方法实际在这些设备上执行，更不证明 GPU 算子本身确定。

每个请求还冻结 `runtime_policy={seed: 0, enforce_torch_determinism: ...}`。控制进程和 fresh-process 子解释器记录 Python、NumPy、PyTorch 与 CUDA 全设备播种的 `applied`/`unavailable`/`not_available`/`error` 状态；要求 PyTorch 确定性时还启用 deterministic algorithms、cuDNN deterministic 并关闭 benchmark。复验把这些状态与原始请求及设备要求重新绑定。该记录证明驱动尝试并观察到声明的运行时设置，不证明被测代码不会在调用内改写 RNG，也不证明具体 GPU kernel 位级确定。

对于[工作台示例](HIGH_QUALITY_PAPER_P1_WORKBENCH_CN.md)的线性投影，可添加：

```yaml
validation:
  schema_version: 1
  timeout_seconds: 10
  cases:
    - id: tiny_projection
      kind: microinstance
      call:
        step: predict
        kwargs:
          X: [[1, 2], [3, 4]]
          W: [[2], [1]]
      expected: [[4], [10]]
      atol: 0
      rtol: 0
```

## 执行与证据

执行器先冻结 MethodSpec、代码目录清单、后端与配置，再归档驱动和调用请求、保留单次时间预约。探针在独立子进程或配置的 Docker 后端运行；Docker 不可用时不能降级为宿主机运行。合成调用的临时项目位于研究目录之外，避免通用 sandbox 的祖先目录检测自动绑定训练、验证及测试特征分区。执行驱动只接收调用信息，没有预期答案，不输出可信的通过结论；宿主机检查器从原始数值重新计算结论。

`method_validation.json` 记录检查结论、已测试/未测试步骤、检查种类、代码与检查器版本、经严格模式校验的控制/子进程运行时快照、单独的验证时间账目，以及以下六项证据的摘要：

```text
evidence_artifacts/method_validation/
  reservation.json
  driver.py
  observations.json
  execution.json
  stdout.txt
  stderr.txt
```

这个时间预算独立于 `formal_matrix_host_wall_time`；它不是全流程资源/API 账本。失败、超时、缺失输出、中断预约都会阻止正式矩阵；不会自动重试探针。恢复成功运行时，重新核对源码与归档输出，不再次调用方法。已冻结规范/代码需要修正时，必须建立新运行，不能在读取测试结果后修改当前规范。

超时的 `charged_seconds` 至少为整个预约额度；正常执行使用测得的墙钟时间。中断后保留预约，不用缺失的回执声称零消耗。正式账本一旦存在，缺失探针报告不能触发新执行；验收还会核对其完成时间早于首个正式调用。

写作证据包加入经过核对的运行检查；最终验收和可移植交付同样重算比较，防止仅修改结论或重新生成其摘要来伪造通过。验证失败归属 experiments；已有研究内容审查、理论审查和最终审查仍分别保留。

## 不能外推的结论

通过只表示这些预声明实例符合预期。`semantic_equivalence` 始终为 `unresolved`，未测步骤显式列出。测试答案本身来自规范，仍需要科学审查；数值梯度不能替代证明，组件输出差异不能替代完整消融实验。

每次类调用使用新实例，并重置已加载的常用随机数生成器，但不认证全部库或 GPU 算子的状态隔离。`process_determinism` 为每次调用建立新解释器，仍共享主机内核、文件系统、驱动和外部服务。宿主机子进程不提供 OS 权限隔离；Docker 的既有共享缓存/网络策略也不是本模块新增的安全隔离。摘要和宿主机记录用于发现内容变化，不能防御恶意代码或能重写全部证据的本机操作者。

真实子进程测试使用小型合成函数，不是科研方法的效果评估。autodiff 用子进程内可用的真实后端或源码目录中的微型自动微分替身验证路径；无后端环境的诚实失败路径有专项测试。设备清单已有真实 CPU 环境路径和合成 CUDA 清单判定测试，但本机没有用真实 GPU 执行科研方法，不能声称 GPU/多设备方法验证已经完成。完整方法语义审查已有[独立通道](HIGH_QUALITY_PAPER_P1_METHOD_SEMANTICS_CN.md)，文本义务辅助拆分与 Z3 线性算术后端已并入工作台，各自的证明边界仍然保留。

真实本地多设备通信另有一个显式运行的有界探针：

```powershell
& .venv/Scripts/python.exe -m researchclaw.experiment.multi_device_probe <evidence-directory> --timeout-seconds 120 --max-devices 8
```

探针要求至少两张真实 CUDA 设备和可用 NCCL，按 rank 绑定不同设备，在最多 8 个新进程中执行一次 NCCL `all_reduce(SUM)`，同步设备，并冻结设备名、计算能力、显存、seed、实际归约值和内容哈希。父进程对所有 rank 共用一个 1–600 秒总时限，失败会终止残留进程；少于两张 CUDA 卡或 NCCL 不可用时写出 `unavailable`，不会合成通过记录。`verify_multi_device_probe` 重算哈希，并严格核对 scope、后端、版本、rank/设备一一对应、数值类型、seed 和归约和。它只认证单主机上这一次小型 collective；不执行科研训练代码，不认证性能、跨主机通信或多机复现。

输入加载同时修复了 `.json` 科学计数法被 YAML 1.1 读成字符串的问题：JSON 文件按 JSON 解析，YAML 文件保持其原有语法。JSON 中的 `1e-06` 可保留为数值；YAML 建议写成 `1.0e-6`，显式字符串仍不作为数值接受。

## 验证记录

较大范围回归 293 项通过；最后补充畸形归档证据处理后的定向回归 128 项通过，其中本模块新增 56 项测试。集合重叠，不累加。最后执行：

```powershell
& .venv/Scripts/python.exe -m pytest tests/test_method_validation.py tests/test_protocol_runner.py tests/test_research_workbench.py tests/test_final_acceptance.py -q --disable-warnings --maxfail=1
```

覆盖真实函数/实例调用、8 单元正式矩阵前置门禁、成功恢复不重跑、缺失证据不可后补、重摘要篡改、错误时间顺序、输出错误、调用/形状/资源上界、独立于原项目的归档复核、写作证据包和验收归属。超时与 Docker 不可降级采用故障注入；未启动真实容器或付费模型。

第十四轮补充 autodiff 与 determinism：新增 17 项测试，含真实子进程自动微分与有限差分交叉核对（微型自动微分替身放入冻结源码目录以遮蔽可选后端）、无后端诚实失败、有状态实现跨调用污染被重复调用检出、重复调用固定追加在全部其他调用之后，以及 8 组执行前拒绝的 autodiff 计划反例与调用预算反例。同轮更大范围回归数字见[开发状态](HIGH_QUALITY_PAPER_DEVELOPMENT_STATUS_CN.md)。

第二十八轮补充 `process_determinism` 与 `device_inventory`：方法验证全集 87 项通过。真实集成测试确认两次调用来自两个不同的新解释器；模块全局计数在两个新进程中各自从初始状态开始，而返回 PID 的方法会被判为不可重复。设备清单在项目源码加入导入路径前获取，源码目录中的 `torch.py` 不能伪造环境版本；真实 CPU 路径通过，CUDA/MPS、设备数量和确定性开关用严格结构化快照覆盖成功与失败判定。未在无 GPU 的本机把合成清单冒充真实硬件执行。

第七十三轮补充冻结随机状态策略：方法验证全集 89 项通过，方法验证及相关门禁扩大回归 161 项通过。真实 fresh-process 用 Python `random.random()` 验证逐调用重播种，合成 CUDA API 覆盖全设备播种与确定性开关；篡改 seed、后端状态或控制/子进程策略都会被可移植复验拒绝。

第七十四轮补充真实本地多设备 collective 探针：定向 19 项通过，包括有效三 rank 报告、scope/版本/布尔数值等自洽重哈希伪造、非通过状态与设备数量矛盾、重复设备、错误归约和、seed 篡改、预算边界及当前主机真实探测。当前环境没有可用 PyTorch/CUDA，实际报告为 `unavailable: fewer_than_two_cuda_devices`，因此本轮交付的是可执行、失败关闭且可复验的载体，没有产生真实 GPU 通过证据。
