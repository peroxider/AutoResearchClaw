# 方法实现的预声明执行检查

MethodSpec 现在支持可选 `validation`。它随研究输入冻结，在正式测试矩阵的第一条调用之前执行。旧规范没有这个字段时仍只有静态映射，运行检查状态为 `not_declared`，不会被补写成通过。

## 契约与范围

一个验证计划有 `schema_version: 1`、`timeout_seconds`（1–120 秒）和 `cases`（1–32 项）。每项需要唯一 `id`、`kind`、`call`、`atol`、`rtol`；两个容差必须在 0–0.01 之间。

`call` 指向已映射的步骤，例如：

```yaml
call:
  step: predict
  kwargs:
    X: [[1, 2], [3, 4]]
    W: [[2], [1]]
  constructor_kwargs: {}
```

当前只支持顶层函数或 `Class.method`，每个步骤只有一个返回张量，调用使用关键字参数；`kwargs` 的键必须与步骤的 `inputs` 一致。因此对应 Python 实现需要保留这些参数名，例如 `Linear.predict(self, X, W)`。类方法调用会新建实例，可以用 `constructor_kwargs` 指定有界数值/布尔构造参数。不执行字符串表达式，也不从路径加载验证输入；数组是显式给定的合成实例。

数值输入须有限，数组须非空且矩形，每个张量最多 256 个元素、最多六维；命名维度按一次调用的一组输入统一约束。预期输出也校验声明形状。计划整体最多 256 KiB，总调用最多 256 次。探针输出支持普通数值、列表/元组及可转换的 NumPy/PyTorch 张量；异步函数及多返回值映射暂不支持。

三种检查：

- `microinstance`：增加 `expected`，逐元素比较返回值和预声明答案，同时比较完整形状。
- `gradient`：增加 `gradient_call`、`argument`、`epsilon`。`call` 返回标量损失，`gradient_call` 返回指定输入的梯度。两者的输入与构造参数必须相同；对最多 64 个坐标逐一做中心有限差分。扰动范围为 `1.0e-8`–`0.01`，过小到浮点无法表示的扰动在执行前拒绝。比较的是局部数值导数，不认证自动微分实现来源。
- `component_toggle`：增加 `parameter`、`expected_enabled`、`expected_disabled`。参数必须是已声明的布尔输入，分别设置为 true/false 后调用；两次输出都要符合预期。只检查这些实例的输出行为，不能证明模块已从内部计算图中移除，也不能证明普遍有效性。

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

`method_validation.json` 记录检查结论、已测试/未测试步骤、检查种类、代码与检查器版本、单独的验证时间账目，以及以下六项证据的摘要：

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

每次类调用使用新实例，并重置已加载的常用随机数生成器，但不认证全部库、模块全局变量或 GPU 算子的状态隔离。宿主机子进程不提供 OS 权限隔离；Docker 的既有共享缓存/网络策略也不是本模块新增的安全隔离。摘要和宿主机记录用于发现内容变化，不能防御恶意代码或能重写全部证据的本机操作者。

真实子进程测试使用小型合成函数，不是科研方法的效果评估。专业自动微分后端、大模型/GPU 状态检查、完整方法语义审查、自动证明义务拆分和专业证明后端仍有待开发。

输入加载同时修复了 `.json` 科学计数法被 YAML 1.1 读成字符串的问题：JSON 文件按 JSON 解析，YAML 文件保持其原有语法。JSON 中的 `1e-06` 可保留为数值；YAML 建议写成 `1.0e-6`，显式字符串仍不作为数值接受。

## 验证记录

较大范围回归 293 项通过；最后补充畸形归档证据处理后的定向回归 128 项通过，其中本模块新增 56 项测试。集合重叠，不累加。最后执行：

```powershell
& .venv/Scripts/python.exe -m pytest tests/test_method_validation.py tests/test_protocol_runner.py tests/test_research_workbench.py tests/test_final_acceptance.py -q --disable-warnings --maxfail=1
```

覆盖真实函数/实例调用、8 单元正式矩阵前置门禁、成功恢复不重跑、缺失证据不可后补、重摘要篡改、错误时间顺序、输出错误、调用/形状/资源上界、独立于原项目的归档复核、写作证据包和验收归属。超时与 Docker 不可降级采用故障注入；未启动真实容器或付费模型。
