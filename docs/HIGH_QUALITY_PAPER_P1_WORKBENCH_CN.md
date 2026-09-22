# P1 第四轮：方法与证明契约基础

本轮增加可校验的 MethodSpec 和 TheoryBundle，并接入输入冻结、模型上下文、正式执行与最终交付检查。它提供有边界的结构验证和精确命题检查，不宣称已实现通用理论发现、通用定理证明或代码语义等价判断。

## 配置

在 ResearchBrief 中按需声明相对路径：

```yaml
protocol_path: protocol.yaml
method_spec_path: method.yaml
theory_bundle_path: theory.yaml
```

两个新增文件支持 YAML/JSON，相对 brief 所在目录解析。MethodSpec 当前要求同时使用冻结实验协议，其 method_id 必须来自协议中的方法 ID。TheoryBundle 可以单独声明。源文件与编译产物都会冻结，恢复时不能改写；打包后重新编译规范进行核对。

## MethodSpec

示例 `method.yaml`（假设实验协议声明了 `proposed`）：

```yaml
schema_version: 1
method_id: proposed
description: 线性投影示例
variables:
  X: {shape: [n, d], description: 输入矩阵}
  W: {shape: [d, 1], description: 权重矩阵}
  Y: {shape: [n, 1], description: 预测值}
equations:
  - id: projection
    latex: 'Y = X W'
    inputs: [X, W]
    outputs: [Y]
steps:
  - id: predict
    phase: inference
    description: 将输入乘以权重矩阵
    inputs: [X, W]
    outputs: [Y]
    equations: [projection]
    depends_on: []
    code_file: model.py
    code_symbol: Linear.predict
losses: []
stopping_rule: 单次前向计算结束
complexity: {time: 'O(nd)', space: 'O(n+d)'}
shape_checks:
  - {op: matmul, left: X, right: W, result: Y}
```

变量、公式、步骤和代码符号采用明确 ID，避免写作阶段靠名字相似度推断对应关系。步骤阶段必须区分 train/inference，步骤依赖不能成环；算法迭代条件写入 stopping_rule，并可在独立的 `control_flow` 中显式声明分支、回环和停止节点，不以隐式循环依赖代替。参见[图与控制流说明](HIGH_QUALITY_PAPER_P1_DIAGRAMS_CN.md)。

当前形状检查支持同形状 add 和二维 matmul；不隐式广播，不自动猜测维度相等。复杂算子需要后续专用检查器。LaTeX、复杂度和步骤描述是规范内容，程序不会把其文字自动解释为已经证明正确。

宿主机执行前用 Python AST 检查指定文件和限定名符号存在，不导入、不执行被检查模块。缺文件、缺符号或语法错误会阻止正式矩阵启动。生成 `method_implementation.json`，绑定代码文件哈希与符号行号。最终验收对归档源码重做映射检查。

**mapped 不等于语义正确**：一个名字正确、函数体错误的实现仍可能完成结构映射，因此 `semantic_equivalence` 保持 `unresolved`。现已增加可选的预声明微型实例、有限差分梯度和组件开关检查，参见[执行验证说明](HIGH_QUALITY_PAPER_P1_METHOD_VALIDATION_CN.md)。有限实例通过仍不能替代公式、算法与实现的语义审核。

## TheoryBundle

以下是精确可判定的恒等式，不是一般收敛定理的替代品：

```yaml
schema_version: 1
definitions:
  domain: x 和 y 是任意实数
obligations:
  - id: square_identity
    required: true
    depends_on: []
    assumptions: []
    statement:
      kind: polynomial_identity
      variables: [x, y]
      left: '(x+y)**2'
      right: 'x**2 + 2*x*y + y**2'
```

受限检查器使用 Python AST 解析语法、Fraction 做精确有理数系数运算；不使用 eval、sympify 或导入用户代码。支持整数、有理常量除法、加减乘、有限非负整数幂。拒绝函数调用、属性访问、列表推导、变量作除数、小数浮点常量和超出资源上界的表达式。

左右多项式系数完全相同时标为 `machine_checked`；不同则标为 `disproved`，并在有界整数搜索中尝试提供具体反例。即使未找到小整数反例，不同系数也已构成“恒等式不成立”的精确判断。该检查器不解释文本假设，不可用文本假设掩盖不成立的无条件恒等式。

一般证明使用：

```yaml
statement:
  kind: informal
  text: 在所列假设下，算法具有某项性质。
proof_text: 完整证明尝试。
```

没有审查记录时仍是 `unresolved`。可以附 `review: {checker: ..., verdict: accepted/rejected/unresolved, evidence: ...}`，已接受者仅标为 `reviewed_informal`；这记录声明的审查，不自动认证审查者独立性或证明正确性。不能把 verdict 写成 machine_checked。

证明依赖必须存在且无环。依赖未解决会阻塞后续义务；存在非形式化审查依赖时，不会把整条链标为完全机器检查。必需义务为 unresolved/disproved 时，最终理论维度不能通过。

## 验证与后续

测试包含形状错配、未绑定公式、错误限定名、步骤/证明依赖环、无效与恶意表达式、精确分数运算、错误恒等式反例、非形式化证明状态、冻结后篡改、正式执行前映射失败与最终验收重新检查。

2026-09-22 最终相关回归 **180 passed**（34.15 秒），同时覆盖输入、协议、宿主机执行、独立评估、正式门禁和最终验收；`git diff --check` 与改动模块编译检查通过。命令：

```powershell
& .venv/Scripts/python.exe -m pytest tests/test_research_workbench.py tests/test_protocol_runner.py tests/test_experiment_protocol.py tests/test_research_inputs.py tests/test_final_acceptance.py tests/test_independent_evaluator.py tests/test_submission_gates.py -q --disable-warnings --maxfail=3
```

第十三轮已补充[预声明方法执行检查](HIGH_QUALITY_PAPER_P1_METHOD_VALIDATION_CN.md)。后续仍需完整方法语义审核、自动拆分与生成证明义务、定向反例工具、更丰富符号/维度检查和形式化证明后端。文献证据、ManuscriptIR、图像/模板已有后续集成，剩余范围见[开发状态](HIGH_QUALITY_PAPER_DEVELOPMENT_STATUS_CN.md)；持续目标保持进行中。
