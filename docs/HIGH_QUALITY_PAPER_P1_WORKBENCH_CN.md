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

第十五轮扩展的形状/类型检查器逐算子判定声明的形状关系：`add`/`elementwise_mul`（同形）、`matmul`（二维内积）、`broadcast_add`（按逐维广播规则核对**显式声明**的结果形状，不替声明者发明结果）、`concat`（非拼接轴逐维相等，拼接轴要求具体整数并核对长度相加）、`transpose`（二维换轴）、`reshape`（要求具体整数形状且元素总数相等）、`reduce`（移除声明的唯一合法轴）、`cast`（要求形状相同、两侧变量声明 `dtype` 且结果 `dtype` 等于转换目标）。变量可声明白名单内的 `dtype`（float16/bfloat16/float32/float64/int8/int16/int32/int64/bool）；参与同一检查的变量一旦声明就必须一致，未声明不施加约束。所有维度比较都是符号的语法相等（`n` 只等于 `n` 或 1），不隐式广播、不自动猜测维度相等；需要算术（元素计数、长度求和）的算子明确要求具体整数维度并拒绝符号推断。LaTeX、复杂度和步骤描述是规范内容，程序不会把其文字自动解释为已经证明正确。

宿主机执行前用 Python AST 检查指定文件和限定名符号存在，不导入、不执行被检查模块。缺文件、缺符号或语法错误会阻止正式矩阵启动。生成 `method_implementation.json`，绑定代码文件哈希与符号行号。最终验收对归档源码重做映射检查。

**mapped 不等于语义正确**：一个名字正确、函数体错误的实现仍可能完成结构映射，因此 `semantic_equivalence` 保持 `unresolved`。现已增加可选的预声明微型实例、有限差分梯度、组件开关、自动微分与确定性检查，参见[执行验证说明](HIGH_QUALITY_PAPER_P1_METHOD_VALIDATION_CN.md)；以及基于重算证据包的具名人工语义审核通道，参见[方法语义审核](HIGH_QUALITY_PAPER_P1_METHOD_SEMANTICS_CN.md)。有限实例与人工审核仍不能替代机器证明，冻结规范内的 `semantic_equivalence` 不会被审核改写。

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

第十四轮增加三类精确语句，全部沿用同一原则：机器检查器不读文本假设，不接受 prose 覆盖，资源超界即拒绝。

- `rational_identity`：有理函数恒等式。表达式展开为 (分子, 分母) 多项式对，支持加减乘除与 −12..12 的整数幂（负幂即倒数，如 `x**-1`），零分母多项式在编译期拒绝。交叉乘积相等时 `machine_checked`，否则 `disproved` 并在避开分母零点的小整数网格上寻找反例；机器检查器为 `rational_function_identity/v1`。
- `symbolic_equality`：可选 SymPy 后端的符号等式。支持 `sin/cos/tan/exp/log/sqrt` 单参调用、整数与十进制字面量、有限幂。语法用白名单 AST 校验，构造只经 `Integer/Rational/Symbol`，不使用 `eval`/`sympify`。**即使后端未安装也先校验语法**，避免机器差异悄悄放宽验收标准。化简差恰为零时 `machine_checked`（`sympy_symbolic_equality/v1`）；差为非零常数时 `disproved` 并记录差值；后端缺失或无法判定（如需要定义域假设的 `log(x*y)`）保持 `unresolved` 并如实记录。
- `conjunction`：结构化合取义务自动拆分。编译期确定性地生成 `{id}_part{i}` 子义务（继承 required/depends_on，ID 冲突即拒绝），逐部分用对应检查器判定后聚合父状态：任一 `disproved` → disproved；全部 `machine_checked` 且无非形式化依赖 → `machine_checked`；全部为机器或已审非形式化 → `reviewed_informal`；否则 `unresolved`。未满足依赖会下传阻塞各部分。

第二十六轮增加 `linear_arithmetic` 专业求解后端。它检查有限的量词自由线性整数/实数蕴含式，使用 Z3 的 QF_LIA/QF_LRA 决策过程；Z3 官方将线性实数算术列为其算术理论之一，并明确非线性整数算术不可判定，因此本接口不把语法扩展到非线性项（[Z3 Arithmetic](https://microsoft.github.io/z3guide/docs/theories/Arithmetic/)）。项目提供 `theory` 可选依赖：`pip install '.[theory]'`。

```yaml
statement:
  kind: linear_arithmetic
  domain: real
  variables: [x, y]
  premises:
    - coefficients: {x: 2, y: '-1/2'}
      relation: '<='
      constant: 3
    - coefficients: {x: 1}
      relation: '>='
      constant: 2
  conclusion:
    coefficients: {y: 1}
    relation: '>='
    constant: 2
```

每个约束表示 `sum(coefficients[name] * name) relation constant`。系数与常数只接受整数或规范有理数字符串 `p/q`；不接受浮点数、表达式或代码。最多 16 个变量、64 个前提，分子/分母最多 512 bit。关系限于 `< <= == != >= >`。机器检查器先独立求解前提：前提不可满足时保持 `unresolved` 并标记 `vacuous_implication_rejected`，不利用逻辑空真把矛盾假设包装成有效科研结论。前提可满足时再求解“前提且结论否定”：`unsat` 才标为 `machine_checked`；`sat` 标为 `disproved`，并将 Z3 模型转换为精确 Fraction，由宿主检查器重新验证所有前提为真且结论为假。`unknown` 或后端缺失均保持 `unresolved`。求解器限定 2000 ms 和确定性资源上限，记录逻辑、Z3 版本、结果与检查方法。

第二十九轮把检查器升级为 `z3_linear_arithmetic_implication/v2`，为实数域的非严格不等式/等式蕴含生成可移植 Farkas 证书。前提先规范展开成 `A x <= b`；证书保存每个规范行的稳定名称、非负精确有理乘子和合成常数。等式结论必须分别给出上界与下界证书。证书还保存一个覆盖全部变量的精确前提可满足见证，独立验证器先检查见证确实满足全部原始前提，再用 Python `Fraction` 验证 `lambda >= 0`、`lambda A = c`、`lambda b <= d`，整个重放过程不导入 Z3。语句哈希、行顺序、乘子数、目标方向和全部字段均严格校验；修改见证、乘子、合成常数或身份都会失败。首次 Z3 编译输出中的证书可复制到原语句的 `portable_certificate` 字段并随输入冻结；后续 `compile_theory`、最终验收与可移植交付会改用 `exact_fraction_farkas/v1` 重放，即使环境没有 Z3 也保持 `machine_checked`。证书身份按移除该字段后的基础语句计算，避免自引用哈希。

这是一条受限专业后端，不是任意定理证明器。实数闭合线性约束的受支持子集已有独立精确有理证书。第四十轮把其中一部分安全扩展到整数声明：若同一非严格线性蕴含能用 Farkas 组合证明在全部实数上成立，则它当然也在整数子集上成立，生成 `farkas_linear_implication_over_reals` v2 证书并明确记录 `proof_domain: real_superset_of_integer_domain`；离线复核仍只用 `Fraction`。第四十四轮再为可从单变量前提推导出每个变量有限整数上下界、笛卡尔积不超过 10000 点的 QF_LIA 语句生成 `exhaustive_bounded_integer_implication` 证书。`exact_bounded_integer_enumeration/v1` 从原语句重新推导边界，逐点用精确 Fraction 核对全部前提与结论，并核对总点数、满足前提点数和首个可满足见证；因此能覆盖依赖整数离散性的结论、严格关系和 `!=`，无需导入 Z3。无显式有限单变量边界、边界空间超限或其他整数逻辑仍保留 `portable_certificate: null`，不能把有限穷举外推成通用整数证明。最终验收仍重新编译整份理论束。Z3 Python 包按其官方安装方式由 `z3-solver` 提供（[Z3 repository](https://github.com/Z3Prover/z3#python)）。结构化前提取代 prose assumptions；`proof_text` 和人工 review 不能覆盖机器结果。

### 文本义务的辅助拆分

`informal_decomposition` 保留原始文本命题，并把模型或研究者建议的证明路线变成有 ID、有假设、有内部依赖的子义务。它不尝试用标点或关键词“自动理解”自然语言。拆分是否充分覆盖原命题必须由单独的 `coverage_review` 判断：

```yaml
statement:
  kind: informal_decomposition
  text: 在条件 R 下，更新保持可行性并降低目标函数。
  parts:
    - id: feasibility
      depends_on: []
      assumptions: []
      statement:
        kind: linear_arithmetic
        domain: real
        variables: [x]
        premises:
          - {coefficients: {x: 1}, relation: '>=', constant: 1}
        conclusion: {coefficients: {x: 1}, relation: '>=', constant: 0}
    - id: descent
      depends_on: [feasibility]
      assumptions: []
      statement:
        kind: polynomial_identity
        variables: [d]
        left: '(d-1)**2'
        right: 'd**2-2*d+1'
  coverage_review:
    checker: 具名拆分审核者
    verdict: accepted
    evidence: 两个子义务在原命题的条件下共同覆盖可行性与下降结论。
```

编译器生成 `{parent}__{part}` 子义务，保留 part assumptions、part proof/review、父级外部依赖和 part 内部依赖，并拒绝缺失引用、环、重复 ID、生成 ID 冲突及 2–16 项之外的拆分。每个 part 仍由其自身的严格检查器判定。前置 part 未解决时，依赖它的后续 part 降为 unresolved。

父命题只有在覆盖审核为 `accepted`、外部依赖均解决且所有子义务为 `machine_checked` 或 `reviewed_informal` 时才成为 `reviewed_informal`。即使所有子义务都是机器检查，覆盖关系仍来自自然语言审核，所以父命题永远不能被升级为 `machine_checked`。缺覆盖审核、rejected/unresolved 审核或不完整 part 都使父命题 unresolved。某个 part 被反证表示当前证明路线失败；它不会自动把原文本命题标为 disproved。审核记录与所有子状态进入哈希并在验收时重编译。

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

测试包含形状错配、未绑定公式、错误限定名、步骤/证明依赖环、无效与恶意表达式、精确分数运算、错误恒等式反例、非形式化证明状态、冻结后篡改、正式执行前映射失败与最终验收重新检查。第十四轮补充有理函数恒等式（含 `x**-1`、`(x-1)/(x**2-1) = 1/(x+1)`、假恒等式的避开分母零点反例）、符号后端判定/无法判定/后端缺失三态、合取自动拆分与状态组合、依赖下传与 ID 冲突反例。第二十六轮补充真实 Z3 4.16 的整数/实数蕴含、精确有理系数、六种关系、反例二次核验、矛盾前提拒绝、后端缺失三态、资源上界与 prose 逃逸反例；同时修复自动拆分子义务进入出版资产时的来源映射。

2026-09-22 最终相关回归 **180 passed**（34.15 秒），同时覆盖输入、协议、宿主机执行、独立评估、正式门禁和最终验收；`git diff --check` 与改动模块编译检查通过。命令：

```powershell
& .venv/Scripts/python.exe -m pytest tests/test_research_workbench.py tests/test_protocol_runner.py tests/test_experiment_protocol.py tests/test_research_inputs.py tests/test_final_acceptance.py tests/test_independent_evaluator.py tests/test_submission_gates.py -q --disable-warnings --maxfail=3
```

2026-09-23 第十五轮形状/类型检查器扩展后，工作台全集 **73 passed**（新增逐算子正向编译与 21 个参数化反例：广播结果不符、符号维度不可广播、拼接轴符号/求和/越界、transpose 秩与多余字段、reshape 元素计数与符号维度、reduce 轴越界/重复/保留维度、dtype 白名单/不一致/cast 未声明与改形状）。

第二十六轮 Z3 后端集成后，工作台全集 **96 passed**；相关扩大回归 **373 passed, 6 skipped**。第二十七轮文本辅助拆分后，工作台全集 **114 passed**，覆盖审核三态、内部/外部依赖下传、机器与人工 part 聚合、错误 part 不外推为父命题反证，以及畸形/碰撞/循环与最终验收反例；跨出版资产、最终验收、稿件、贡献链、输入、方法验证/语义、投稿与模板的扩大回归 **363 passed**（集合重叠，不累加）。第四十四轮有限整数域穷举证书后，工作台全集 **119 passed**，相关扩大回归 **254 passed**。第二十六轮跳过项是未启用显式开关的真实 TeX 集成用例，不涉及证明后端。`uv lock --check`、Python 编译与 CR 感知差异空白检查通过。

第十三轮补充[预声明方法执行检查](HIGH_QUALITY_PAPER_P1_METHOD_VALIDATION_CN.md)，第十四轮补充语义审核通道（[方法语义审核](HIGH_QUALITY_PAPER_P1_METHOD_SEMANTICS_CN.md)）与扩展检查器，第十五轮扩展逐算子形状/类型检查器，第二十六轮接入 Z3 有界线性算术后端，第二十七轮接入文本义务辅助拆分，第四十四轮接入有限整数域的可移植精确穷举证书。仍待开发：更强的形式化语言、未覆盖/超界整数域的可移植证书后端及 GPU/多设备状态检查。文献证据、ManuscriptIR、图像/模板已有后续集成，剩余范围见[开发状态](HIGH_QUALITY_PAPER_DEVELOPMENT_STATUS_CN.md)；持续目标保持进行中。
