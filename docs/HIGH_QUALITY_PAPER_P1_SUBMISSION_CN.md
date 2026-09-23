# 投稿文件包与严格编译

`deliverables` 是内部审计目录，包含原始数据、代码、文献与模板快照、历史记录和编译日志，不应整体上传到匿名投稿系统。结构化稿件打包现在另行生成 `submission.zip`，并在 `submission_bundle.json` 记录来源及检查状态。

## 投稿 ZIP 的内容

固定包含当前 `paper.tex`、`paper.pdf`、`references.bib`、编译后的 `paper.bbl`，按模板要求添加 `highlights.md`。图表、文档类、样式等从当前已校验的资产/模板中选择，只有被本次 TeX 编译实际读取的本地资源才能进入包；自定义 `.bst` 由已解析的 bibliography style 补入，因为它由 BibTeX 单独读取。显式 Biber 模板的本地 `.bbx/.cbx/.lbx` 由模板清单与编译 recorder 共同筛选，Biber 生成的 `.bbl` 仍是提交包中的固定书目产物。

编译使用 `-recorder` 生成 `paper.fls`，并把其 SHA-256 绑定到 `compilation.json`。ZIP 不包含 `.fls`、`.aux`、日志、原始模板入口、未使用的模板样例、研究数据和历史快照。安装在外部目录的 TeX 包及字体是运行环境依赖，其机器路径不会进入投稿 ZIP。相对路径越界和未列入当前契约的本地输入会使打包失败。

ZIP 的顺序、时间戳、文件权限固定，不加入用户名、源文件 mtime、运行路径或 run ID。验收重新构造允许的文件集合和 ZIP 字节；给篡改 ZIP 更新哈希，或额外塞入文件，不能获得通过。编译缺失、来源过期、PDF 内有嵌入附件、模板约束失败时不会保留旧投稿 ZIP 冒充当前成功输出。

这解决了“原始模板与历史材料自动被一并投稿”的问题，不保证正文、致谢、自引、图片内容或样式源码已经匿名。模板作者/引用作者等也不能通过简单删名字处理。

## 匿名验收

匿名模板要求 `final_reviews.json` 中额外存在 `dimensions.anonymity` 审核：

```json
{
  "input_version": "完整交付目录当前 inventory 的内容摘要",
  "dimensions": {
    "anonymity": {
      "status": "passed",
      "checker": "实际执行完整内容匿名检查的人或检查器",
      "evidence": "对应审核记录"
    }
  }
}
```

此示例不是自动填写通过的配置。没有实际审核时保留 unknown，不能成为 submission_candidate。审核绑定整个最终版本，任何后续修改使其失效。`submission_bundle.json.content_anonymity=unreviewed` 表示打包器本身没有执行内容审核，不能通过手工改该字段伪造完成。

## 严格编译

编译器现在要求末轮 TeX 成功、书目编译成功且生成新的 PDF。正式稿件的 `allow_repairs=False` 还拒绝未解析引用、丢失浮动体、无法显示的 Unicode 等错误，保留原始已审核 TeX/Bib 字节。

真实引擎集成用例：

```powershell
$env:ARC_RUN_TEX_INTEGRATION = '1'
& .venv/Scripts/python.exe -m pytest tests/test_real_publication_compile.py -q
```

这要求 pdflatex/xelatex 与 bibtex 已可用。用例覆盖单/双栏模板、公式、证明附录、控制流图、数值图、引用和最小文件包。测试数据、文字模型响应与评审均是明确的 fixture；它验证工程链路，不证明科研结论或实际模型能力。

2026-09-22 已在 MiKTeX 26.5 Portable 实际执行全部 4 组用例，包含投稿 ZIP 解压后重新编译与完整提取文本相等检查。加入字体下限、横向溢出、结果身份/数值同页、内部任务标题不进入正文等断言。最终相关测试为 84 passed；四份样稿共 38 页已用 Poppler 渲染查看。仍需改进结果表的紧凑展示和浮动体布局，不能把这些工程 fixture 当作可投稿论文。

后续紧凑结果表改造的最终相关回归为 **141 passed**，同样包含全部 4 组真实编译及 ZIP 重建。新增实际 PDF 的逐行方法/配置/种子/数值对应检查和章节浮动顺序检查；四份最终样稿共 **32 页**，全部 Poppler 渲染查看。完整配置标识使用等宽字体，避免 PDF 连字损坏复制的哈希。字号下限不变，日志无横向溢出。工程样稿正文较短，浮动页仍有较多留白；不能外推为生产论文的专业版式认证。

本会话安装的便携环境可在当前 PowerShell 进程中使用：

```powershell
$arcTexBin = Join-Path $env:TEMP 'arc-miktex-portable/portable/texmfs/install/miktex/bin/x64'
$env:PATH = $arcTexBin + [IO.Path]::PathSeparator + $env:PATH
```

它位于临时目录，清理后需重新安装或使用已有 TeX 环境。上述设置只影响当前进程，不修改系统 PATH。

临时便携引擎的安装依据 [MiKTeX 官方命令行文档](https://docs.miktex.org/manual/miktexsetup.html)，安装器摘要以[官方下载页](https://miktex.org/download)为准。运行中不把安装目录加入系统 PATH；测试进程通过自己的 PATH 使用它。
