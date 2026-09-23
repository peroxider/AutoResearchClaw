# P1 本地模板导入与出版约束

结构化 ManuscriptIR 现在可以使用本地 LaTeX 模板目录或 ZIP。未指定模板时使用通用 journal 稿件；模板文件、解析后的版式及用户声明的规则一起冻结并加入稿件依赖。模板变更需要重新生成和复核受影响稿件。

## 配置与包结构

```yaml
export:
  template_path: "./my-journal-template"
  authors: "Author A and Author B"
```

路径相对于配置的 project root 解析。`template_path` 为空时使用通用稿件；它是结构化路径的选择项，旧探索性路径的 `target_conference` 注册表行为保留。

最小包包含 `main.tex`。如果只有一个含 `documentclass` 的 TeX 文件，也可自动找到入口。可选 `template.json`：

```json
{
  "schema_version": 1,
  "name": "my-journal",
  "entrypoint": "main.tex",
  "engine": "pdflatex",
  "bibliography_backend": "bibtex",
  "anonymous": true,
  "columns": 2,
  "max_pages": 16,
  "max_main_pages": 10,
  "max_title_characters": 180,
  "max_abstract_characters": 2000,
  "max_figures": 8,
  "max_tables": 6,
  "appendix_roles": ["theory", "results"],
  "highlights_required": true
}
```

引擎支持 `pdflatex` 和 `xelatex`；书目后端支持 `bibtex` 和显式 `biber`，需要相应可执行文件已安装。单/双栏可由明确的文档类选项推断，样式包内部决定栏数时应显式声明。页数、标题/摘要字符数和图表数量上限为正整数或 null；匿名和 highlights 字段必须为布尔值。正文页数指附录前的物理 PDF 页，包含出现在该区域的参考文献，不自动推断某个期刊对字数、参考文献和附录的特殊豁免。

标题字符上限在生成 TeX 框架前检查；摘要字符数从权威 ManuscriptIR 的 abstract blocks 重算；图和表的数量从最终 `paper.tex` 中的 `figure/figure*`、`table/table*/longtable` 环境重算，普通注释中的伪环境不计数。这些实际值进入 `template_constraints.json`，最终验收和 submission.zip 准备都会复核。字符数是 Unicode code point 数，不等同于任意期刊的自然语言词数；需要词数、彩页、补充材料或参考文献豁免等规则时仍须增加对应的显式检查器。

这些规则来自用户模板包的声明，不把默认值当成已经查证的期刊政策。支持的附录角色是 methods、theory、experiments、results、related_work、discussion；abstract、introduction、conclusion 保留在正文。

## LaTeX 入口的两种方式

标准入口包含一个 `documentclass`、`title`、`author`、`begin{document}`、`maketitle` 和 `end{document}`。导入器保留前导设置和书目风格，替换示例标题/作者，丢弃示例论文正文，再插入经过核验的稿件。不能把样例论文的研究结论带入新研究。

特殊标题区或定制期刊结构可显式提供以下五个标记，每个恰好一次：

```tex
\documentclass{article}
{{ARC_PACKAGES}}
\title{ {{ARC_TITLE}} }
\author{ {{ARC_AUTHORS}} }
\begin{document}
\maketitle
{{ARC_CONTENT}}
{{ARC_BIBLIOGRAPHY}}
\end{document}
```

正文与书目必须位于文档环境内且顺序明确；包声明必须位于前导区。BibTeX 模板已有 `bibliographystyle` 会被提取并由书目标记统一输出。BibLaTeX 模板必须在 `template.json` 显式声明 `bibliography_backend: biber` 并加载 `biblatex`；导入器移除样例 `addbibresource`/`printbibliography`，在前导区统一写入 `\addbibresource{references.bib}`，在正文书目位置写入 `\printbibliography`。声明与源码不一致会拒绝，绝不静默换成 BibTeX。模板可携带 `.bbx/.cbx/.lbx` 资源，其内容摘要、物化副本和编译输入均进入现有校验。复杂会议模板若没有标准标题宏，需要使用标记入口；当前不声称能自动转换任意宏语言。

## 版本与资源

原始包以内容摘要命名，保存在 `publication_templates/<version>/`；清单为 `publication_template.json`。新版本不覆盖旧模板快照。导出时复制所需的本地样式/类/图片等资源，并检查它们与冻结源一致。修改冻结文件或导出后的 `.sty` 都会导致验收失败。

导入不联网、不执行模板。目录/ZIP 有 100 项、20 MB 的边界，拒绝符号链接、越界路径、大小写冲突、Windows 设备名及生成稿件文件名冲突。模板入口只接受 UTF-8；资源文件名使用可移植的字母、数字、下划线、点和连字符。当前不支持所有字体/外部构建工具文件类型。TeX 编译显式传递 `-no-shell-escape`，这不是完整操作系统隔离的替代品；任意模板宏和外部系统包仍需要受控执行与完整文档检查。

## 论文内容与限制的联动

模板规则进入每个小节任务的上下文。附录角色重新排序后，Markdown、TeX 和每个数值跨度从共同源重新生成。双栏模板采用跨栏表格/图，避免在双栏中直接使用 longtable。数学算子只在尚未定义时声明，保留样式包提供的定义。

只有 `highlights_required=true` 才生成 `highlights.md`，内容来自已经校验的贡献记录，保留零提升/未决状态和适用范围。辅助文件摘要加入导出绑定，篡改 highlights 会失败。取消要求时只移除仍与旧清单摘要一致的系统生成 highlights，不删除无法确认归属的文件。

匿名模式替换作者参数，拒绝入口中残留的已识别身份字段。编译后另检查 PDF author 元数据。正文中的自引、机构、数据集路径或样式宏间接泄漏身份仍属于完整内容审查，不能由作者字段为空推断已完全匿名。模板原始快照仍保存在私有审计目录；新增的 [submission.zip 筛选与匿名审核门禁](HIGH_QUALITY_PAPER_P1_SUBMISSION_CN.md) 避免一并投稿这些快照，完整内容匿名审核仍为独立要求。

`template_constraints.json` 来自实际 PDF、权威 ManuscriptIR 和最终 TeX：总页数直接读取，附录前页数根据可提取且唯一的 `Appendix` 起始标题定位；标题/摘要字符数和图表数量按上述固定规则重算。标题缺失、重复或无法定位时为未知/失败，不能靠 `.aux` 页码计数器猜测。最终验收再次读取全部来源并比较约束报告，拒绝过期报告和超限文件。

## 验证边界

测试使用真实生成的多页 PDF 检查物理页数、附录定位、重复标题及作者元数据；另覆盖导入、版本失效、特殊入口、双栏资源、数值跨度、highlights 和编译命令。它们验证解析和门禁，不等于真实期刊样式的版面验收。

第十轮已用临时 MiKTeX Portable 运行 pdflatex/xelatex × 单/双栏真实整稿工程样例，并用 Poppler 渲染全部页面检查。第三十五轮增加 Biber 路由、统一资源绑定和模板资源校验；当前主机未安装可调用的 Biber/TeX，因此只用可控编译替身验证后端选择、禁止 BibTeX 替换和 `.bbl` 必需产物，尚未完成真实 Biber 整稿编译。通用 DiagramSpec 与矢量后备已接入；图像模型评审路线、生产论文的完整 PDF 内容/视觉审查、专业图表和期刊特有格式要求仍是持续目标的一部分。测试样稿及 fixture 评审不等于科学有效性证明。
