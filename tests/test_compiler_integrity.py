import pytest

from researchclaw.templates import compiler


@pytest.fixture
def compiler_case(tmp_path, monkeypatch):
    tex = tmp_path / "paper.tex"
    tex.write_text(r"\documentclass{article}\begin{document}Text\bibliography{references}\end{document}")
    (tmp_path / "references.bib").write_text("@article{test,title={Test}}")
    monkeypatch.setattr(compiler.shutil, "which", lambda name: "fixture")
    monkeypatch.setattr(compiler, "_run_bibtex", lambda *a, **k: True)
    return tex


@pytest.mark.parametrize("failure", ["timeout", "exit", "missing_pdf", "stale_pdf", "bad_pdf", "bibtex", "unicode", "lost_float", "reference"])
def test_formal_compile_never_accepts_missing_or_broken_final_output(compiler_case, monkeypatch, failure):
    tex = compiler_case
    pdf = tex.with_suffix(".pdf")
    if failure == "stale_pdf":
        pdf.write_bytes(b"%PDF-old")
    calls = []
    def run(*args, **kwargs):
        calls.append(1)
        if failure not in {"stale_pdf", "missing_pdf"}:
            pdf.write_bytes(b"invalid" if failure == "bad_pdf" else b"%PDF-new")
        if len(calls) == 3:
            if failure == "timeout":
                return None, False
            if failure == "exit":
                return "no recognized LaTeX errors", False
            if failure == "unicode":
                return "! LaTeX Error: Unicode character not set up for use with LaTeX", True
            if failure == "lost_float":
                return "! LaTeX Error: Float(s) lost.", True
            if failure == "reference":
                return "LaTeX Warning: Reference `missing' undefined on input line 1.", True
        return "Compilation pass complete", True
    monkeypatch.setattr(compiler, "_run_pdflatex", run)
    if failure == "bibtex":
        monkeypatch.setattr(compiler, "_run_bibtex", lambda *a, **k: False)
    result = compiler.compile_latex(tex, allow_repairs=False)
    assert not result.success and result.errors


def test_formal_compile_allows_recovered_first_pass_but_requires_good_final_pass(compiler_case, monkeypatch):
    tex = compiler_case
    calls = []
    def run(*args, **kwargs):
        calls.append(1)
        tex.with_suffix(".pdf").write_bytes(b"%PDF-fresh")
        return ("! Undefined control sequence.", False) if len(calls) == 1 else ("Compiled", True)
    monkeypatch.setattr(compiler, "_run_pdflatex", run)
    result = compiler.compile_latex(tex, allow_repairs=False)
    assert result.success and result.errors == [] and len(calls) == 3
