# Manuscript source for Journal of Intelligent Information Systems

`main.tex` and `references.bib` are the current working manuscript source. The source uses the Springer Nature `sn-jnl` class and included bibliography styles.

Build from this directory with a LaTeX installation that includes `pdflatex`, `bibtex` or `bibtex8`, and the packages used by `sn-jnl`:

```bash
pdflatex main.tex
bibtex8 main.aux
pdflatex main.tex
pdflatex main.tex
```

The accompanying working PDF is supplied separately in the revision bundle. This source is not submission-ready until all authors verify contribution statements, the reference audit is completed, the public data/code snapshot is frozen, and the implementation/evaluation checks in the revision report are complete. The manuscript intentionally makes no unsupported benchmark-performance claims.
