# LaTeX Course Report Cleanup Prompt

## Current Issues

Your course report (`course_report/main.tex`) has several critical problems:

1. **Missing Bibliography References** - All citations show as `??` because:
   - `references.bib` file is missing or incomplete
   - Citations like `gohr2019improving`, `speck2013`, `biham1991differential`, `belghazi2018mine` are undefined

2. **Figure Placement Issues** - Figures are not rendering where expected due to:
   - Improper float placement specifiers
   - Missing `\clearpage` or `\FloatBarrier` commands between sections
   - Too many figures competing for limited space

3. **Document Flow Problems**:
   - Sections may be running together without proper breaks
   - Figure/table references may be out of order
   - Content organization needs better logical flow

## Required Fixes

### 0. Update Title Page

Fix the institution and author details in `main.tex`:
```latex
\title{
    \usefont{OT1}{bch}{b}{n}
    \normalfont \normalsize \textsc{IIIT-Delhi} \\ [25pt]
    \horrule{0.5pt} \\[0.4cm]
    \huge Neural Cryptanalysis of SPECK32/64 \\
    \horrule{2pt} \\[0.5cm]
}
\author{
    \normalfont \normalsize
    Angadjeet Singh \\
    2022071 \\[-3pt] \normalsize
    \today
}
```

### 1. Create/Fix Bibliography File

Create `course_report/references.bib` with all required citations:

```bibtex
@inproceedings{gohr2019improving,
  title={Improving attacks on round-reduced Speck32/64 using deep learning},
  author={Gohr, Aron},
  booktitle={Annual International Cryptology Conference},
  pages={150--179},
  year={2019},
  organization={Springer}
}

@techreport{speck2013,
  title={The SIMON and SPECK families of lightweight block ciphers},
  author={Beaulieu, Ray and Shors, Douglas and Smith, Jason and Treatman-Clark, Stefan and Weeks, Bryan and Wingers, Louis},
  year={2013},
  institution={Cryptology ePrint Archive}
}

@inproceedings{biham1991differential,
  title={Differential cryptanalysis of DES-like cryptosystems},
  author={Biham, Eli and Shamir, Adi},
  booktitle={Journal of Cryptology},
  volume={4},
  number={1},
  pages={3--72},
  year={1991}
}

@inproceedings{belghazi2018mine,
  title={Mutual information neural estimation},
  author={Belghazi, Mohamed Ishmael and Baratin, Aristide and Rajeshwar, Sai and Ozair, Sherjil and Bengio, Yoshua and Courville, Aaron and Hjelm, Devon},
  booktitle={International Conference on Machine Learning},
  pages={531--540},
  year={2018},
  organization={PMLR}
}
```

### 2. Fix Figure Placement

Add to preamble in `main.tex`:
```latex
\usepackage{placeins}  % For \FloatBarrier
```

Then strategically add `\FloatBarrier` commands:
- After each major subsection with multiple figures
- Before starting new sections
- Between experiments to prevent figure drift

Example pattern:
```latex
\subsection{E01: Accuracy vs Rounds}
[content with table and figure]
\FloatBarrier

\subsection{E02: Representation Comparison}
[content]
\FloatBarrier
```

### 3. Improve Float Specifiers

Change all figure/table environments from:
```latex
\begin{figure}[hbt!]
```

To more permissive:
```latex
\begin{figure}[htbp]
```

This allows LaTeX more flexibility: here (h), top (t), bottom (b), or separate page (p).

### 4. Fix Document Flow

**Add section breaks:**
```latex
\clearpage  % Force new page before major sections
\section{Core Experiments}
```

**Reorder content if needed:**
- Ensure figures are referenced BEFORE they appear
- Move large figures to end of subsections
- Consider using `\begin{figure*}` for wide figures if using two-column layout

### 5. Compile Multiple Times

LaTeX requires multiple compilation passes:
```bash
cd course_report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The sequence is critical:
1. First pass: generates `.aux` file with citation keys
2. BibTeX: processes bibliography
3. Second pass: incorporates citations
4. Third pass: resolves all cross-references

### 6. Check for Additional Issues

After fixing bibliography and figures, check for:
- Undefined labels (Table~\ref{tab:xyz} where label doesn't exist)
- Overfull/underfull hbox warnings (text extending into margins)
- Missing figure files in `figures/` directory
- Proper caption formatting and numbering

## Verification Steps

1. **Check log file** for remaining warnings:
   ```bash
   grep -i "warning\|undefined\|missing" course_report/main.log
   ```

2. **Verify all figures render** by checking PDF page-by-page

3. **Confirm bibliography** shows actual citations, not `??`

4. **Test cross-references** - all Table/Figure references should show numbers

## Quick Fix Priority

1. **CRITICAL**: Create `references.bib` with all citations
2. **HIGH**: Add `\FloatBarrier` after each experiment subsection
3. **MEDIUM**: Change float specifiers to `[htbp]`
4. **LOW**: Fine-tune spacing and page breaks

## Expected Outcome

After applying these fixes:
- All citations display properly with author names and years
- Figures appear near their references, not pages away
- Document flows logically from section to section
- No `??` marks anywhere in the PDF
- Professional appearance suitable for course submission
