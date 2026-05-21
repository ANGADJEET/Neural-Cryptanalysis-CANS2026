import os
import glob
import re

paper_dir = r"c:\Documents\college\SEM-7\ac\project\neural_cryptanalysis\paper"

tex_files = glob.glob(os.path.join(paper_dir, "**", "*.tex"), recursive=True)

for filepath in tex_files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace [t] or [h] with [htbp] for figure, figure*, table, table*
    content = re.sub(r'\\begin\{(figure\*?|table\*?)\}\[t\]', r'\\begin{\1}[htbp]', content)
    content = re.sub(r'\\begin\{(figure\*?|table\*?)\}\[h\]', r'\\begin{\1}[htbp]', content)
    
    # Add \FloatBarrier before \subsection in sections with many floats
    if any(x in filepath for x in ["experiments.tex", "additional.tex", "discussion.tex", "ablations.tex"]):
        content = re.sub(r'(?<!\\FloatBarrier\n)\\subsection\{', r'\\FloatBarrier\n\\subsection{', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# Update main.tex to include placeins
main_tex = os.path.join(paper_dir, "main.tex")
with open(main_tex, 'r', encoding='utf-8') as f:
    main_content = f.read()

if r"\usepackage{placeins}" not in main_content:
    main_content = main_content.replace(r"\usepackage{float}", "\\usepackage{float}\n\\usepackage{placeins}")
    with open(main_tex, 'w', encoding='utf-8') as f:
        f.write(main_content)

print("Formatting updated.")
