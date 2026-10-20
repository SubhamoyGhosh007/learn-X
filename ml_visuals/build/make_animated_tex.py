"""Optional: build a LaTeX file whose PDF embeds each Manim scene as a playable animation (Acrobat Reader).
Requires: ffmpeg, TeX Live with the 'animate' package (texlive-latex-extra / full), rendered MP4s from manim/render_all.sh."""
import glob, os, subprocess
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); OUT = os.path.join(ROOT, 'build'); FPS = 15
scenes = ['GradientDescentScene', 'LinearRegressionScene', 'LogisticBoundaryScene', 'KMeansScene', 'PCAScene', 'OverfittingScene']
body = []
for s in scenes:
    hits = glob.glob(f'{ROOT}/manim/media/videos/ml_scenes/*/{s}.mp4')
    if not hits: print('missing', s, '(render it first)'); continue
    d = f'{OUT}/anim_{s}'; os.makedirs(d, exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', hits[0], '-vf', f'fps={FPS},scale=960:-1', f'{d}/f%04d.png'], check=True)
    n = len(glob.glob(f'{d}/f*.png'))
    body.append(f'\\section*{{{s}}}\n\\begin{{center}}\n\\animategraphics[controls,loop,width=0.9\\linewidth]{{{FPS}}}{{anim_{s}/f}}{{0001}}{{{n:04d}}}\n\\end{{center}}\n\\newpage')
open(f'{OUT}/animated_scenes.tex', 'w').write('\\documentclass{article}\n\\usepackage[landscape,margin=1.5cm]{geometry}\n\\usepackage{animate}\n\\pagestyle{empty}\n\\begin{document}\n' + '\n'.join(body) + '\n\\end{document}\n')
print('wrote build/animated_scenes.tex  ->  cd build && pdflatex animated_scenes.tex  (open in Adobe Acrobat Reader to play)')
