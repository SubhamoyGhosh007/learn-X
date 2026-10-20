"""
Builds ML_Visual_Companion.pdf.

For every Manim scene it looks for a rendered MP4 (manim/media/videos/ml_scenes/*/<Scene>.mp4).
  * MP4 found  -> extracts N evenly spaced frames with ffmpeg and lays them out as a filmstrip (real Manim frames).
  * MP4 missing -> falls back to the matplotlib storyboard (same algorithm, same data, same numbers).
Graphviz PNGs are placed one per page.

Usage:   python build_companion_pdf.py [--out ML_Visual_Companion.pdf] [--frames 8]
Needs:   pip install reportlab pillow   and   ffmpeg on PATH (only for the MP4 path).
"""
import argparse, glob, os, subprocess, tempfile
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image, PageBreak, KeepTogether, Preformatted
from PIL import Image as PILImage

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GV = sorted(glob.glob(f'{ROOT}/graphviz/*.png'))
SB = f'{ROOT}/storyboards'

SCENES = [  # (scene class, filmstrip file prefix, title, what to watch, report section)
 ('GradientDescentScene', '1_GradientDescentScene', 'Gradient descent: converge vs. diverge',
  'A dot slides down the loss curve f(x) = x^2/2. With alpha = 0.3 each step shrinks x by 30% and the dot settles in the valley. With alpha = 2.2 each step overshoots by more than it corrects, so x flips sign and grows: divergence.', 'Sections 5 and 9'),
 ('LinearRegressionScene', '2_LinearRegressionScene', 'Linear regression by gradient descent',
  'Twenty-five noisy points were generated from y = 1.8x + 1. The line starts at a bad guess (slope -1, intercept -1) and is pulled onto the cloud; the MSE printed underneath falls at every step.', 'Section 9'),
 ('LogisticBoundaryScene', '3_LogisticBoundaryScene', 'Logistic regression: the boundary learns',
  'Two classes of 30 points each. The white line is where w.x + b = 0 (probability exactly 0.5). It starts badly oriented, rotates and shifts as log-loss falls, and ends between the classes.', 'Section 10'),
 ('KMeansScene', '4_KMeansScene', 'k-means: assign, move, repeat',
  'Three centroids start clumped in the middle. Each iteration has two beats: points take the colour of the nearest centroid, then each centroid jumps to the mean of its points. Stop when nothing changes.', 'Section 18'),
 ('PCAScene', '5_PCAScene', 'PCA: project onto the direction of greatest variance',
  'Correlated 2-D data is centered, the two eigenvectors of the covariance matrix are drawn as axes (longer arrow = more variance), and every point is dropped onto PC1, compressing 2-D to 1-D.', 'Section 19'),
 ('OverfittingScene', '6_OverfittingScene', 'Underfitting vs. overfitting',
  '12 noisy samples of a sine wave. Raising the polynomial degree always lowers training error, but test error is lowest near degree 3 to 5 and climbs again as the curve starts chasing noise.', 'Sections 8 and 11'),
]
GV_NOTES = {
 '01_ml_workflow': ('The end-to-end workflow', 'Section 32. Dashed arrows are the loops that matter: weak validation scores send you back to features, never to the test set.'),
 '02_ml_taxonomy': ('A map of classical machine learning', 'Sections 1 and 2, and the algorithms of Part B.'),
 '03_algorithm_chooser': ('Which algorithm should I try first?', 'Section 20 summary map, as a decision flowchart. Always start with a simple baseline.'),
 '04_cross_validation': ('Data splitting and 5-fold cross-validation', 'Sections 8 and 20. Every sample is used for validation exactly once; the test set stays sealed.'),
 '05_bagging_vs_boosting': ('Two ways to combine trees', 'Sections 15 and 16. Bagging averages independent deep trees (cuts variance); boosting adds shallow trees one after another (cuts bias).'),
 '06_decision_tree_example': ('A small decision tree', 'Section 14. Entropy values use H = -sum p log2 p; the best split maximizes information gain.'),
 '07_gradient_descent_loop': ('Gradient descent: the learning loop', 'Sections 5, 9 and 10.'),
 '08_sklearn_pipeline': ('A scikit-learn Pipeline with ColumnTransformer', 'Sections 25 and 26. Everything inside is fitted on training folds only, which prevents leakage.'),
}

def extract_frames(mp4, n, outdir):
    dur = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', mp4]).decode().strip())
    paths = []
    for i in range(n):
        t = min(dur * (i + 0.5) / n, max(dur - 0.3, 0)); p = f'{outdir}/{os.path.basename(mp4)[:-4]}_{i:02d}.png'
        for _ in range(6):                       # if the seek lands past the last frame, back off and retry
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{t:.3f}', '-i', mp4, '-frames:v', '1', p], check=True)
            if os.path.exists(p): break
            t *= 0.9
        if not os.path.exists(p): raise RuntimeError(f'could not extract a frame from {mp4}')
        paths.append(p)
    return paths

def make_strip(frames, cols, out):
    ims = [PILImage.open(f).convert('RGB') for f in frames]; w, h = ims[0].size; sc = 640 / w; ims = [i.resize((640, int(h * sc))) for i in ims]
    rows = -(-len(ims) // cols); W, H = 640 * cols + 8 * (cols + 1), ims[0].size[1] * rows + 8 * (rows + 1)
    sheet = PILImage.new('RGB', (W, H), 'white')
    for k, im in enumerate(ims): sheet.paste(im, (8 + (k % cols) * (640 + 8), 8 + (k // cols) * (ims[0].size[1] + 8)))
    sheet.save(out); return out

def find_mp4(scene):
    hits = glob.glob(f'{ROOT}/manim/media/videos/ml_scenes/*/{scene}.mp4'); return hits[0] if hits else None

def fit(path, maxw, maxh):
    w, h = PILImage.open(path).size; s = min(maxw / w, maxh / h); return Image(path, width=w * s, height=h * s)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=f'{ROOT}/ML_Visual_Companion.pdf'); ap.add_argument('--frames', type=int, default=8); a = ap.parse_args()
    PW, PH = landscape(A4); M = 14 * mm; W, H = PW - 2 * M, PH - 2 * M
    H1 = ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=20, leading=24, spaceAfter=6, textColor=colors.HexColor('#1F3A6E'))
    H2 = ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=14, leading=18, spaceAfter=4, textColor=colors.HexColor('#1F3A6E'))
    B = ParagraphStyle('b', fontName='Helvetica', fontSize=10, leading=14, spaceAfter=5)
    S = ParagraphStyle('s', fontName='Helvetica-Oblique', fontSize=8.5, leading=11, textColor=colors.HexColor('#555555'), spaceAfter=4)
    C = ParagraphStyle('c', fontName='Courier', fontSize=8.5, leading=11)
    doc = BaseDocTemplate(a.out, pagesize=landscape(A4), leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
                          title='Machine Learning from Zero to Hero: Visual Companion', author='Claude')
    def footer(c, d): c.setFont('Helvetica', 8); c.setFillColor(colors.HexColor('#888888')); c.drawRightString(PW - M, 8 * mm, f'ML Visual Companion  |  page {d.page}')
    doc.addPageTemplates([PageTemplate(id='p', frames=[Frame(M, M, W, H, id='f', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=footer)])
    st = []
    st += [Paragraph('Machine Learning from Zero to Hero: Visual Companion', H1),
           Paragraph('Graphviz diagrams and Manim animations that go with the written report. Part 1 holds eight diagrams. Part 2 holds six animated scenes; a PDF cannot play video, so each animation is shown here as a filmstrip of its frames, in order, left to right and top to bottom.', B)]
    tmp = tempfile.mkdtemp(); used_real = []
    st.append(Paragraph('Contents', H2))
    for k, (_, _, t, _, sec) in enumerate(SCENES): pass
    st.append(Paragraph('Part 1: ' + '; '.join(v[0] for v in GV_NOTES.values()) + '.', B))
    st.append(Paragraph('Part 2: ' + '; '.join(s[2] for s in SCENES) + '.', B))
    st.append(PageBreak())
    st.append(Paragraph('Part 1. Graphviz diagrams', H1)); st.append(Paragraph('Sources are in graphviz/*.dot (re-render with: dot -Tpng -Gdpi=170 file.dot -o file.png). SVG versions are included for crisp scaling.', B)); st.append(PageBreak())
    for g in GV:
        key = os.path.basename(g)[:-4]; ttl, note = GV_NOTES.get(key, (key, ''))
        st += [Paragraph(ttl, H2), fit(g, W, H - 40 * mm), Spacer(1, 3), Paragraph(note, S), PageBreak()]
    st.append(Paragraph('Part 2. Manim animations as filmstrips', H1))
    st.append(Paragraph('Each page: a short description of what to watch, then the frames. Where a rendered Manim MP4 is present, the frames are real Manim output; otherwise a storyboard drawn with matplotlib from the same seeds, data and hyper-parameters is shown (labelled below each strip).', B)); st.append(PageBreak())
    for scene, pref, ttl, desc, sec in SCENES:
        mp4 = find_mp4(scene)
        if mp4:
            fr = extract_frames(mp4, a.frames, tmp); strip = make_strip(fr, 4, f'{tmp}/{scene}_strip.png'); src = f'Real Manim frames from {os.path.basename(mp4)} ({a.frames} evenly spaced frames).'; used_real.append(scene)
        else:
            strip = glob.glob(f'{SB}/{pref}_filmstrip.png')[0]; src = 'Storyboard preview (matplotlib; identical algorithm, data and numbers to the Manim scene). Render the scene with Manim to get the true animation.'
        st += [Paragraph(f'{ttl}   <font size=9 color="#777777">(report: {sec};  scene: {scene})</font>', H2), Paragraph(desc, B), fit(strip, W, H - 48 * mm), Spacer(1, 3), Paragraph(src, S), PageBreak()]
    st.append(Paragraph('How to make the animations play, and how to rebuild this PDF with the real Manim frames', H1))
    for t in ['1. pip install manim   (Python 3.9+, ffmpeg installed)',
              '2. cd manim && ./render_all.sh        # writes MP4s to manim/media/videos/ml_scenes/720p30/',
              '3. python build/build_companion_pdf.py   # re-extracts frames from those MP4s and rebuilds this PDF',
              '4. (optional) playable animation inside the PDF: python build/make_animated_tex.py  then  pdflatex animated_scenes.tex']:
        st.append(Preformatted(t, C))
    st += [Spacer(1, 8), Paragraph('About playable PDFs: PDF has no universal video support. The LaTeX animate package embeds the frames as a JavaScript slideshow that plays in Adobe Acrobat Reader and a few other viewers; browsers, Preview and most phone viewers show only the first frame, which is why the filmstrips above are the portable form. The MP4 files always play everywhere.', B)]
    doc.build(st); print('wrote', a.out, '| real Manim scenes used:', used_real or 'none (storyboards)')
if __name__ == '__main__': main()
