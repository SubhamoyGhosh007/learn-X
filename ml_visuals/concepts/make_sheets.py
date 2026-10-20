"""Compose one 'concept sheet' per topic: title + Graphviz diagram (top) + animation frames (bottom)."""
import os, glob, sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
HERE = os.path.dirname(os.path.abspath(__file__)); W = 1800
sys.path.insert(0, HERE)
import make_anims as MA
FONT_B = fm.findfont('DejaVu Sans:bold'); FONT = fm.findfont('DejaVu Sans')
DIAGRAM_ONLY = {'A01_linear_regression', 'A02_logistic_regression', 'A11_kmeans', 'A15_pca'}   # animation already embedded earlier in the report
ANIM_ONLY = {'S12_cross_validation'}                                                       # diagram already embedded earlier
def fit_w(im, w, maxh=None):
    s = w / im.width
    if maxh and im.height * s > maxh: s = maxh / im.height
    return im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
def sheet(key):
    title = MA.ANIMS[key]['title']; parts = []
    if key not in ANIM_ONLY: parts.append(('Diagram', fit_w(Image.open(f'{HERE}/diagrams/{key}.png').convert('RGB'), W - 40, 1000)))
    if key not in DIAGRAM_ONLY: parts.append(('Animation, shown as frames in order (left to right, top to bottom)', fit_w(Image.open(f'{HERE}/filmstrips/{key}.png').convert('RGB'), W - 40)))
    head = 110; label_h = 54; H = head + sum(label_h + im.height + 24 for _, im in parts) + 10
    out = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(out); d.rectangle([0, 0, W, head - 20], fill='#1F3A6E')
    d.text((24, 22), key.split('_', 1)[1].replace('_', ' ').title() if False else title, font=ImageFont.truetype(FONT_B, 34), fill='white')
    y = head
    for lab, im in parts:
        d.text((24, y + 8), lab, font=ImageFont.truetype(FONT_B, 26), fill='#1F3A6E'); y += label_h
        out.paste(im, ((W - im.width) // 2, y)); y += im.height + 24
    out.save(f'{HERE}/sheets/{key}.png', optimize=True); return out.size
if __name__ == '__main__':
    for key in MA.ANIMS: print(key, sheet(key))
