"""
Manim Community Edition scenes for the "Machine Learning from Zero to Hero" report.

Install:   pip install manim            (needs ffmpeg; Cairo/Pango come with the wheels on most systems)
Render:    manim -qm ml_scenes.py GradientDescentScene        (-ql = fast preview, -qh = 1080p)
All six:   ./render_all.sh

No LaTeX is required: every label uses Text(), not MathTex().
Each scene uses the same seeds and hyper-parameters as storyboards/make_storyboards.py,
so the animation and the storyboard in the PDF show the same numbers.
"""
from manim import *
import numpy as np

C_BLUE, C_GREEN, C_ORANGE, C_RED, C_GRAY = '#3B6EDC', '#2E8B57', '#D9822B', '#B03A2E', '#777777'


def title_text(s):
    return Text(s, font_size=34, weight=BOLD).to_edge(UP, buff=0.3)


def clipped_curve(axes, f, x0, x1, ymax, color, n=300):
    """Curve that is cut where |y| > ymax so wild polynomials stay inside the frame."""
    xs = np.linspace(x0, x1, n); ys = f(xs); ok = np.abs(ys) <= ymax
    group, start = VGroup(), None
    for i in range(n + 1):
        good = i < n and ok[i]
        if good and start is None:
            start = i
        if (not good) and start is not None:
            pts = [axes.c2p(xs[j], ys[j]) for j in range(start, i)]
            if len(pts) > 1:
                seg = VMobject(); seg.set_points_as_corners(pts); seg.set_stroke(color, width=5); group.add(seg)
            start = None
    return group


# ---------------------------------------------------------------- 1. gradient descent
class GradientDescentScene(Scene):
    """f(x) = x^2 / 2, gradient f'(x) = x, update x <- x - alpha * x."""
    def construct(self):
        self.add(title_text('Gradient descent'))
        axes = Axes(x_range=[-9, 9, 3], y_range=[0, 42, 10], x_length=10.5, y_length=4.6,
                    axis_config={'include_numbers': False, 'include_tip': False}).shift(DOWN * 0.7)
        curve = axes.plot(lambda x: 0.5 * x * x, color=C_BLUE, stroke_width=5)
        xlab = Text('weight x', font_size=22).next_to(axes, DOWN, buff=0.15)
        ylab = Text('loss', font_size=22).next_to(axes, LEFT, buff=0.15).shift(UP * 1.5)
        self.play(Create(axes), Create(curve), FadeIn(xlab), FadeIn(ylab))
        for alpha, steps, label_color, note in [(0.3, 12, C_GREEN, 'good learning rate: converges'),
                                                (2.2, 5, C_RED, 'learning rate too large: overshoots and diverges')]:
            self.run_phase(axes, 3.5, alpha, steps, label_color, note)
        self.wait(1.5)

    def run_phase(self, axes, x0, alpha, steps, color, note):
        pt = lambda x: axes.c2p(x, 0.5 * x * x)
        dot = Dot(pt(x0), color=C_RED, radius=0.12)
        note_t = Text(note, font_size=26, color=color).to_edge(UP, buff=1.1)
        info = Text(f'alpha = {alpha}   step 0   x = {x0:.2f}   loss = {0.5*x0*x0:.2f}', font_size=24).to_edge(DOWN, buff=0.4)
        trails = VGroup()
        self.play(FadeIn(dot), FadeIn(note_t), FadeIn(info))
        x = x0
        for k in range(1, steps + 1):
            nx = x - alpha * x
            trail = Line(pt(x), pt(nx), color=C_ORANGE, stroke_width=3); trails.add(trail)
            new_info = Text(f'alpha = {alpha}   step {k}   x = {nx:.2f}   loss = {0.5*nx*nx:.2f}', font_size=24).to_edge(DOWN, buff=0.4)
            self.play(Create(trail), dot.animate.move_to(pt(nx)), Transform(info, new_info), run_time=0.6)
            x = nx
        self.wait(1)
        self.play(FadeOut(dot), FadeOut(trails), FadeOut(note_t), FadeOut(info))


# ---------------------------------------------------------------- 2. linear regression
def lr_data():
    rng = np.random.default_rng(0); x = rng.uniform(-3, 3, 25); y = 1.8 * x + 1 + rng.normal(0, 0.8, 25); return x, y

def lr_path(alpha=0.05, n=100):
    x, y = lr_data(); w, b = -1.0, -1.0; path = [(w, b)]
    for _ in range(n):
        p = w * x + b; gw = (2 / len(x)) * np.sum((p - y) * x); gb = (2 / len(x)) * np.sum(p - y)
        w -= alpha * gw; b -= alpha * gb; path.append((w, b))
    return path

class LinearRegressionScene(Scene):
    def construct(self):
        self.add(title_text('Linear regression by gradient descent'))
        x, y = lr_data(); path = lr_path()
        axes = Axes(x_range=[-3.5, 3.5, 1], y_range=[-8, 9, 2], x_length=9, y_length=5,
                    axis_config={'include_numbers': False, 'include_tip': False}).shift(DOWN * 0.6)
        dots = VGroup(*[Dot(axes.c2p(a, b), radius=0.07, color=C_BLUE) for a, b in zip(x, y)])
        self.play(Create(axes), FadeIn(dots))
        def make(k):
            w, b = path[k]; mse = np.mean((w * x + b - y) ** 2)
            line = axes.plot(lambda t: w * t + b, x_range=[-3.5, 3.5], color=C_ORANGE, stroke_width=6)
            txt = Text(f'step {k}:   y = {w:.2f}x + {b:.2f}     MSE = {mse:.2f}', font_size=26).to_edge(DOWN, buff=0.35)
            return line, txt
        line, txt = make(0); self.play(Create(line), FadeIn(txt))
        for k in (1, 2, 3, 5, 8, 12, 20, 35, 60, 100):
            nl, nt = make(k); self.play(Transform(line, nl), Transform(txt, nt), run_time=0.7)
        self.wait(2)


# ---------------------------------------------------------------- 3. logistic regression
def lg_data():
    rng = np.random.default_rng(3)
    X0 = rng.normal([-1.5, -1.0], 0.9, (30, 2)); X1 = rng.normal([1.5, 1.0], 0.9, (30, 2))
    return np.vstack([X0, X1]), np.r_[np.zeros(30), np.ones(30)]

def lg_path(alpha=0.5, n=400):
    X, y = lg_data(); w = np.array([-1.0, 0.3]); b = 0.5; path = [(w.copy(), b)]
    for _ in range(n):
        p = 1 / (1 + np.exp(-(X @ w + b))); w = w - alpha * (X.T @ (p - y)) / len(y); b = b - alpha * np.mean(p - y)
        path.append((w.copy(), b))
    return path

class LogisticBoundaryScene(Scene):
    def construct(self):
        self.add(title_text('Logistic regression: the decision boundary learns'))
        X, y = lg_data(); path = lg_path()
        axes = Axes(x_range=[-4, 4, 1], y_range=[-4, 4, 1], x_length=6.2, y_length=6.2,
                    axis_config={'include_numbers': False, 'include_tip': False}).shift(DOWN * 0.45)
        dots = VGroup(*[Dot(axes.c2p(*p), radius=0.07, color=C_BLUE if t == 1 else C_RED) for p, t in zip(X, y)])
        self.play(Create(axes), FadeIn(dots))
        def make(k):
            w, b = path[k]; xs = np.linspace(-4, 4, 400); ys = -(w[0] * xs + b) / w[1]; m = np.abs(ys) <= 4
            p = 1 / (1 + np.exp(-(X @ w + b))); loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
            if m.sum() >= 2:
                i0, i1 = np.where(m)[0][[0, -1]]; line = Line(axes.c2p(xs[i0], ys[i0]), axes.c2p(xs[i1], ys[i1]), color=WHITE, stroke_width=6)
            else:
                line = Line(axes.c2p(0, 0), axes.c2p(0.01, 0), color=WHITE)
            txt = Text(f'step {k}     log-loss = {loss:.3f}', font_size=26).to_edge(DOWN, buff=0.3)
            return line, txt
        line, txt = make(0); self.play(Create(line), FadeIn(txt))
        for k in (1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 400):
            nl, nt = make(k); self.play(Transform(line, nl), Transform(txt, nt), run_time=0.6)
        self.wait(2)


# ---------------------------------------------------------------- 4. k-means
def km_states():
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(c, 0.9, (30, 2)) for c in ([-3, -2], [3, -1], [0, 3])])
    cent = np.array([[-1.0, -0.5], [-0.5, 0.0], [0.5, 0.5]]); states = [(cent.copy(), None, 'initial centroids')]
    for it in range(1, 12):
        lab = np.argmin(((X[:, None, :] - cent[None]) ** 2).sum(-1), axis=1); states.append((cent.copy(), lab, f'iteration {it}: assign each point to its nearest centroid'))
        new = np.array([X[lab == j].mean(0) if np.any(lab == j) else cent[j] for j in range(3)])
        states.append((new.copy(), lab, f'iteration {it}: move each centroid to the mean of its points'))
        if np.allclose(new, cent): break
        cent = new
    return X, states

class KMeansScene(Scene):
    def construct(self):
        self.add(title_text('k-means clustering'))
        X, states = km_states(); cols = [C_BLUE, C_GREEN, C_ORANGE]
        axes = Axes(x_range=[-6, 6, 1], y_range=[-5, 6, 1], x_length=9.5, y_length=5.3,
                    axis_config={'include_numbers': False, 'include_tip': False}).shift(DOWN * 0.55)
        dots = VGroup(*[Dot(axes.c2p(*p), radius=0.07, color=C_GRAY) for p in X])
        c0 = states[0][0]
        cents = VGroup(*[Square(side_length=0.28, fill_opacity=1, fill_color=cols[j], stroke_color=WHITE, stroke_width=2).move_to(axes.c2p(*c0[j])) for j in range(3)])
        txt = Text(states[0][2], font_size=26).to_edge(DOWN, buff=0.35)
        self.play(Create(axes), FadeIn(dots), FadeIn(cents), FadeIn(txt))
        prev = states[0][0]
        for cent, lab, name in states[1:]:
            new_txt = Text(name, font_size=26).to_edge(DOWN, buff=0.35)
            if np.allclose(cent, prev):      # 'assign' step: recolor points
                self.play(*[d.animate.set_color(cols[l]) for d, l in zip(dots, lab)], Transform(txt, new_txt), run_time=0.9)
            else:                            # 'move' step: slide centroids
                self.play(*[cents[j].animate.move_to(axes.c2p(*cent[j])) for j in range(3)], Transform(txt, new_txt), run_time=0.9)
            prev = cent; self.wait(0.4)
        done = Text('Converged: assignments stopped changing', font_size=28, color=C_GREEN).to_edge(DOWN, buff=0.35)
        self.play(Transform(txt, done)); self.wait(2)


# ---------------------------------------------------------------- 5. PCA
def pca_data():
    rng = np.random.default_rng(5); X = rng.multivariate_normal([0, 0], [[3, 2.2], [2.2, 2]], 60)
    mu = X.mean(0); Xc = X - mu; vals, vecs = np.linalg.eigh(np.cov(Xc.T)); o = np.argsort(vals)[::-1]
    return X, mu, Xc, vals[o], vecs[:, o]

class PCAScene(Scene):
    def construct(self):
        self.add(title_text('PCA: directions of greatest variance'))
        X, mu, Xc, vals, vecs = pca_data()
        axes = Axes(x_range=[-6, 6, 1], y_range=[-4.5, 4.5, 1], x_length=7.8, y_length=5.85,
                    axis_config={'include_numbers': False, 'include_tip': False}).shift(DOWN * 0.5)
        dots = VGroup(*[Dot(axes.c2p(*p), radius=0.07, color=C_BLUE) for p in X])
        cap = lambda s: Text(s, font_size=26).to_edge(DOWN, buff=0.3)
        c = cap('1. raw 2-D data'); self.play(Create(axes), FadeIn(dots), FadeIn(c))
        n = cap('2. center the data: subtract the mean')
        self.play(*[d.animate.move_to(axes.c2p(*p)) for d, p in zip(dots, Xc)], Transform(c, n), run_time=1.2)
        arrows = VGroup()
        for j, col in enumerate([C_ORANGE, C_GREEN]):
            v = vecs[:, j] * 2 * np.sqrt(vals[j]); arrows.add(Arrow(axes.c2p(0, 0), axes.c2p(*v), buff=0, color=col, stroke_width=7))
        n = cap('3. principal axes = eigenvectors of the covariance matrix')
        self.play(*[GrowArrow(a) for a in arrows], Transform(c, n)); self.wait(0.8)
        t = Xc @ vecs[:, 0]; P = np.outer(t, vecs[:, 0])
        links = VGroup(*[Line(axes.c2p(*a), axes.c2p(*b), color=C_GRAY, stroke_width=1.5) for a, b in zip(Xc, P)])
        n = cap('4. drop each point onto the first axis (PC1)')
        self.play(Create(links), Transform(c, n)); self.wait(0.6)
        n = cap(f'5. 2-D compressed to 1-D, keeping {vals[0]/vals.sum()*100:.1f}% of the variance')
        self.play(*[d.animate.move_to(axes.c2p(*p)).set_color(C_ORANGE) for d, p in zip(dots, P)], FadeOut(links), Transform(c, n), run_time=1.4)
        self.wait(2)


# ---------------------------------------------------------------- 6. overfitting
def of_data():
    rng = np.random.default_rng(1); x = np.sort(rng.uniform(0, 1, 12)); y = np.sin(2 * np.pi * x) + rng.normal(0, .25, 12)
    xt = rng.uniform(0, 1, 300); yt = np.sin(2 * np.pi * xt) + rng.normal(0, .25, 300); return x, y, xt, yt

class OverfittingScene(Scene):
    def construct(self):
        self.add(title_text('Underfitting, good fit, overfitting'))
        x, y, xt, yt = of_data()
        axes = Axes(x_range=[0, 1, 0.2], y_range=[-2, 2, 1], x_length=9, y_length=5,
                    axis_config={'include_numbers': False, 'include_tip': False}).shift(DOWN * 0.5)
        truth = DashedVMobject(axes.plot(lambda t: np.sin(2 * np.pi * t), color=C_GRAY, stroke_width=3), num_dashes=40)
        dots = VGroup(*[Dot(axes.c2p(a, b), radius=0.08, color=C_BLUE) for a, b in zip(x, y)])
        self.play(Create(axes), Create(truth), FadeIn(dots))
        def make(deg):
            c = np.polyfit(x, y, deg); f = lambda t: np.polyval(c, t)
            tr = np.mean((f(x) - y) ** 2); te = np.mean((f(xt) - yt) ** 2)
            word = {1: 'too simple (underfit)', 3: 'about right', 5: 'about right', 9: 'too flexible (overfit)'}.get(deg, '')
            txt = Text(f'degree {deg}: {word}     train MSE {tr:.3f}     test MSE {te:.3f}', font_size=24).to_edge(DOWN, buff=0.35)
            return clipped_curve(axes, f, 0, 1, 2.0, C_ORANGE), txt
        curve, txt = make(1); self.play(Create(curve), FadeIn(txt))
        for deg in (3, 5, 9):
            nc, nt = make(deg); self.play(ReplacementTransform(curve, nc), Transform(txt, nt), run_time=1.2); curve = nc; self.wait(1)
        self.wait(2)
