"""Storyboards: static renders of the exact algorithm states shown in the Manim scenes.
Same seeds, same data, same hyperparameters as manim/ml_scenes.py."""
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, GREEN, ORANGE, RED, GRAY, PURPLE = '#3B6EDC', '#2E8B57', '#D9822B', '#B03A2E', '#777777', '#7A4FC2'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

SCENES = {}
def scene(key, title, ncols):
    def deco(fn):
        SCENES[key] = dict(title=title, ncols=ncols, panels=fn()); return fn
    return deco

# ---------- 1. Gradient descent (converge vs diverge) ----------
def gd_path(x0, alpha, n):
    xs = [x0]
    for _ in range(n): xs.append(xs[-1] - alpha * xs[-1])      # f(x)=x^2/2, f'(x)=x
    return xs
def gd_panel(xs, k, alpha):
    def draw(ax):
        x = np.linspace(-9, 9, 400); ax.plot(x, 0.5 * x**2, color=BLUE, lw=1.6)
        pts = xs[:k + 1]; ax.plot(pts, [0.5 * p * p for p in pts], '-o', color=ORANGE, ms=3.5, lw=1, alpha=.85)
        ax.scatter([xs[k]], [0.5 * xs[k]**2], s=55, color=RED, zorder=5)
        ax.set_xlim(-9, 9); ax.set_ylim(-2, 42)
        ax.set_title(f'alpha={alpha}  step {k}\nx={xs[k]:.2f}   loss={0.5*xs[k]**2:.2f}', fontsize=8)
    return draw
@scene('GradientDescentScene', 'Gradient descent: a good learning rate converges, a too-large one diverges', 6)
def _():
    a = gd_path(3.5, 0.3, 15); b = gd_path(3.5, 2.2, 5)
    return [gd_panel(a, k, 0.3) for k in (0, 1, 2, 4, 8, 15)] + [gd_panel(b, k, 2.2) for k in range(6)]

# ---------- 2. Linear regression fit ----------
def lr_data():
    rng = np.random.default_rng(0); x = rng.uniform(-3, 3, 25); y = 1.8 * x + 1 + rng.normal(0, 0.8, 25); return x, y
def lr_path(alpha=0.05, n=100):
    x, y = lr_data(); w, b = -1.0, -1.0; path = [(w, b)]
    for _ in range(n):
        p = w * x + b; gw = (2 / len(x)) * np.sum((p - y) * x); gb = (2 / len(x)) * np.sum(p - y)
        w -= alpha * gw; b -= alpha * gb; path.append((w, b))
    return path
def lr_panel(k):
    def draw(ax):
        x, y = lr_data(); w, b = lr_path()[k]; mse = np.mean((w * x + b - y)**2)
        ax.scatter(x, y, s=14, color=BLUE); xx = np.array([-3.5, 3.5]); ax.plot(xx, w * xx + b, color=ORANGE, lw=2)
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-8, 9); ax.set_title(f'step {k}:  y = {w:.2f}x + {b:.2f}\nMSE = {mse:.2f}', fontsize=8)
    return draw
@scene('LinearRegressionScene', 'Linear regression: the line is pulled onto the data by gradient descent', 3)
def _(): return [lr_panel(k) for k in (0, 2, 5, 10, 25, 100)]

# ---------- 3. Logistic regression boundary ----------
def lg_data():
    rng = np.random.default_rng(3)
    X0 = rng.normal([-1.5, -1.0], 0.9, (30, 2)); X1 = rng.normal([1.5, 1.0], 0.9, (30, 2))
    return np.vstack([X0, X1]), np.r_[np.zeros(30), np.ones(30)]
def lg_path(alpha=0.5, n=400):
    X, y = lg_data(); w = np.array([-1.0, 0.3]); b = 0.5; path = [(w.copy(), b)]
    for _ in range(n):
        p = 1 / (1 + np.exp(-(X @ w + b))); w = w - alpha * (X.T @ (p - y)) / len(y); b = b - alpha * np.mean(p - y); path.append((w.copy(), b))
    return path
def lg_panel(k):
    def draw(ax):
        X, y = lg_data(); w, b = lg_path()[k]
        p = 1 / (1 + np.exp(-(X @ w + b))); loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        gx, gy = np.meshgrid(np.linspace(-4, 4, 80), np.linspace(-4, 4, 80)); z = 1 / (1 + np.exp(-(w[0] * gx + w[1] * gy + b)))
        ax.contourf(gx, gy, z, levels=[0, .5, 1], colors=['#FADBD8', '#DCE9FF'], alpha=.8)
        ax.scatter(*X[y == 0].T, s=14, color=RED); ax.scatter(*X[y == 1].T, s=14, color=BLUE)
        xx = np.array([-4, 4]); ax.plot(xx, -(w[0] * xx + b) / w[1], color='black', lw=1.8)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_title(f'step {k}   log-loss = {loss:.3f}', fontsize=8)
    return draw
@scene('LogisticBoundaryScene', 'Logistic regression: the decision boundary rotates and shifts to separate the classes', 3)
def _(): return [lg_panel(k) for k in (0, 3, 10, 30, 100, 400)]

# ---------- 4. k-means ----------
def km_states():
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(c, 0.9, (30, 2)) for c in ([-3, -2], [3, -1], [0, 3])])
    cent = np.array([[-1.0, -0.5], [-0.5, 0.0], [0.5, 0.5]]); states = [(cent.copy(), None, 'initial centroids')]
    for it in range(1, 12):
        lab = np.argmin(((X[:, None, :] - cent[None])**2).sum(-1), axis=1); states.append((cent.copy(), lab, f'iteration {it}: assign'))
        new = np.array([X[lab == j].mean(0) if np.any(lab == j) else cent[j] for j in range(3)])
        states.append((new.copy(), lab, f'iteration {it}: move centroids'))
        if np.allclose(new, cent): break
        cent = new
    return X, states
def km_panel(k):
    def draw(ax):
        X, st = km_states(); cent, lab, name = st[min(k, len(st) - 1)] if k >= 0 else st[-1]
        cols = np.array([BLUE, GREEN, ORANGE]); ax.scatter(*X.T, s=14, color=GRAY if lab is None else cols[lab])
        ax.scatter(*cent.T, s=110, marker='X', c=cols, edgecolors='black', linewidths=1.2, zorder=5)
        ax.set_xlim(-6, 6); ax.set_ylim(-5, 6); ax.set_title(name, fontsize=8)
    return draw
@scene('KMeansScene', 'k-means: assign points to the nearest centroid, move centroids to the means, repeat', 3)
def _(): return [km_panel(k) for k in (0, 1, 2, 3, 4, -1)]

# ---------- 5. PCA ----------
def pca_data():
    rng = np.random.default_rng(5); X = rng.multivariate_normal([0, 0], [[3, 2.2], [2.2, 2]], 60)
    mu = X.mean(0); Xc = X - mu; vals, vecs = np.linalg.eigh(np.cov(Xc.T)); o = np.argsort(vals)[::-1]; return X, mu, Xc, vals[o], vecs[:, o]
def pca_panel(stage):
    def draw(ax):
        X, mu, Xc, vals, vecs = pca_data(); ax.set_aspect('equal'); ax.set_xlim(-6, 6); ax.set_ylim(-5, 5)
        if stage == 0: ax.scatter(*X.T, s=14, color=BLUE); ax.set_title('1. raw 2-D data', fontsize=8)
        if stage == 1: ax.scatter(*Xc.T, s=14, color=BLUE); ax.scatter(0, 0, s=60, color=RED, marker='+', zorder=5); ax.set_title('2. center the data (subtract the mean)', fontsize=8)
        if stage >= 2:
            ax.scatter(*Xc.T, s=14, color=BLUE, alpha=.5 if stage >= 4 else 1)
            for j, c in enumerate([ORANGE, GREEN]):
                v = vecs[:, j] * 2 * np.sqrt(vals[j]); ax.annotate('', xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=c, lw=2.2))
        if stage == 2: ax.set_title('3. eigenvectors of the covariance = principal axes', fontsize=8)
        if stage >= 3:
            t = Xc @ vecs[:, 0]; P = np.outer(t, vecs[:, 0])
            if stage == 3:
                for a, b in zip(Xc, P): ax.plot([a[0], b[0]], [a[1], b[1]], color=GRAY, lw=.6)
                ax.set_title('4. drop each point onto PC1', fontsize=8)
            if stage >= 4: ax.scatter(*P.T, s=16, color=ORANGE, zorder=4)
        if stage == 4: ax.set_title('5. result: 2-D data compressed to 1-D', fontsize=8)
        if stage == 5:
            ax.clear(); r = vals / vals.sum(); ax.bar(['PC1', 'PC2'], r, color=[ORANGE, GREEN]); ax.set_ylim(0, 1)
            for i, v in enumerate(r): ax.text(i, v + .02, f'{v*100:.1f}%', ha='center', fontsize=9)
            ax.set_title('6. explained variance ratio', fontsize=8)
    return draw
@scene('PCAScene', 'PCA: find the directions of greatest variance and project onto them', 3)
def _(): return [pca_panel(s) for s in range(6)]

# ---------- 6. Overfitting ----------
def of_data():
    rng = np.random.default_rng(1); x = np.sort(rng.uniform(0, 1, 12)); y = np.sin(2 * np.pi * x) + rng.normal(0, .25, 12)
    xt = rng.uniform(0, 1, 300); yt = np.sin(2 * np.pi * xt) + rng.normal(0, .25, 300); return x, y, xt, yt
def of_panel(deg):
    def draw(ax):
        x, y, xt, yt = of_data(); c = np.polyfit(x, y, deg); xx = np.linspace(0, 1, 400); yy = np.polyval(c, xx)
        tr = np.mean((np.polyval(c, x) - y)**2); te = np.mean((np.polyval(c, xt) - yt)**2)
        ax.plot(xx, np.sin(2 * np.pi * xx), color=GRAY, ls='--', lw=1, label='truth'); yy = np.where(np.abs(yy) < 2.2, yy, np.nan)
        ax.plot(xx, yy, color=ORANGE, lw=2); ax.scatter(x, y, s=18, color=BLUE, zorder=5)
        ax.set_xlim(0, 1); ax.set_ylim(-2, 2); ax.set_title(f'degree {deg}\ntrain MSE {tr:.3f}   test MSE {te:.3f}', fontsize=8)
    return draw
@scene('OverfittingScene', 'Under- and overfitting: more flexibility lowers train error but eventually raises test error', 3)
def _(): return [of_panel(d) for d in (1, 2, 3, 5, 7, 9)]

def render_all():
    for i, (key, s) in enumerate(SCENES.items(), 1):
        n = len(s['panels']); nc = s['ncols']; nr = int(np.ceil(n / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(3.3 * nc, 3.0 * nr + .4)); axs = np.atleast_1d(axs).ravel()
        for ax, p in zip(axs, s['panels']): p(ax)
        for ax in axs[n:]: ax.axis('off')
        fig.suptitle(s['title'], fontsize=11, fontweight='bold'); fig.tight_layout()
        fig.savefig(f'{HERE}/{i}_{key}_filmstrip.png', dpi=150, bbox_inches='tight'); plt.close(fig)
        fd = f'{HERE}/frames/{key}'; os.makedirs(fd, exist_ok=True)
        for j, p in enumerate(s['panels']):
            fig, ax = plt.subplots(figsize=(6.4, 4.8)); p(ax); fig.tight_layout(); fig.savefig(f'{fd}/f{j:03d}.png', dpi=100); plt.close(fig)
        print('rendered', key, n, 'panels')
if __name__ == '__main__':
    render_all()
