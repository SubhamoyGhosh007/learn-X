"""Real animations (GIF) + filmstrips for every math idea and algorithm. Every frame is computed from real data / the real algorithm.
Usage: python make_anims.py [key ...]     (no args = all)"""
import os, sys, functools, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'storyboards'))
import make_storyboards as sb          # reuse the exact panels of the six original scenes

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, GREEN, ORANGE, RED, GRAY, PURPLE = '#3B6EDC', '#2E8B57', '#D9822B', '#B03A2E', '#777777', '#7A4FC2'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9.5})
ANIMS = {}
def anim(key, title, n, naxes=1, ms=120, last_ms=None):
    def deco(fn): ANIMS[key] = dict(title=title, n=n, naxes=naxes, ms=ms, last_ms=last_ms or ms * 6, fn=fn); return fn
    return deco
def arrow(ax, p, color, lw=2.8, q=(0, 0)): ax.annotate('', xy=p, xytext=q, arrowprops=dict(arrowstyle='->', color=color, lw=lw))

# ============================================================ MATHEMATICS
@anim('S01_dot_product', 'Dot product: how much two vectors point the same way', 48, 1, 90)
def _(axs, k, n):
    ax = axs[0]; a = np.array([3.0, 1.0]); th = 2 * np.pi * k / n; b = 2.5 * np.array([np.cos(th), np.sin(th)])
    arrow(ax, a, BLUE); arrow(ax, b, ORANGE); pr = (a @ b) / (a @ a) * a
    ax.plot([b[0], pr[0]], [b[1], pr[1]], '--', color=GRAY, lw=1); ax.plot(*pr, 'o', color=GREEN, ms=7)
    ax.text(*(a * 1.08), 'a', color=BLUE, fontsize=12); ax.text(*(b * 1.12), 'b', color=ORANGE, fontsize=12)
    c = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)); word = 'point the same way' if c > .3 else ('opposite ways' if c < -.3 else 'nearly perpendicular')
    ax.axhline(0, color='#ddd'); ax.axvline(0, color='#ddd'); ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect('equal')
    ax.set_title(f'a = (3, 1)   b rotates   a.b = {a@b:+.2f}   cos(angle) = {c:+.2f}\n{word}  (green dot = projection of b on a)')

@anim('S02_matrix_eigen', 'A matrix transforms space; eigenvectors keep their direction', 30, 1, 100)
def _(axs, k, n):
    ax = axs[0]; t = k / (n - 1); A = np.array([[2., 1.], [1., 2.]]); M = (1 - t) * np.eye(2) + t * A
    th = np.linspace(0, 2 * np.pi, 200); C = np.vstack([np.cos(th), np.sin(th)]); ax.plot(*C, color='#cccccc', lw=1); ax.plot(*(M @ C), color=BLUE, lw=2)
    for v, c in [((1, 1), GREEN), ((1, -1), ORANGE)]:
        v = np.array(v) / np.sqrt(2); ax.plot([-4 * v[0], 4 * v[0]], [-4 * v[1], 4 * v[1]], ':', color=c, lw=1); arrow(ax, M @ v, c)
    for e in ((1, 0), (0, 1)): arrow(ax, M @ np.array(e), GRAY, 1.5)
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6); ax.set_aspect('equal')
    ax.set_title(f'A = [[2,1],[1,2]] applied {t:.0%}\ncircle -> ellipse; green/orange arrows (eigenvectors) only stretch, by 3 and 1')

@anim('S03_derivative', 'The derivative is the slope of a function at a point', 40, 2, 100)
def _(axs, k, n):
    f = lambda x: np.sin(x) + 0.15 * x**2; df = lambda x: np.cos(x) + 0.3 * x; x0 = -4 + 8 * k / (n - 1); xs = np.linspace(-4, 4, 300); s = df(x0)
    a = axs[0]; a.plot(xs, f(xs), color=BLUE, lw=2); xt = np.array([x0 - 1.3, x0 + 1.3]); a.plot(xt, f(x0) + s * (xt - x0), color=ORANGE, lw=2.5); a.plot(x0, f(x0), 'o', color=RED, ms=8)
    a.set_ylim(-1.8, 3.4); a.set_title(f"f(x) and its tangent line    slope f'(x) = {s:+.2f}")
    b = axs[1]; b.plot(xs, df(xs), color=GREEN, lw=2); b.axhline(0, color='#bbb'); b.plot(x0, s, 'o', color=RED if abs(s) > .12 else PURPLE, ms=8); b.set_ylim(-2, 2)
    b.set_title("f'(x): zero crossings are the minima / maxima of f" + ('   <- flat here' if abs(s) < .12 else ''))

@anim('S04_gradient', 'Gradient: the arrow of steepest ascent; descent walks the opposite way', 31, 1, 150)
def _(axs, k, n):
    ax = axs[0]; w = np.array([-3., 2.]); path = [w.copy()]
    for _ in range(30): w = w - 0.12 * np.array([w[0], 6 * w[1]]); path.append(w.copy())
    path = np.array(path); L = lambda a, b: .5 * (a**2 + 6 * b**2); g1, g2 = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-3, 3, 200))
    ax.contourf(g1, g2, L(g1, g2), levels=14, cmap='Blues_r', alpha=.85); ax.contour(g1, g2, L(g1, g2), levels=14, colors='white', linewidths=.4)
    ax.plot(*path[:k + 1].T, '-o', color=ORANGE, ms=3.5, lw=1.4); p = path[k]; gr = np.array([p[0], 6 * p[1]])
    arrow(ax, p - 0.35 * gr, GREEN, 2.5, q=p); ax.plot(*p, 'o', color=RED, ms=8)
    ax.set_xlim(-4, 4); ax.set_ylim(-3, 3); ax.set_xlabel('w1'); ax.set_ylabel('w2'); ax.set_title(f'step {k}   loss = {L(*p):.3f}   green arrow = -gradient (downhill)')

@anim('S05_chain_rule', 'Chain rule: slopes multiply along the chain  x -> u -> y', 40, 2, 100)
def _(axs, k, n):
    x0 = -3 + 6 * k / (n - 1); u0 = 1 + x0**2; xs = np.linspace(-3.2, 3.2, 200); us = np.linspace(0.8, 10.5, 200)
    gs, fs = 2 * x0, 1 / u0; a = axs[0]; a.plot(xs, 1 + xs**2, color=BLUE, lw=2); xt = np.array([x0 - .8, x0 + .8]); a.plot(xt, u0 + gs * (xt - x0), color=ORANGE, lw=2.5); a.plot(x0, u0, 'o', color=RED, ms=8)
    a.set_ylim(0, 11); a.set_xlim(-3.4, 3.4); a.set_xlabel('x'); a.set_title(f"u = g(x) = 1 + x^2     du/dx = {gs:+.2f}")
    b = axs[1]; b.plot(us, np.log(us), color=GREEN, lw=2); ut = np.array([u0 - 1.2, u0 + 1.2]); b.plot(ut, np.log(u0) + fs * (ut - u0), color=ORANGE, lw=2.5); b.plot(u0, np.log(u0), 'o', color=RED, ms=8)
    b.set_xlim(0, 11); b.set_ylim(-.4, 2.6); b.set_xlim(0, 11); b.set_ylim(-.4, 2.6); b.set_xlabel('u'); b.set_title(f"y = f(u) = ln u     dy/du = {fs:.2f}")
    e = 1e-5; num = (np.log(1 + (x0 + e)**2) - np.log(1 + x0**2)) / e
    return f'dy/dx = (dy/du)(du/dx) = {fs:.2f} x {gs:+.2f} = {fs*gs:+.3f}      numeric check: {num:+.3f}'

@anim('S06_bayes', "Bayes' rule: the same positive test means very different things at different priors", 40, 2, 130)
def _(axs, k, n):
    pri = np.logspace(-3, np.log10(.5), n); sens, fpr, N = .99, .05, 10000; post = lambda p: sens * p / (sens * p + fpr * (1 - p)); p = pri[k]
    TP, FP = sens * p * N, fpr * (1 - p) * N; a = axs[0]; a.bar(['true positives\n(sick, test +)', 'false positives\n(healthy, test +)'], [TP, FP], color=[GREEN, RED]); a.set_ylim(0, 5200)
    for i, v in enumerate((TP, FP)): a.text(i, v + 80, f'{v:.0f}', ha='center')
    a.set_title(f'Of {N:,} people tested, prior (prevalence) = {p*100:.2f}%')
    b = axs[1]; b.semilogx(pri, [post(q) for q in pri], color=BLUE, lw=2); b.plot(p, post(p), 'o', color=RED, ms=9); b.set_ylim(0, 1.02); b.set_xlabel('prior P(sick)'); b.set_ylabel('P(sick | positive)')
    b.set_title(f'posterior = TP / (TP + FP) = {post(p):.1%}')

@anim('S07_normal_clt', 'Central limit theorem: averages of anything become bell-shaped', 10, 1, 700)
def _(axs, k, n):
    ns = [1, 2, 3, 4, 6, 8, 12, 20, 30, 50]; m = ns[k]; rng = np.random.default_rng(0); means = rng.uniform(0, 1, (6000, m)).mean(1); ax = axs[0]
    ax.hist(means, bins=40, density=True, color=BLUE, alpha=.6); xs = np.linspace(0, 1, 300); sd = np.sqrt(1 / 12 / m)
    ax.plot(xs, np.exp(-(xs - .5)**2 / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi)), color=ORANGE, lw=2.5); ax.set_xlim(0, 1); ax.set_ylim(0, 11)
    ax.set_title(f'histogram of 6000 averages of {m} uniform draw(s)   (orange = normal curve)')

@anim('S08_covariance', 'Correlation: from perfectly linked to unrelated to opposite', 40, 1, 100)
def _(axs, k, n):
    rng = np.random.default_rng(2); z1, z2 = rng.normal(size=80), rng.normal(size=80); r = 0.95 * np.cos(np.pi * k / (n - 1)); x = z1; y = r * z1 + np.sqrt(1 - r * r) * z2; ax = axs[0]
    ax.scatter(x, y, s=18, color=BLUE); c = np.polyfit(x, y, 1); xx = np.array([-3, 3]); ax.plot(xx, np.polyval(c, xx), color=ORANGE, lw=2); ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5); ax.set_aspect('equal')
    ax.set_title(f'correlation r = {np.corrcoef(x, y)[0,1]:+.2f}    covariance = {np.cov(x, y)[0,1]:+.2f}')

@anim('S09_mle', 'Maximum likelihood: 7 heads in 10 flips -> the likelihood peaks at p = 0.7', 40, 2, 100)
def _(axs, k, n):
    ps = np.linspace(.01, .99, 200); L = ps**7 * (1 - ps)**3; p = .05 + .9 * k / (n - 1); l = p**7 * (1 - p)**3; a = axs[0]
    a.plot(ps, L, color=BLUE, lw=2); a.plot(p, l, 'o', color=RED, ms=9); a.axvline(.7, color='#bbb', ls=':'); a.set_xlabel('p (probability of heads)'); a.set_title(f'likelihood  p^7 (1-p)^3 = {l:.5f}')
    b = axs[1]; b.plot(ps, np.log(L), color=GREEN, lw=2); b.plot(p, np.log(l), 'o', color=RED, ms=9); b.axvline(.7, color='#bbb', ls=':'); b.set_ylim(-14, -5); b.set_xlabel('p'); b.set_title(f'log-likelihood = {np.log(l):.2f}   (same peak, easier maths)')

@anim('S10_bias_variance', 'Bias-variance: flexibility lowers training error, then test error turns upward', 9, 2, 900)
def _(axs, k, n):
    x, y, xt, yt = sb.of_data(); degs = list(range(1, 10)); d = degs[k]
    tr = [np.mean((np.polyval(np.polyfit(x, y, g), x) - y)**2) for g in degs]; te = [np.mean((np.polyval(np.polyfit(x, y, g), xt) - yt)**2) for g in degs]
    a = axs[0]; xx = np.linspace(0, 1, 400); yy = np.polyval(np.polyfit(x, y, d), xx); a.plot(xx, np.sin(2 * np.pi * xx), '--', color=GRAY, lw=1); a.plot(xx, np.where(np.abs(yy) < 2.2, yy, np.nan), color=ORANGE, lw=2)
    a.scatter(x, y, s=18, color=BLUE, zorder=5); a.set_ylim(-2, 2); a.set_title(f'polynomial degree {d}')
    b = axs[1]; b.plot(degs, tr, '-o', color=BLUE, label='train error'); b.plot(degs, te, '-o', color=RED, label='test error'); b.plot(d, te[k], 's', color='black', ms=10, mfc='none'); b.set_yscale('log'); b.legend(); b.set_xlabel('model flexibility (degree)')
    b.set_title('left side: high bias  |  right side: high variance')

@anim('S11_scaling', 'Why feature scaling speeds up gradient descent', 31, 2, 140)
def _(axs, k, n):
    def run(h, a):
        w = np.array([-3., 1.]); P = [w.copy()]
        for _ in range(30): w = w - a * h * w; P.append(w.copy())
        return np.array(P)
    for ax, h, a, name in [(axs[0], np.array([1., 50.]), 0.035, 'raw features (very different scales)'), (axs[1], np.array([1., 1.]), 0.5, 'standardized features')]:
        P = run(h, a); L = lambda u, v: .5 * (h[0] * u**2 + h[1] * v**2); g1, g2 = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-1.2, 1.2, 200))
        ax.contourf(g1, g2, L(g1, g2), levels=14, cmap='Blues_r', alpha=.85); ax.plot(*P[:k + 1].T, '-o', color=ORANGE, ms=3, lw=1.2); ax.plot(*P[k], 'o', color=RED, ms=8)
        ax.set_xlim(-4, 4); ax.set_ylim(-1.2, 1.2); ax.set_title(f'{name}\nstep {k}   loss = {L(*P[k]):.4f}')

@anim('S12_cross_validation', '5-fold cross-validation, one fold at a time', 6, 1, 1100)
def _(axs, k, n):
    ax = axs[0]; sc = [.91, .88, .93, .90, .89]; ax.set_xlim(0, 5); ax.set_ylim(-.2, 5.2); ax.axis('off')
    for f in range(5):
        for i in range(5):
            val = (i == f and k < 5); ax.add_patch(plt.Rectangle((i, 4 - f), 1, .8, color='#FFF3C4' if val else '#DCE9FF', ec='#666666')); ax.text(i + .5, 4.4 - f, 'validate' if val else 'train', ha='center', va='center', fontsize=8)
        if f < min(k + 1, 5): ax.text(5.05, 4.4 - f, f'score {sc[f]:.2f}', va='center', fontsize=9, color=GREEN)
    ax.set_xlim(0, 6.2); ax.set_title(f'fold {min(k+1,5)} of 5' if k < 5 else f'CV estimate = mean of 5 scores = {np.mean(sc):.3f}  (+/- {np.std(sc):.3f})')

# ============================================================ ALGORITHMS
@anim('A01_linear_regression', 'Linear regression pulled onto the data by gradient descent', 16, 1, 220)
def _(axs, k, n): sb.lr_panel([0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 30, 40, 60, 100][k])(axs[0])
@anim('A02_logistic_regression', 'Logistic regression decision boundary learning', 15, 1, 220)
def _(axs, k, n): sb.lg_panel([0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 400][k])(axs[0])
@anim('A11_kmeans', 'k-means: assign, move, repeat', len(sb.km_states()[1]), 1, 900)
def _(axs, k, n): sb.km_panel(k)(axs[0])
@anim('A15_pca', 'PCA step by step', 6, 1, 1200)
def _(axs, k, n): sb.pca_panel(k)(axs[0])

@functools.lru_cache(None)
def _reg_paths(n):
    from sklearn.linear_model import Ridge, Lasso
    rng = np.random.default_rng(0); X = rng.normal(size=(100, 6)); wt = np.array([3, -2, 0, 0, 1.5, 0.]); y = X @ wt + rng.normal(0, 1, 100)
    ra = np.logspace(-1, 3.5, n); la = np.logspace(-3, .7, n)
    return ra, la, np.array([Ridge(alpha=a).fit(X, y).coef_ for a in ra]), np.array([Lasso(alpha=a, max_iter=5000).fit(X, y).coef_ for a in la])
@anim('A03_regularization', 'Ridge shrinks coefficients smoothly; lasso drives some to exactly zero', 40, 2, 110)
def _(axs, k, n):
    ra, la, cr, cl = _reg_paths(n)
    for ax, al, C, nm in [(axs[0], ra, cr, 'Ridge (L2)'), (axs[1], la, cl, 'Lasso (L1)')]:
        for j in range(6): ax.semilogx(al, C[:, j], color=plt.cm.tab10(j), lw=1.8)
        ax.axvline(al[k], color='black', lw=1.5); ax.set_ylim(-3.5, 3.5); ax.set_xlabel('lambda (regularization strength)')
        ax.set_title(f'{nm}: {int(np.sum(np.abs(C[k]) < 1e-6))} of 6 weights exactly zero' if nm.startswith('Lasso') else f'{nm}: all weights shrink, none exactly zero')

@anim('A04_knn', 'k-NN: small k = jagged boundary (overfit), large k = smooth (underfit)', 6, 1, 1100)
def _(axs, k, n):
    from sklearn.datasets import make_moons; from sklearn.neighbors import KNeighborsClassifier
    X, y = make_moons(120, noise=.3, random_state=0); kk = [1, 3, 5, 9, 15, 31][k]; clf = KNeighborsClassifier(kk).fit(X, y); ax = axs[0]
    g1, g2 = np.meshgrid(np.linspace(-2, 3, 220), np.linspace(-1.5, 2, 220)); Z = clf.predict(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)
    ax.contourf(g1, g2, Z, levels=[-.5, .5, 1.5], colors=['#FADBD8', '#DCE9FF']); ax.scatter(*X[y == 0].T, s=16, color=RED); ax.scatter(*X[y == 1].T, s=16, color=BLUE)
    ax.set_title(f'k = {kk}    training accuracy = {clf.score(X, y):.2f}')

@anim('A05_naive_bayes', 'Naive Bayes: posterior = prior x likelihood, then normalise', 40, 2, 100)
def _(axs, k, n):
    mu, sd = [-1, 1.8], [1, 1.3]; pdf = lambda x, m, s: np.exp(-(x - m)**2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi)); xs = np.linspace(-5, 6, 400); x0 = -4 + 9 * k / (n - 1); cols = [BLUE, ORANGE]
    a = axs[0]
    for c in range(2): a.plot(xs, .5 * pdf(xs, mu[c], sd[c]), color=cols[c], lw=2, label=f'class {c}: prior x likelihood'); a.plot(x0, .5 * pdf(x0, mu[c], sd[c]), 'o', color=cols[c], ms=8)
    a.axvline(x0, color=GRAY, ls=':'); a.legend(loc='upper right', fontsize=7); a.set_xlabel('feature value x'); a.set_title(f'query x = {x0:+.2f}')
    s = [.5 * pdf(x0, mu[c], sd[c]) for c in range(2)]; p = np.array(s) / sum(s); b = axs[1]; b.bar(['class 0', 'class 1'], p, color=cols); b.set_ylim(0, 1.05)
    for i, v in enumerate(p): b.text(i, v + .02, f'{v:.2f}', ha='center')
    b.set_title('posterior probabilities (they sum to 1)')

@anim('A06_decision_tree', 'Decision tree: deeper = more splits = a more fragmented boundary', 6, 1, 1100)
def _(axs, k, n):
    from sklearn.datasets import make_moons; from sklearn.tree import DecisionTreeClassifier
    X, y = make_moons(300, noise=.3, random_state=1); Xtr, ytr, Xte, yte = X[:150], y[:150], X[150:], y[150:]; d = [1, 2, 3, 4, 6, 12][k]; clf = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr); ax = axs[0]
    g1, g2 = np.meshgrid(np.linspace(-2, 3, 260), np.linspace(-1.5, 2, 260)); Z = clf.predict(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)
    ax.contourf(g1, g2, Z, levels=[-.5, .5, 1.5], colors=['#FADBD8', '#DCE9FF']); ax.scatter(*Xtr[ytr == 0].T, s=14, color=RED); ax.scatter(*Xtr[ytr == 1].T, s=14, color=BLUE)
    ax.set_title(f'max depth {d}   leaves = {clf.get_n_leaves()}   train acc {clf.score(Xtr, ytr):.2f}   test acc {clf.score(Xte, yte):.2f}')

@anim('A07_random_forest', 'Random forest: more trees average out the noise', 6, 1, 1100)
def _(axs, k, n):
    from sklearn.datasets import make_moons; from sklearn.ensemble import RandomForestClassifier
    X, y = make_moons(300, noise=.35, random_state=1); Xtr, ytr, Xte, yte = X[:150], y[:150], X[150:], y[150:]; m = [1, 3, 10, 30, 100, 300][k]; clf = RandomForestClassifier(m, random_state=0, min_samples_leaf=1).fit(Xtr, ytr); ax = axs[0]
    g1, g2 = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200)); Z = clf.predict_proba(np.c_[g1.ravel(), g2.ravel()])[:, 1].reshape(g1.shape)
    ax.contourf(g1, g2, Z, levels=np.linspace(0, 1, 11), cmap='RdBu', alpha=.85); ax.scatter(*Xtr[ytr == 0].T, s=14, color=RED, ec='white', lw=.4); ax.scatter(*Xtr[ytr == 1].T, s=14, color=BLUE, ec='white', lw=.4)
    ax.set_title(f'{m} tree(s)   test accuracy {clf.score(Xte, yte):.2f}   (colour = predicted probability)')

@functools.lru_cache(None)
def _gb_fit():
    from sklearn.ensemble import GradientBoostingRegressor
    rng = np.random.default_rng(0); x = rng.uniform(0, 1, 80); y = np.sin(2 * np.pi * x) + rng.normal(0, .25, 80); xt = rng.uniform(0, 1, 300); yt = np.sin(2 * np.pi * xt) + rng.normal(0, .25, 300)
    return x, y, xt, yt, GradientBoostingRegressor(n_estimators=100, learning_rate=.1, max_depth=2, random_state=0).fit(x[:, None], y)
@anim('A08_gradient_boosting', 'Gradient boosting: each stage adds a small tree that fixes the leftover error', 12, 2, 700)
def _(axs, k, n):
    x, y, xt, yt, g = _gb_fit()
    st = [0, 1, 2, 3, 5, 8, 12, 20, 40, 60, 80, 100]; m = st[k]; xx = np.linspace(0, 1, 300)
    sp = list(g.staged_predict(xx[:, None])); yy = np.full_like(xx, y.mean()) if m == 0 else sp[m - 1]; a = axs[0]; a.scatter(x, y, s=14, color=BLUE); a.plot(xx, yy, color=ORANGE, lw=2.5); a.set_ylim(-1.8, 1.8); a.set_title(f'after {m} tree(s)')
    rm = lambda X_, y_: [np.sqrt(np.mean((np.full(len(y_), y.mean()) - y_)**2))] + [np.sqrt(np.mean((p - y_)**2)) for p in g.staged_predict(X_[:, None])]
    b = axs[1]; b.plot(range(101), rm(x, y), color=BLUE, label='train RMSE'); b.plot(range(101), rm(xt, yt), color=RED, label='test RMSE'); b.axvline(m, color='black'); b.legend(); b.set_xlabel('number of trees'); b.set_title('error falls, then flattens (early stopping point)')

@anim('A09_svm_margin', 'Linear SVM: C trades margin width against margin violations', 5, 1, 1200)
def _(axs, k, n):
    from sklearn.svm import SVC
    rng = np.random.default_rng(7); X = np.vstack([rng.normal([-1.2, -1], 1, (40, 2)), rng.normal([1.2, 1], 1, (40, 2))]); y = np.r_[np.zeros(40), np.ones(40)]; C = [.01, .1, 1, 10, 100][k]
    clf = SVC(kernel='linear', C=C).fit(X, y); ax = axs[0]; g1, g2 = np.meshgrid(np.linspace(-4.5, 4.5, 200), np.linspace(-4.5, 4.5, 200)); Z = clf.decision_function(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)
    ax.contourf(g1, g2, Z, levels=[-99, 0, 99], colors=['#FADBD8', '#DCE9FF'], alpha=.6); ax.contour(g1, g2, Z, levels=[-1, 0, 1], colors='black', linestyles=['--', '-', '--'], linewidths=1.3)
    ax.scatter(*X[y == 0].T, s=16, color=RED); ax.scatter(*X[y == 1].T, s=16, color=BLUE); ax.scatter(*clf.support_vectors_.T, s=90, facecolors='none', edgecolors='black', lw=1.2)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_title(f'C = {C}   margin width = {2/np.linalg.norm(clf.coef_):.2f}   support vectors = {len(clf.support_)}   (circled)')

@anim('A10_svm_kernel', 'Kernel trick: lift the data so a flat cut separates the classes', 30, 1, 110)
def _(axs, k, n):
    from sklearn.datasets import make_circles
    X, y = make_circles(120, noise=.08, factor=.4, random_state=0); t = k / (n - 1); Y = (1 - t) * X[:, 1] + t * 3 * (X[:, 0]**2 + X[:, 1]**2 - .55); ax = axs[0]
    ax.scatter(X[y == 0, 0], Y[y == 0], s=18, color=RED); ax.scatter(X[y == 1, 0], Y[y == 1], s=18, color=BLUE)
    if t > .6: ax.axhline(0, color='black', lw=2, alpha=min(1, (t - .6) / .3))
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.9, 1.9); ax.set_xlabel('x'); ax.set_ylabel('(1-t)*y + t*3(x^2+y^2-0.55)'); ax.set_title(f'feature map progress {t:.0%}:  from (x, y) to (x, x^2+y^2)' + ('   -> a straight line now separates the classes' if t > .95 else ''))

@anim('A12_hierarchical', 'Hierarchical clustering: merge the two closest clusters, again and again', 12, 2, 600)
def _(axs, k, n):
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    rng = np.random.default_rng(3); X = np.vstack([rng.normal(c, .45, (4, 2)) for c in ([0, 0], [3, .5], [1.2, 3])]); Z = linkage(X, 'ward'); c = 12 - k
    thr = Z[0, 2] / 2 if c == 12 else (Z[11 - c, 2] + Z[12 - c, 2]) / 2 if c > 1 else Z[-1, 2] * 1.05; lab = fcluster(Z, thr, 'distance')
    a = axs[0]; a.scatter(*X.T, s=60, c=[plt.cm.tab10((l - 1) % 10) for l in lab], ec='black', lw=.6); a.set_title(f'{len(set(lab))} cluster(s)'); a.set_aspect('equal')
    b = axs[1]; dendrogram(Z, ax=b, color_threshold=0, above_threshold_color='#555555', no_labels=True); b.axhline(thr, color=RED, lw=2); b.set_title('dendrogram: red line = where we cut'); b.spines['bottom'].set_visible(False)

@anim('A13_dbscan', 'DBSCAN: eps decides what counts as dense', 6, 1, 1100)
def _(axs, k, n):
    from sklearn.cluster import DBSCAN; from sklearn.datasets import make_moons
    X, _y = make_moons(150, noise=.06, random_state=0); rng = np.random.default_rng(1); X = np.vstack([X, rng.uniform([-1.5, -1], [2.5, 1.5], (15, 2))]); eps = [.05, .08, .12, .18, .3, .7][k]; lab = DBSCAN(eps=eps, min_samples=5).fit_predict(X); ax = axs[0]
    for l in sorted(set(lab)):
        m = lab == l; ax.scatter(*X[m].T, s=18 if l >= 0 else 30, color=GRAY if l < 0 else plt.cm.tab10(l % 10), marker='x' if l < 0 else 'o')
    ax.add_patch(Circle(X[0], eps, fill=False, ec='black', ls='--')); ax.set_aspect('equal'); ax.set_xlim(-1.5, 2.6); ax.set_ylim(-1, 1.6)
    ax.set_title(f'eps = {eps}   clusters = {len(set(lab) - {-1})}   noise points (x) = {int(np.sum(lab == -1))}   dashed circle = eps neighbourhood')

@functools.lru_cache(None)
def _em_states(n_iter=20):
    rng = np.random.default_rng(9); X = np.vstack([rng.multivariate_normal(m, c, 70) for m, c in [([-3, 0], [[1, .5], [.5, .7]]), ([2, 2], [[1, -.4], [-.4, .6]]), ([1.5, -2.5], [[.8, 0], [0, 1.2]])]])
    mu = np.array([[-1., -.5], [0., 0.], [1., .5]]); cov = np.array([np.eye(2) * 2.5] * 3); pi = np.ones(3) / 3; S = []
    def resp(mu, cov, pi):
        R = np.zeros((len(X), 3))
        for j in range(3): d = X - mu[j]; R[:, j] = pi[j] * np.exp(-.5 * np.einsum('ij,jk,ik->i', d, np.linalg.inv(cov[j]), d)) / np.sqrt((2 * np.pi)**2 * np.linalg.det(cov[j]))
        ll = np.log(R.sum(1)).sum(); return R / R.sum(1, keepdims=True), ll
    for _i in range(n_iter + 1):
        R, ll = resp(mu, cov, pi); S.append((mu.copy(), cov.copy(), R, ll)); Nk = R.sum(0); pi = Nk / len(X); mu = (R.T @ X) / Nk[:, None]
        cov = np.array([((R[:, j, None] * (X - mu[j])).T @ (X - mu[j])) / Nk[j] + 1e-6 * np.eye(2) for j in range(3)])
    return X, S
@anim('A14_gmm_em', 'EM for a Gaussian mixture: soft assignments and ellipses converge together', 21, 1, 350)
def _(axs, k, n):
    X, S = _em_states(); mu, cov, R, ll = S[k]; ax = axs[0]; col = np.array([[.23, .43, .86], [.18, .55, .34], [.85, .51, .17]]); ax.scatter(*X.T, s=18, c=R @ col, ec='none')
    for j in range(3):
        l, v = np.linalg.eigh(cov[j]); ang = np.degrees(np.arctan2(v[1, 1], v[0, 1])); ax.add_patch(Ellipse(mu[j], 4 * np.sqrt(l[1]), 4 * np.sqrt(l[0]), angle=ang, fill=False, ec=col[j], lw=2.5)); ax.plot(*mu[j], 'X', color=col[j], ms=11, mec='black')
    ax.set_xlim(-6, 6); ax.set_ylim(-5.5, 5); ax.set_aspect('equal'); ax.set_title(f'EM iteration {k}   log-likelihood = {ll:.1f}   (point colour = mix of component responsibilities)')

@anim('A16_roc_metrics', 'Sliding the threshold moves you along the ROC curve', 30, 2, 160)
def _(axs, k, n):
    rng = np.random.default_rng(0); neg = rng.normal(0, 1, 500); pos = rng.normal(1.6, 1, 300); th = 4 - 7 * k / (n - 1); allt = np.linspace(5, -4, 400)
    tpr = [(pos >= t).mean() for t in allt]; fpr = [(neg >= t).mean() for t in allt]; TP = int((pos >= th).sum()); FN = len(pos) - TP; FP = int((neg >= th).sum()); TN = len(neg) - FP
    a = axs[0]; a.hist(neg, bins=30, color=RED, alpha=.55, label='negatives'); a.hist(pos, bins=30, color=BLUE, alpha=.55, label='positives'); a.axvline(th, color='black', lw=2); a.legend(); a.set_xlabel('model score'); a.set_title(f'threshold = {th:+.2f}   TP={TP}  FP={FP}  FN={FN}  TN={TN}')
    b = axs[1]; b.plot(fpr, tpr, color=GREEN, lw=2); b.plot([0, 1], [0, 1], ':', color=GRAY); b.plot(FP / len(neg), TP / len(pos), 'o', color=RED, ms=10); b.set_xlabel('false positive rate'); b.set_ylabel('recall (true positive rate)')
    prec = TP / max(TP + FP, 1); b.set_title(f'recall = {TP/len(pos):.2f}   FPR = {FP/len(neg):.2f}   precision = {prec:.2f}   AUC = {abs(np.trapezoid(tpr, fpr)):.2f}')

# ============================================================ rendering
def render(key, dpi=90):
    a = ANIMS[key]; frames = []; size = (6.6, 4.5) if a['naxes'] == 1 else (10.2, 4.3)
    for k in range(a['n']):
        fig, axs = plt.subplots(1, a['naxes'], figsize=size); axs = np.atleast_1d(axs); sup = a['fn'](axs, k, a['n'])
        if sup: fig.suptitle(sup, fontsize=10, fontweight='bold'); fig.tight_layout(rect=(0, 0, 1, .93))
        else: fig.tight_layout()
        fig.canvas.draw(); frames.append(Image.frombuffer('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert('RGB')); plt.close(fig)
    dur = [a['ms']] * (len(frames) - 1) + [a['last_ms']]
    frames[0].save(f'{HERE}/gifs/{key}.gif', save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=True)
    idx = sorted(set(np.linspace(0, len(frames) - 1, 6).round().astype(int))); pick = [frames[i] for i in idx]
    cols = 3 if a['naxes'] == 1 else 2; w, h = pick[0].size; rows = -(-len(pick) // cols); pad = 10
    sheet = Image.new('RGB', (cols * w + pad * (cols + 1), rows * h + pad * (rows + 1)), 'white')
    for i, im in enumerate(pick): sheet.paste(im, (pad + (i % cols) * (w + pad), pad + (i // cols) * (h + pad)))
    sheet.save(f'{HERE}/filmstrips/{key}.png'); return len(frames)
if __name__ == '__main__':
    keys = sys.argv[1:] or list(ANIMS)
    for key in keys: print(key, render(key), 'frames', flush=True)
