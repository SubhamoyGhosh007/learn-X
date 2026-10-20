"""One Graphviz diagram per mathematical idea / algorithm. Writes .dot + .png + .svg to diagrams/."""
import os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f'{HERE}/diagrams'
FILL = {'b': ('#DCE9FF', '#3B6EDC'), 'g': ('#DFF3E4', '#2E8B57'), 'o': ('#FFE9D2', '#D9822B'), 'y': ('#FFF3C4', '#C9A100'),
        'r': ('#FADBD8', '#B03A2E'), 'w': ('white', '#666666'), 'p': ('#EBDFFB', '#7A4FC2'), 'n': ('#EEEEEE', '#444444')}
DIAG = {}
def D(key, title, nodes, edges, rankdir='TB', extra=''):
    s = (f'digraph G {{ rankdir={rankdir}; bgcolor="white"; nodesep=0.35; ranksep=0.42; pad=0.3; labelloc=t; fontname="Helvetica-Bold"; fontsize=16; label="{title}";\n'
         'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.16,0.09", penwidth=1.4];\n'
         'edge [fontname="Helvetica", fontsize=10.5, color="#555555", arrowsize=0.8];\n')
    for nid, v in nodes.items():
        label, k = v[0], v[1]; fill, bd = FILL[k]; attr = v[2] if len(v) > 2 else ''
        s += f'{nid} [label="{label}", fillcolor="{fill}", color="{bd}"{attr}];\n'
    DIAG[key] = s + edges + '\n' + extra + '}\n'
DM = ', shape=diamond, margin="0.04,0.04"'

# ---------------- mathematics ----------------
D('S01_dot_product', 'Dot product: a weighted sum and a measure of alignment',
  {'a': ('vector a\\n(a1, a2, ..., an)', 'b'), 'b': ('vector b\\n(b1, b2, ..., bn)', 'b'), 'm': ('multiply matching entries\\na1*b1, a2*b2, ...', 'g'), 's': ('add them up\\na.b = a1b1 + ... + anbn', 'g'),
   'c': ('same as |a| |b| cos(angle)', 'o'), 'p': ('a.b > 0\\npoint the same way', 'w'), 'z': ('a.b = 0\\nperpendicular', 'w'), 'n': ('a.b < 0\\npoint opposite ways', 'w'),
   'ml': ('In ML: w . x = weighted sum of features\\n(the core of every linear model)', 'p')},
  'a -> m; b -> m; m -> s -> c; c -> p; c -> z; c -> n; s -> ml [style=dashed];')
D('S02_matrix_eigen', 'Matrices: many dot products at once, and special directions',
  {'X': ('X: m samples x n features', 'b'), 'w': ('w: n weights', 'b'), 'r': ('Xw: m predictions\\n(row i of X dot w)', 'g'), 'rule': ('shape rule: (m x n)(n x p) = (m x p)\\nthe two inner sizes must match', 'y'),
   'A': ('square matrix A\\n(a transformation of space)', 'o'), 'e': ('eigenvector v:  A v = lambda v\\nA only stretches v, never turns it', 'p'), 'pca': ('PCA uses the eigenvectors of the\\ncovariance matrix as its axes', 'g'),
   'inv': ('inverse: A A^-1 = I\\nsolves  X^T X w = X^T y', 'n')},
  'X -> r; w -> r; r -> rule [style=dashed]; A -> e -> pca; A -> inv [style=dashed];')
D('S03_derivative', 'The derivative: the slope of a function at a point',
  {'f': ('function f(x)', 'b'), 'h': ('nudge x by a tiny h', 'g'), 'q': ('ratio  (f(x+h) - f(x)) / h', 'g'), 'l': ('let h -> 0', 'g'), 'd': ("f'(x) = slope at x", 'o'),
   'p': ("f'(x) > 0\\nfunction rising", 'w'), 'm': ("f'(x) < 0\\nfunction falling", 'w'), 'z': ("f'(x) = 0\\nflat: a minimum or maximum", 'r'),
   'ml': ('In ML: derivative of the loss tells us\\nwhich way to move each parameter', 'p')},
  'f -> h -> q -> l -> d; d -> p; d -> m; d -> z; z -> ml [style=dashed];')
D('S04_gradient', 'Gradient: all the slopes together point uphill',
  {'L': ('loss L(w1, w2, ...)', 'b'), 'pd': ('partial derivative per parameter\\ndL/dw1, dL/dw2, ... (others held fixed)', 'g'), 'gr': ('gradient vector\\n(dL/dw1, dL/dw2, ...)', 'o'), 'up': ('points in the direction of\\nsteepest INCREASE of L', 'w'),
   'st': ('step the opposite way:\\nw = w - alpha * gradient', 'y'), 'lp': ('repeat until the loss stops falling', 'g')},
  'L -> pd -> gr -> up; gr -> st -> lp; lp -> L [style=dashed, label=" new w"];')
D('S05_chain_rule', 'Chain rule: derivatives multiply along a chain of functions',
  {'x': ('x', 'n'), 'u': ('u = g(x)', 'b'), 'y': ('y = f(u)', 'b'), 'a': ('du/dx = g\'(x)', 'g'), 'bb': ('dy/du = f\'(u)', 'g'), 'r': ('dy/dx = (dy/du) * (du/dx)', 'o'),
   'ml': ('In ML: loss <- prediction <- weights.\\nMultiply the slopes along that chain to get dLoss/dWeight\\n(this is how every gradient in Part B is derived)', 'p')},
  'x -> u [label=" g"]; u -> y [label=" f"]; u -> a [style=dashed, arrowhead=none]; y -> bb [style=dashed, arrowhead=none]; a -> r; bb -> r; r -> ml [style=dashed];', 'LR')
D('S06_bayes', "Bayes' rule: update a belief with evidence",
  {'pr': ('prior  P(A)\\nwhat you believed before', 'b'), 'li': ('likelihood  P(B | A)\\nhow well the evidence fits A', 'b'), 'nu': ('multiply: P(B | A) * P(A)', 'g'), 'ev': ('evidence  P(B)\\ntotal chance of seeing B', 'y'),
   'po': ('posterior  P(A | B)\\nupdated belief', 'o'), 'ex': ('example: 1% prevalence, 99% sensitivity, 5% false alarms\\nPositive test -> only about 17% chance of being sick\\n(a tiny prior drags the posterior down)', 'p')},
  'pr -> nu; li -> nu; nu -> po; ev -> po [label=" divide by"]; po -> ex [style=dashed];')
D('S07_normal_clt', 'Normal distribution and the central limit theorem',
  {'e': ('many small independent effects', 'b'), 'avg': ('add or average them', 'g'), 'bell': ('bell curve (normal)\\nmean mu, spread sigma', 'o'), 'r': ('68% within 1 sigma\\n95% within 2 sigma\\n99.7% within 3 sigma', 'w'),
   'ml': ('Why it matters: measurement noise is often normal,\\nso least squares (squared error) is the natural loss', 'p')},
  'e -> avg [label=" central limit theorem"]; avg -> bell -> r; bell -> ml [style=dashed];', 'LR')
D('S08_covariance', 'Variance, covariance and correlation',
  {'x': ('feature x', 'b'), 'y': ('feature y', 'b'), 'dx': ('deviations from the mean\\nxi - mean(x),  yi - mean(y)', 'g'), 'pr': ('multiply paired deviations', 'g'), 'av': ('average = covariance\\n>0 move together, <0 opposite', 'o'),
   'sc': ('divide by sigma_x * sigma_y', 'y'), 'r': ('correlation r in [-1, +1]\\nunit-free strength of the LINEAR link', 'o'), 'var': ('variance of one feature =\\ncovariance of x with itself', 'w'), 'cm': ('covariance matrix of all features\\n-> eigenvectors -> PCA', 'p')},
  'x -> dx; y -> dx; dx -> pr -> av -> sc -> r; av -> var [style=dashed]; av -> cm [style=dashed];')
D('S09_mle', 'Maximum likelihood: pick the parameters that make the data most probable',
  {'d': ('observed data', 'b'), 'm': ('probabilistic model with parameters theta', 'b'), 'l': ('likelihood L(theta) = P(data | theta)', 'g'), 'lg': ('take logs: products become sums\\nlog-likelihood', 'g'), 'mx': ('maximize over theta  ->  theta-hat', 'o'),
   'g': ('Gaussian noise\\n=> minimize squared error\\n(linear regression)', 'p'), 'bn': ('Bernoulli labels\\n=> minimize log-loss\\n(logistic regression)', 'p')},
  'd -> l; m -> l; l -> lg -> mx; mx -> g [style=dashed]; mx -> bn [style=dashed];')
D('S10_bias_variance', 'Bias, variance and the complexity trade-off',
  {'er': ('expected error on NEW data\\n= bias^2 + variance + noise', 'o'), 'b': ('bias: model too simple\\n=> underfitting', 'b'), 'v': ('variance: model chases noise\\n=> overfitting', 'r'), 'n': ('irreducible noise\\n(cannot be removed)', 'w'),
   'up': ('more flexible model:\\nbias falls, variance rises', 'y'), 'fx1': ('fight bias: richer model,\\nbetter features', 'g'), 'fx2': ('fight variance: more data, regularization,\\nensembles, simpler model', 'g')},
  'er -> b; er -> v; er -> n; b -> up [style=dashed]; v -> up [style=dashed]; b -> fx1; v -> fx2;')
D('S11_scaling', 'Feature scaling without leaking',
  {'raw': ('raw features\\n(different units and ranges)', 'n'), 'sp': ('split train / test FIRST', 'y'), 'fit': ('fit scaler on TRAIN only\\nmean mu, std sigma', 'g'), 'tr': ('transform train', 'b'), 'te': ('transform validation / test\\nwith the SAME mu, sigma', 'b'),
   'fm': ('z = (x - mu) / sigma\\n(mean 0, std 1)', 'o'), 'mm': ('or min-max: (x - min) / (max - min)', 'w'), 'md': ('model (k-NN, SVM, ridge, gradient descent\\nall need this)', 'p')},
  'raw -> sp -> fit; fit -> tr; fit -> te; fit -> fm [style=dashed]; fm -> mm [style=dashed, arrowhead=none]; tr -> md; te -> md;')
D('S12_cross_validation', 'k-fold cross-validation procedure',
  {'sp': ('split data into k folds', 'y'), 'lp': ('for each fold i = 1..k', 'g'), 'tr': ('train on the other k-1 folds', 'b'), 'va': ('validate on fold i -> score i', 'b'), 'av': ('average the k scores\\n(and look at their spread)', 'o'), 'ch': ('compare models / hyperparameters\\nby that average', 'p'), 'ts': ('touch the TEST set once, at the end', 'r')},
  'sp -> lp -> tr -> va; va -> lp [style=dashed, label=" next fold"]; va -> av -> ch -> ts;')

# ---------------- algorithms ----------------
D('A01_linear_regression', 'Linear regression: model, loss, learning',
  {'x': ('features X', 'b'), 'w': ('weights w (and intercept)', 'b'), 'p': ('prediction  y-hat = X w', 'g'), 'r': ('residuals  y-hat - y', 'g'), 'l': ('loss: MSE = mean of squared residuals', 'o'),
   'gr': ('gradient = (2/m) X^T (X w - y)', 'y'), 'up': ('w = w - alpha * gradient', 'g'), 'ne': ('or in one step:\\nw = (X^T X)^-1 X^T y', 'p'), 'rg': ('add a penalty on w -> Ridge / Lasso', 'w')},
  'x -> p; w -> p; p -> r -> l -> gr -> up; up -> p [style=dashed, label=" repeat"]; l -> ne [style=dashed]; l -> rg [style=dashed];')
D('A02_logistic_regression', 'Logistic regression: from features to a probability',
  {'x': ('features x', 'b'), 'z': ('score  z = w . x + b', 'g'), 's': ('sigmoid  p = 1 / (1 + e^-z)\\nsquashes to (0, 1)', 'o'), 'th': ('p > threshold (0.5)?', 'y', DM), 'c1': ('predict class 1', 'g'), 'c0': ('predict class 0', 'r'),
   'ls': ('loss: log-loss\\n-[y ln p + (1-y) ln(1-p)]', 'p'), 'gr': ('gradient = (1/m) X^T (p - y)\\nthen w = w - alpha * gradient', 'p')},
  'x -> z -> s -> th; th -> c1 [label=" yes"]; th -> c0 [label=" no"]; s -> ls [style=dashed]; ls -> gr [style=dashed];')
D('A03_regularization', 'Regularization: pay a price for large weights',
  {'ld': ('data loss  (e.g. MSE)', 'b'), 'pn': ('penalty on weights, scaled by lambda', 'y'), 'tot': ('total loss = data loss + lambda * penalty', 'o'), 'l2': ('Ridge (L2): sum of w^2\\nshrinks all weights smoothly', 'g'), 'l1': ('Lasso (L1): sum of |w|\\nsets some weights EXACTLY to zero', 'g'),
   'en': ('Elastic Net: mix of both', 'w'), 'cv': ('choose lambda by cross-validation\\n(standardize features first)', 'p')},
  'ld -> tot; pn -> tot; pn -> l2; pn -> l1; l2 -> en [style=dashed, arrowhead=none]; l1 -> en [style=dashed, arrowhead=none]; tot -> cv [style=dashed];')
D('A04_knn', 'k-Nearest Neighbors: ask the closest examples',
  {'q': ('new point to predict', 'b'), 'sc': ('scale features first', 'y'), 'ds': ('distance to EVERY stored training point\\nEuclidean / Manhattan / cosine', 'g'), 'pk': ('pick the k closest', 'g'), 'vt': ('classification: majority vote\\nregression: average of targets', 'o'),
   'out': ('prediction', 'g'), 'kk': ('small k: wiggly, overfits\\nlarge k: smooth, underfits\\n(choose k by cross-validation)', 'p')},
  'q -> sc -> ds -> pk -> vt -> out; pk -> kk [style=dashed];')
D('A05_naive_bayes', 'Naive Bayes: Bayes rule with an independence shortcut',
  {'x': ('features x1 ... xn', 'b'), 'pr': ('class prior P(c)\\n(class frequencies)', 'b'), 'lk': ('per-feature likelihoods P(xj | c)\\nGaussian / multinomial / Bernoulli', 'g'), 'nv': ('naive: assume features are independent\\ngiven the class -> multiply them', 'y'),
   'sc': ('score(c) = P(c) * product of P(xj | c)\\n(use logs to avoid underflow)', 'o'), 'ar': ('predict the class with the highest score', 'g'), 'ls': ('Laplace smoothing: add 1 to counts\\nso no probability is exactly 0', 'p')},
  'x -> lk; pr -> sc; lk -> nv -> sc -> ar; lk -> ls [style=dashed];')
D('A06_decision_tree', 'Decision tree: grow by repeatedly choosing the best split',
  {'n': ('node with some samples', 'b'), 'tr': ('try every feature and every threshold', 'g'), 'sc': ('score each split by impurity drop\\nentropy / Gini / variance', 'g'), 'bs': ('take the split with the highest\\ninformation gain', 'o'), 'sp': ('send samples left / right', 'g'),
   'st': ('stop? pure node, max depth,\\nmin samples reached', 'y', DM), 'lf': ('leaf: predict majority class\\n(or mean for regression)', 'g'), 'pr': ('too deep => overfits:\\nlimit depth or prune', 'p')},
  'n -> tr -> sc -> bs -> sp -> st; st -> lf [label=" yes"]; st -> n [label=" no: recurse\\n on each child"]; lf -> pr [style=dashed];')
D('A07_random_forest', 'Random forest: many decorrelated trees, averaged',
  {'d': ('training data', 'n'), 'bs': ('draw B bootstrap samples\\n(rows with replacement)', 'g'), 't1': ('deep tree 1', 'b'), 't2': ('deep tree 2', 'b'), 't3': ('deep tree B', 'b'), 'rf': ('at every split each tree sees only\\na random subset of features', 'y'),
   'ag': ('classification: vote\\nregression: average', 'o'), 'oob': ('out-of-bag rows (about 37%)\\n= free validation estimate', 'p'), 'imp': ('permutation importance\\nranks the features', 'p')},
  'd -> bs; bs -> t1; bs -> t2; bs -> t3; rf -> t1 [style=dashed]; rf -> t2 [style=dashed]; rf -> t3 [style=dashed]; t1 -> ag; t2 -> ag; t3 -> ag; bs -> oob [style=dashed]; ag -> imp [style=dashed];')
D('A08_gradient_boosting', 'Gradient boosting: each tree fixes the leftover errors',
  {'f0': ('F0 = a constant\\n(the mean of y)', 'n'), 'rs': ('residuals = negative gradient of the loss\\n(y - F for squared error)', 'g'), 'tr': ('fit a SHALLOW tree to the residuals', 'b'), 'up': ('F = F + eta * tree\\n(eta = learning rate)', 'o'),
   'st': ('more trees? early stopping on a\\nvalidation set decides', 'y', DM), 'out': ('final model = sum of all trees', 'g'), 'lib': ('XGBoost, LightGBM, CatBoost,\\nHistGradientBoosting', 'p')},
  'f0 -> rs -> tr -> up -> st; st -> rs [label=" yes"]; st -> out [label=" no"]; out -> lib [style=dashed];')
D('A09_svm_margin', 'Support vector machine: the widest street between classes',
  {'d': ('labeled points  y = +1 / -1', 'b'), 'h': ('hyperplane  w . x + b = 0', 'g'), 'mg': ('margin = 2 / |w|\\nmaximize margin = minimize |w|', 'o'), 'sv': ('support vectors: the points on the margin\\n(only they define the boundary)', 'y'),
   'sf': ('soft margin: allow violations,\\npay C per unit of slack', 'g'), 'c': ('large C: narrow margin, fits tightly\\nsmall C: wide margin, smoother', 'p'), 'hl': ('same as hinge loss + L2 penalty', 'w')},
  'd -> h -> mg -> sv; mg -> sf -> c; sf -> hl [style=dashed];')
D('A10_svm_kernel', 'The kernel trick: curved boundaries from a linear method',
  {'ns': ('not linearly separable\\nin the original space', 'r'), 'fm': ('feature map  phi(x)\\ne.g. add x^2 + y^2', 'g'), 'sp': ('separable by a flat plane\\nin the new space', 'g'), 'kt': ('SVM only needs dot products,\\nso use K(a, b) = phi(a) . phi(b)\\nwithout ever computing phi', 'o'),
   'rb': ('RBF kernel  exp(-gamma |a - b|^2)\\ninfinite-dimensional feature space', 'b'), 'pm': ('tune C and gamma together\\n(log grid + cross-validation)', 'p')},
  'ns -> fm -> sp; fm -> kt [style=dashed]; kt -> rb -> pm;')
D('A11_kmeans', 'k-means: assign, update, repeat',
  {'k': ('choose k', 'y'), 'in': ('initialize k centroids\\n(k-means++ spreads them out)', 'g'), 'as': ('assign every point to its nearest centroid', 'b'), 'up': ('move each centroid to the mean\\nof its assigned points', 'b'), 'cv': ('assignments unchanged?', 'y', DM),
   'dn': ('done: clusters + centroids\\n(local minimum of inertia)', 'o'), 'ch': ('choose k: elbow plot or silhouette score\\nrun several restarts', 'p')},
  'k -> in -> as -> up -> cv; cv -> as [label=" no"]; cv -> dn [label=" yes"]; dn -> ch [style=dashed];')
D('A12_hierarchical', 'Hierarchical (agglomerative) clustering',
  {'e': ('every point is its own cluster', 'n'), 'dm': ('distance matrix between clusters', 'g'), 'mg': ('merge the two closest clusters', 'b'), 'lk': ('linkage decides what closest means:\\nsingle / complete / average / Ward', 'y'), 'one': ('one cluster left?', 'y', DM),
   'dn': ('dendrogram records every merge', 'o'), 'cut': ('cut the tree at a height\\n=> choose the number of clusters', 'p')},
  'e -> dm -> mg -> one; lk -> dm [style=dashed]; one -> dm [label=" no: update distances"]; one -> dn [label=" yes"]; dn -> cut;')
D('A13_dbscan', 'DBSCAN: clusters are dense regions',
  {'p': ('parameters: radius eps, min_samples', 'y'), 'cn': ('count neighbors within eps of each point', 'g'), 'cr': ('>= min_samples neighbors:\\nCORE point', 'b'), 'bd': ('within eps of a core point\\nbut not core: BORDER point', 'w'), 'ns': ('neither: NOISE (label -1)', 'r'),
   'ch': ('link core points that are within eps\\nof each other -> grow a cluster', 'o'), 'out': ('arbitrary-shaped clusters + outliers\\n(no k needed)', 'g')},
  'p -> cn; cn -> cr; cn -> bd; cn -> ns; cr -> ch -> out; bd -> ch [style=dashed];')
D('A14_gmm_em', 'Gaussian mixture model fitted by EM',
  {'in': ('initialize means, covariances, weights', 'n'), 'e': ('E-step: for each point compute the\\nprobability it came from each component\\n(responsibilities)', 'b'), 'm': ('M-step: re-estimate each component using\\nresponsibilities as soft weights', 'g'), 'll': ('log-likelihood improved?', 'y', DM),
   'out': ('soft cluster memberships +\\nelliptical clusters', 'o'), 'cmp': ('vs k-means: soft assignments,\\nflexible shapes; choose k with BIC', 'p')},
  'in -> e -> m -> ll; ll -> e [label=" yes"]; ll -> out [label=" converged"]; out -> cmp [style=dashed];')
D('A15_pca', 'PCA: from covariance to a smaller representation',
  {'x': ('data matrix X (m x n)', 'b'), 'sc': ('center (and standardize) each feature', 'y'), 'cv': ('covariance matrix (n x n)', 'g'), 'eg': ('eigen-decomposition (or SVD)\\nS w = lambda w', 'g'), 'so': ('sort eigenvectors by eigenvalue\\n(variance captured)', 'g'),
   'k': ('keep top k: explained variance\\n90 to 99%, or a scree-plot elbow', 'y'), 'pj': ('project:  Z = X W   (m x k)', 'o'), 'cav': ('linear and unsupervised; scale first;\\nuse t-SNE / UMAP only for plotting', 'p')},
  'x -> sc -> cv -> eg -> so -> k -> pj; pj -> cav [style=dashed];')
D('A16_roc_metrics', 'Confusion matrix, metrics and the ROC curve',
  {'cm': ('<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="9"><TR><TD BORDER="0"></TD><TD BGCOLOR="#EEEEEE"><B>predicted +</B></TD><TD BGCOLOR="#EEEEEE"><B>predicted -</B></TD></TR><TR><TD BGCOLOR="#EEEEEE"><B>actual +</B></TD><TD BGCOLOR="#D5F5E3">TP</TD><TD BGCOLOR="#FADBD8">FN (miss)</TD></TR><TR><TD BGCOLOR="#EEEEEE"><B>actual -</B></TD><TD BGCOLOR="#FADBD8">FP (false alarm)</TD><TD BGCOLOR="#D5F5E3">TN</TD></TR></TABLE>>', 'w'),
   'pre': ('precision = TP / (TP + FP)\\nhow trustworthy a positive call is', 'b'), 'rec': ('recall = TP / (TP + FN)\\nhow many real positives we catch', 'b'), 'f1': ('F1 = 2PR / (P + R)', 'o'), 'acc': ('accuracy = (TP + TN) / total\\nmisleading when classes are imbalanced', 'w'),
   'th': ('slide the decision threshold:\\ntrade precision against recall', 'y'), 'roc': ('ROC: plot recall (TPR) against FPR\\nAUC = chance a random positive outranks\\na random negative', 'g'), 'pr': ('imbalanced data: use the\\nprecision-recall curve', 'p')},
  'cm -> pre; cm -> rec; cm -> acc; pre -> f1; rec -> f1; th -> roc; th -> pr [style=dashed]; rec -> roc [style=dashed];')
# html label fix: Graphviz needs the label as HTML <...>; patch the one node afterwards
DIAG['A16_roc_metrics'] = DIAG['A16_roc_metrics'].replace('label="<<TABLE', 'label=<<TABLE').replace('</TABLE>>", fillcolor', '</TABLE>>, fillcolor')

def main():
    for key, src in DIAG.items():
        open(f'{OUT}/{key}.dot', 'w').write(src)
        subprocess.run(['dot', '-Tpng', '-Gdpi=170', f'{OUT}/{key}.dot', '-o', f'{OUT}/{key}.png'], check=True)
        subprocess.run(['dot', '-Tsvg', f'{OUT}/{key}.dot', '-o', f'{OUT}/{key}.svg'], check=True)
    print(len(DIAG), 'diagrams rendered')
if __name__ == '__main__': main()
