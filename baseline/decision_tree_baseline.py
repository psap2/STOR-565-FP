"""Class baseline: decision tree (CART) predicting the perfect-play outcome of UCI Connect-4 positions.
Board encoding: x (first player, to move) = +1, o = -1, blank = 0. Columns are a..g, rows 1..6 bottom to top.
Usage: python decision_tree_baseline.py <connect-4.data> <results_dir>
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

SEED = 565
data_path, out_dir = sys.argv[1], sys.argv[2]
df = pd.read_csv(data_path, header=None)
X = df.iloc[:, :42].replace({'x': 1, 'o': -1, 'b': 0}).astype(int).values
y = df.iloc[:, 42].values
# 80/20 stratified split. The SAME split indices are saved so the value-head evaluation uses the same test set.
idx = np.arange(len(y))
tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
np.save(f"{out_dir}/uci_test_idx.npy", te); np.save(f"{out_dir}/uci_train_idx.npy", tr)

classes = ['loss', 'draw', 'win']
maj = pd.Series(y[tr]).value_counts().idxmax()
maj_acc = accuracy_score(y[te], np.full(len(te), maj))

grid = {'max_depth': [2, 4, 6, 8, 10, 12, 16, 20, None], 'min_samples_leaf': [1, 5, 20]}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
gs = GridSearchCV(DecisionTreeClassifier(random_state=SEED), grid, cv=cv, scoring='accuracy', n_jobs=-1)
gs.fit(X[tr], y[tr])
pred = gs.predict(X[te])
acc = accuracy_score(y[te], pred)
cm = confusion_matrix(y[te], pred, labels=classes)
n = len(te); z = 1.96
d = 1 + z*z/n; c = (acc + z*z/(2*n))/d; h = z*np.sqrt(acc*(1-acc)/n + z*z/(4*n*n))/d
res = dict(n_total=len(y), n_train=len(tr), n_test=n, seed=SEED,
           class_counts_total=pd.Series(y).value_counts().to_dict(),
           majority_class=maj, majority_test_acc=round(maj_acc, 4),
           best_params={k: (None if v is None else int(v)) for k, v in gs.best_params_.items()},
           cv_acc=round(gs.best_score_, 4), test_acc=round(acc, 4),
           test_acc_wilson95=[round(c-h, 4), round(c+h, 4)],
           confusion_labels_rows_true_cols_pred=classes, confusion=cm.tolist(),
           per_class_recall={cl: round(cm[i, i]/cm[i].sum(), 4) for i, cl in enumerate(classes)})
json.dump(res, open(f"{out_dir}/decision_tree_baseline.json", 'w'), indent=2)
print(json.dumps(res, indent=2))
