"""
Anomaly detection experiments.
  A1  Isolation Forest per graph (fully unsupervised): most anomalous edges first
  A2  Local Outlier Factor per graph (unsupervised)
  A3  Novelty detection: Isolation Forest fitted on "rest" edges of the training graphs only;
      test edges that look least like ordinary edges come first
  A4  Isolation-Forest score added as an extra feature to the best LightGBM regression
"""
import json, os, sys, time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import gbm

BLOCKS = ["xn", "xe", "xp"]


def iforest_scores(X, seed=0):
    f = IsolationForest(n_estimators=200, max_samples=512, random_state=seed, n_jobs=2).fit(X)
    return -f.score_samples(X)                        # higher = more anomalous


def main():
    base = base_records()
    train, test = base[:14], base[16:20]
    big = extra_records(50, start_seed=5000, tag="test_ba200")
    rows = {}
    t = time.time()
    for name, sets in [("4 test graphs", test), ("50 new test graphs", big[:20])]:
        rows[f"A1 IsolationForest per graph [{name}]"] = eval_many([iforest_scores(features(r, BLOCKS)) for r in sets], sets)
        rows[f"A2 LOF per graph [{name}]"] = eval_many(
            [-LocalOutlierFactor(n_neighbors=50, n_jobs=2).fit(features(r, BLOCKS)).negative_outlier_factor_ for r in sets], sets)
        # A3: novelty detection trained on ordinary ("rest") edges of the training graphs
        rng = np.random.default_rng(0)
        Xn = np.concatenate([features(r, BLOCKS)[gbm.labels_from(r["red"]) == 0][rng.choice(15000, 2000, replace=False)] for r in train])
        f = IsolationForest(n_estimators=200, max_samples=512, random_state=0, n_jobs=2).fit(Xn)
        rows[f"A3 novelty IF (fit on rest edges) [{name}]"] = eval_many([-f.score_samples(features(r, BLOCKS)) for r in sets], sets)
    print("anomaly scores done %.0fs" % (time.time() - t), flush=True)
    print(header())
    for k, m in rows.items():
        print(fmt(k, m), flush=True)
    json.dump(rows, open(os.path.join(ROOT, "experiments/v3/results/anomaly.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
