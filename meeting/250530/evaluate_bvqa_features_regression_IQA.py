import argparse
import os
import sys
import time
import warnings

import numpy as np
import scipy.io
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR, LinearSVR
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import scipy.stats
from scipy.optimize import curve_fit
from tqdm import tqdm

warnings.filterwarnings("ignore")

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_file', type=str, required=True, help='Input .mat file containing feats_mat and mos')
    parser.add_argument('--out_file', type=str, required=True, help='Output results file prefix')
    parser.add_argument('--log_file', type=str, default='logs/eval.log', help='Log file')
    parser.add_argument('--log_short', action='store_true', help='Short log format')
    parser.add_argument('--num_iterations', type=int, default=6, help='Number of 80-20 splits')
    return parser.parse_args()

def logistic_func(X, b1, b2, b3, b4):
    logistic = 1 + np.exp(-(X - b3) / abs(b4))
    return b2 + (b1 - b2) / logistic

def compute_metrics(y_pred, y):
    srcc = scipy.stats.spearmanr(y, y_pred)[0]
    try:
        krcc = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        krcc = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]
    beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
    popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
    y_pred_log = logistic_func(y_pred, *popt)
    plcc = scipy.stats.pearsonr(y, y_pred_log)[0]
    rmse = np.sqrt(mean_squared_error(y, y_pred_log))
    return [srcc, krcc, plcc, rmse], y_pred_log

def evaluate_once(X, y, log_short):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    param_grid = {'C': np.logspace(1, 10, 10, base=2), 'gamma': np.logspace(-10, -6, 5, base=2)}
    grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=8, n_jobs=4, verbose=0)
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    grid.fit(X_train, y_train)
    best = grid.best_params_
    model = SVR(C=best['C'], gamma=best['gamma'])
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    m_train, _ = compute_metrics(train_pred, y_train)
    m_test, y_test_log = compute_metrics(test_pred, y_test)
    if not log_short:
        print("Train:", m_train)
        print("Test:", m_test)
    return m_train + m_test

def main(args):
    mat = scipy.io.loadmat(args.feature_file)
    X = np.asarray(mat['feats_mat'], dtype=np.float32)
    y = np.asarray(mat['mos']).squeeze().astype(np.float32)
    X[np.isinf(X)] = np.nan
    X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)

    results = []
    for i in tqdm(range(args.num_iterations), desc="Evaluations"):
        results.append(evaluate_once(X, y, args.log_short))

    results = np.asarray(results, dtype=np.float32) 
    np.save(args.out_file + ".npy", results)
    scipy.io.savemat(args.out_file + ".mat", {'all_iterations': results})
    print("Saved to:", args.out_file)

if __name__ == '__main__':
    args = arg_parser()
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    sys.stdout = Logger(args.log_file)
    print(args)
    main(args)
