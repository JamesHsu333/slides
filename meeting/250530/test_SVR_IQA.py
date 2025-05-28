# -*- coding: utf-8 -*-
'''
This script shows how to predict the quality score by pretrained SVR/linearSVR

python test_SVR_IQA.py \
  --feature_file feat_files/cinnamoroll_feats.mat \
  --best_parameter best_pamtr/cinnamoroll_GAMIVAL_pamtr \
  --test_csv ../video_quality_accessment/250523/mos_scores_test.csv
'''

import pandas as pd
import scipy.io
import numpy as np
import argparse
import time
import os, sys
import warnings
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from scipy.optimize import curve_fit
from sklearn.svm import SVR, LinearSVR
import scipy.stats

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
  parser.add_argument('--feature_file', type=str, required=True)
  parser.add_argument('--best_parameter', type=str, required=True)
  parser.add_argument('--predicted_score', type=str, default='predicted_score/cinnamoroll_GAMIVAL_predicted_score')
  parser.add_argument('--log_file', type=str, default='logs/cinnamoroll_predict.log')
  parser.add_argument('--log_short', action='store_true')
  parser.add_argument('--test_csv', type=str, required=True)
  return parser.parse_args()

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
  logisticPart = 1 + np.exp(-(X - bayta3) / np.abs(bayta4))
  yhat = bayta2 + (bayta1 - bayta2) / logisticPart
  return yhat

def compute_metrics(y_pred, y):
  SRCC = scipy.stats.spearmanr(y, y_pred)[0]
  try:
    KRCC = scipy.stats.kendalltau(y, y_pred)[0]
  except:
    KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

  beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
  popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
  y_pred_logistic = logistic_func(y_pred, *popt)

  PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
  RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
  return [SRCC, KRCC, PLCC, RMSE], y_pred_logistic

def evaluate_and_predict(X, y, best_params_SVR, best_params_linearSVR):
  model_svr = SVR(C=best_params_SVR['C'], gamma=best_params_SVR['gamma'])
  model_svr.fit(X, y)
  y_pred_svr = model_svr.predict(X)
  metrics_svr, y_pred_logistic_svr = compute_metrics(y_pred_svr, y)

  model_lsvr = LinearSVR(C=best_params_linearSVR['C'], epsilon=best_params_linearSVR['epsilon'])
  model_lsvr.fit(X, y)
  y_pred_lsvr = model_lsvr.predict(X)
  metrics_lsvr, y_pred_logistic_lsvr = compute_metrics(y_pred_lsvr, y)

  return metrics_svr, y_pred_logistic_svr, metrics_lsvr, y_pred_logistic_lsvr

def main(args):
  df = pd.read_csv(args.test_csv)
  y = df['score'].values.astype(np.float32)
  image_name_list = df['image_name'].tolist()

  mat = scipy.io.loadmat(args.feature_file)
  X_all = np.asarray(mat['feats_mat'], dtype=np.float32)
  raw_names = mat['image_names'].squeeze()
  mat_names = [n.strip().lower() for n in raw_names]

  name_to_index = {name: idx for idx, name in enumerate(mat_names)}
  indices = [name_to_index[name] for name in image_name_list if name in name_to_index]

  X = X_all[indices]
  y = y[:len(indices)]

  X[np.isinf(X)] = np.nan
  X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)
  X = preprocessing.MinMaxScaler().fit_transform(X)

  best_svr_mat = scipy.io.loadmat(args.best_parameter+'_SVR.mat')
  best_params_SVR = {
    'C': float(best_svr_mat['best_parameters'][0,0][0][0][0]),
    'gamma': float(best_svr_mat['best_parameters'][0,0][0][0][1])
  }

  best_lsvr_mat = scipy.io.loadmat(args.best_parameter+'_linearSVR.mat')
  best_params_linearSVR = {
    'C': float(best_lsvr_mat['best_parameters'][0,0][0][0][0]),
    'epsilon': float(best_lsvr_mat['best_parameters'][0,0][0][0][1])
  }

  metrics_svr, y_pred_svr, metrics_lsvr, y_pred_lsvr = evaluate_and_predict(X, y, best_params_SVR, best_params_linearSVR)

  print("SVR:", metrics_svr)
  print("LinearSVR:", metrics_lsvr)

  dir_path = os.path.dirname(args.predicted_score)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

  scipy.io.savemat(args.predicted_score+'_SVR.mat', {'predicted_score': y_pred_svr})
  scipy.io.savemat(args.predicted_score+'_linearSVR.mat', {'predicted_score': y_pred_lsvr})

if __name__ == '__main__':
  args = arg_parser()
  log_dir = os.path.dirname(args.log_file)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  sys.stdout = Logger(args.log_file)
  print(args)
  main(args)