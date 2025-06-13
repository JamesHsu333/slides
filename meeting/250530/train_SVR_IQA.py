import pandas
import scipy.io
import numpy as np
import argparse
import time
import os, sys
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import GridSearchCV
import warnings
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
  parser.add_argument('--feature_file', type=str, required=True,
                      help='Path to .mat file containing feats_mat and mos')
  parser.add_argument('--best_parameter', type=str, required=True,
                      help='Output path to save best parameters')
  parser.add_argument('--log_file', type=str, default='logs/train.log',
                      help='Path to save training logs')
  parser.add_argument('--log_short', action='store_true',
                      help='Whether log short')
  parser.add_argument('--train_csv', type=str, required=True)
  args = parser.parse_args()
  return args

def evaluate_svr(X, y, log_short):
  print(f"Training SVR with {X.shape[1]}-dim features")
  param_grid = {'C': np.logspace(1, 10, 10, base=2),
                'gamma': np.logspace(-10, -6, 5, base=2)}
  grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=8, n_jobs=4, verbose=0)
  scaler = preprocessing.MinMaxScaler().fit(X)
  X = scaler.transform(X)
  print("Performing GridSearchCV for SVR...")
  grid.fit(X, y)
  return grid.best_params_

def evaluate_linear_svr(X, y, log_short):
  print(f"Training LinearSVR with {X.shape[1]}-dim features")
  param_grid = {'C': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.],
                'epsilon': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.]}
  grid = GridSearchCV(LinearSVR(random_state=1, max_iter=100), param_grid, n_jobs=4, cv=8, verbose=0)
  scaler = preprocessing.MinMaxScaler().fit(X)
  X = scaler.transform(X)
  print("Performing GridSearchCV for LinearSVR...")
  grid.fit(X, y)
  return grid.best_params_

def evaluate_ridge(X, y, log_short):
  print(f"Training Ridge Regression with {X.shape[1]}-dim features")
  alphas = np.logspace(-5, 5, 11)
  scaler = preprocessing.MinMaxScaler().fit(X)
  X = scaler.transform(X)
  ridge = RidgeCV(alphas=alphas, cv=8)
  ridge.fit(X, y)
  print(f"Best alpha for Ridge: {ridge.alpha_}")
  return {'alpha': ridge.alpha_}

def evaluate_lasso(X, y, log_short):
  print(f"Training Lasso Regression with {X.shape[1]}-dim features")
  alphas = np.logspace(-5, 5, 11)
  scaler = preprocessing.MinMaxScaler().fit(X)
  X = scaler.transform(X)
  lasso = LassoCV(alphas=alphas, cv=8)
  lasso.fit(X, y)
  print(f"Best alpha for Lasso: {lasso.alpha_}")
  return {'alpha': lasso.alpha_}

def evaluate_elastic_net(X, y, log_short):
  print(f"Training Elastic Net Regression with {X.shape[1]}-dim features")
  alphas = np.logspace(-5, 5, 11)
  scaler = preprocessing.MinMaxScaler().fit(X)
  X = scaler.transform(X)
  en = ElasticNetCV(alphas=alphas, cv=8)
  en.fit(X, y)
  print(f"Best alpha for Elastic Net: {en.alpha_}")
  return {'alpha': en.alpha_}

def main(args):
  print("Loading features and metadata from:", args.feature_file)
  mat = scipy.io.loadmat(args.feature_file)
  X_all = np.asarray(mat['feats_mat'], dtype=np.float32)
  raw_names = mat['image_names'].squeeze()
  mat_names = [n.strip().lower() for n in raw_names]

  print("Loading MOS scores from:", args.train_csv)
  df = pandas.read_csv(args.train_csv)
  df['image_name'] = df['image_name'].str.strip()
  matched_rows = df[df['image_name'].isin(mat_names)]

  if matched_rows.empty:
    raise ValueError("No matching images between .mat file and CSV file.")

  X = []
  y = []
  for name in matched_rows['image_name']:
    idx = mat_names.index(name)
    X.append(X_all[idx])
  X = np.array(X)
  y = matched_rows['score'].astype(np.float32).to_numpy()

  X[np.isinf(X)] = np.nan
  X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)

  t_start = time.time()
  best_params_svr = evaluate_svr(X, y, args.log_short)
  best_params_linear = evaluate_linear_svr(X, y, args.log_short)
  best_params_ridge = evaluate_ridge(X, y, args.log_short)
  best_params_lasso = evaluate_lasso(X, y, args.log_short)
  best_params_elastic_net = evaluate_elastic_net(X, y, args.log_short)
  print('Total training time: {:.2f} sec'.format(time.time() - t_start))

  os.makedirs(os.path.dirname(args.best_parameter), exist_ok=True)
  scipy.io.savemat(args.best_parameter + '_SVR.mat', {
    'best_parameters': np.asarray(best_params_svr, dtype=object)
  })
  scipy.io.savemat(args.best_parameter + '_linearSVR.mat', {
    'best_parameters': np.asarray(best_params_linear, dtype=object)
  })
  scipy.io.savemat(args.best_parameter + '_Ridge.mat', {
    'best_parameters': np.asarray(best_params_ridge, dtype=object)
  })
  scipy.io.savemat(args.best_parameter + '_Lasso.mat', {
    'best_parameters': np.asarray(best_params_lasso, dtype=object)
  })
  scipy.io.savemat(args.best_parameter + '_Elastic_Net.mat', {
    'best_parameters': np.asarray(best_params_elastic_net, dtype=object)
  })
  print(f"Saved best parameters to {args.best_parameter}_*.mat")

if __name__ == '__main__':
  args = arg_parser() 
  sys.stdout = Logger(args.log_file)
  print(args)
  main(args)