from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from tqdm import tqdm
from random import choices
import xgboost as xgb
import optuna
import gc

TRAINING = False
USE_FINETUNE = True
FOLDS = 4
SEED = 42

train = pd.read_csv('../input/jane-street-market-prediction/train.csv')
train = train.query('date > 85').reset_index(drop = True)
train = train.astype({c: np.float32 for c in train.select_dtypes(include='float64').columns}) #limit memory use
train.fillna(train.mean(),inplace=True)
train = train.query('weight > 0').reset_index(drop = True)
#train['action'] = (train['resp'] > 0).astype('int')
train['action'] =  (  (train['resp_1'] > 0 ) & (train['resp_2'] > 0 ) & (train['resp_3'] > 0 ) & (train['resp_4'] > 0 ) &  (train['resp'] > 0 )   ).astype('int')
features = [c for c in train.columns if 'feature' in c]

resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']

X = train[features].values
y = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T #Multitarget

f_mean = np.mean(train[features[1:]].values,axis=0)

SEED = 42
del train
rubbish = gc.collect()
param = {'learning_rate': 0.05,
          'max_depth': 9,
          'gamma': 6.617762097799007,
          'min_child_weight': 8.864374132546748,
          'subsample': 0.9,
          'colsample_bytree': 0.7,
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'tree_method': 'gpu_hist',
          'random_state': SEED,
         }

if TRAINING:
    for i in range(5):
        d_tr = xgb.DMatrix(X, y[:, i])
        clf = xgb.train(param, d_tr, 500)
        clf.save_model(f'./model_{SEED}_{i}.json')
        del d_tr
        del clf
        gc.collect()
        print(i)

else:
    models = []
    for i in range(5):
        bst = xgb.Booster({'nthread': 4})  # init model
        bst.load_model(f'../input/janestreet-xgbt-multitarget/model_{SEED}_{i}.json')  # load data
        models.append(bst)


if not TRAINING:
    f = np.median
    #models = models[-2:]
    import janestreet
    env = janestreet.make_env()
    # th = 0.6  # 全是0
    # th = 0.503  # score：7291.876
    # th = 0.5  # 7405.589
    # th = 0.495  #7884.875
    # th = 0.49 # 7607.603
    # th = 0.48 # 6702.828
    # th = 0.475   # 5711.415
    # th = 0.45   # 2906.051
    th = 0.4925
    for (test_df, pred_df) in tqdm(env.iter_test()):
        if test_df['weight'].item() > 0:
            x_tt = test_df.loc[:, features].values
            if np.isnan(x_tt[:, 1:].sum()):
                x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * f_mean
            d_tt = xgb.DMatrix(x_tt)
            pred = 0.
            for clf in models:
                pred += clf.predict(d_tt) / len(models)
            pred_df.action = np.where(pred >= th, 1, 0).astype(int)
        else:
            pred_df.action = 0
        env.predict(pred_df)