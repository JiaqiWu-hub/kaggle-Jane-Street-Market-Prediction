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

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args


# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]

                train_array = np.sort(np.unique(
                    np.concatenate((train_array,
                                    train_array_tmp)),
                    axis=None), axis=None)

            train_end = train_array.size

            for test_group_idx in unique_groups[group_test_start:
            group_test_start +
            group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                    np.concatenate((test_array,
                                    test_array_tmp)),
                    axis=None), axis=None)

            test_array = test_array[group_gap:]

            if self.verbose > 0:
                pass

            yield [int(i) for i in train_array], [int(i) for i in test_array]


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


def create_autoencoder(input_dim, output_dim, noise=0.05):
    i = Input(input_dim)
    encoded = BatchNormalization()(i)
    encoded = GaussianNoise(noise)(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    decoded = Dropout(0.2)(encoded)
    decoded = Dense(input_dim, name='decoded')(decoded)
    x = Dense(32, activation='relu')(decoded)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(output_dim, activation='sigmoid', name='label_output')(x)

    encoder = Model(inputs=i, outputs=encoded)
    autoencoder = Model(inputs=i, outputs=[decoded, x])

    autoencoder.compile(optimizer=Adam(0.005), loss={'decoded': 'mse', 'label_output': 'binary_crossentropy'})
    return autoencoder, encoder


def create_model(input_dim, output_dim, encoder):
    inputs = Input(input_dim)

    x = encoder(inputs)
    x = Concatenate()([x, inputs])  # use both raw and encoded features
    x = BatchNormalization()(x)
    x = Dropout(0.13)(x)

    hidden_units = [384, 896, 896, 394]
    for idx, hidden_unit in enumerate(hidden_units):
        x = Dense(hidden_unit)(x)
        x = BatchNormalization()(x)
        x = Lambda(tf.keras.activations.relu)(x)
        x = Dropout(0.25)(x)
    x = Dense(output_dim, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(0.0005), loss=BinaryCrossentropy(label_smoothing=0.05),
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

'''Defining and training the autoencoder.

We add gaussian noise with mean and std from training data. After training we lock the layers in the encoder from further training.'''

autoencoder, encoder = create_autoencoder(X.shape[-1],y.shape[-1],noise=0.1)
if TRAINING:
    autoencoder.fit(X,(X,y),
                    epochs=1000,
                    batch_size=4096,
                    validation_split=0.1,
                    callbacks=[EarlyStopping('val_loss',patience=10,restore_best_weights=True)])
    encoder.save_weights('./encoder.hdf5')
else:
    encoder.load_weights('../input/janestreet2021118mlp/encoder.hdf5')
encoder.trainable = False

'''Running CV

Following this notebook which use 5 PurgedGroupTimeSeriesSplit split on the dates in the training data.

We add the locked encoder as the first layer of the MLP. This seems to help in speeding up the submission rather than first predicting using the encoder then using the MLP.

We use a Baysian Optimizer to find the optimal HPs for out model. 20 trials take about 2 hours on GPU.'''

FOLDS = 4
SEED = 42

if TRAINING:
    gkf = PurgedGroupTimeSeriesSplit(n_splits = FOLDS, group_gap=20)
    splits = list(gkf.split(y, groups=train['date'].values))

    for fold, (train_indices, test_indices) in enumerate(splits):
        model = create_model(130, 5, encoder)
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=4096,callbacks=[EarlyStopping('val_auc',mode='max',patience=10,restore_best_weights=True)])
        model.save_weights(f'./model_{SEED}_{fold}.hdf5')
        model.compile(Adam(0.00001),loss='binary_crossentropy')
        model.fit(X_test,y_test,epochs=3,batch_size=4096)
        model.save_weights(f'./model_{SEED}_{fold}_finetune.hdf5')
else:
    models = []
    for f in range(FOLDS):
        model = create_model(130, 5, encoder)
        if USE_FINETUNE:
            model.load_weights(f'../input/janestreet2021118mlp/model_{SEED}_{f}_finetune.hdf5')
        else:
            model.load_weights(f'../input/janestreet2021118mlp/model_{SEED}_{f}.hdf5')
        models.append(model)

if not TRAINING:
    f = np.median
    models = models[-2:]
    import janestreet
    env = janestreet.make_env()
    # th = 0.503  # 9074.085
    # th = 0.495 9039.974
    th = 0.5
    for (test_df, pred_df) in tqdm(env.iter_test()):
        if test_df['weight'].item() > 0:
            x_tt = test_df.loc[:, features].values
            if np.isnan(x_tt[:, 1:].sum()):
                x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * f_mean
            pred = np.mean([model(x_tt, training = False).numpy() for model in models],axis=0)
            pred = f(pred)
            pred_df.action = np.where(pred >= th, 1, 0).astype(int)
        else:
            pred_df.action = 0
        env.predict(pred_df)