import glob
import json
import pickle
import calendar
import holidays
import numpy as np
import pandas as pd
from pandas import concat
from pandas import DataFrame
from datetime import datetime
from collections import UserDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extract day of year and time of day features from a timestamp
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Timestamps must be in index
        field = X.index
        X["time_of_day"] = field.hour + field.minute / 60
        X["day_of_year"] = field.dayofyear

        return X


class CyclicalDateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Make cyclically encoded day of year and time of day features
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Apply formula for sin and cosine
        X["sin_time_of_day"], X["cos_time_of_day"] = _cyclical_encoding(
            X["time_of_day"], period=24)

        #NOTE: period=366?
        X["sin_day_of_year"], X["cos_day_of_year"] = _cyclical_encoding(
            X["day_of_year"], period=366)
        return X

def _cyclical_encoding(series, period):
    """
    Cyclical encoding of a series with a specified period
    """
    # Basic formula for sin/cosine equation
    base = 2 * np.pi * series / period

    return np.sin(base), np.cos(base)

class HolidaysFeatures(BaseEstimator, TransformerMixin):
    """
    Encode holidays
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        #feries = holidays.Canada(prov='QC') 
        holidays_qc = holidays.Canada(prov='QC')
        df = pd.DataFrame()
        field = X.index
        df['date'] = field.date
        df['date'] = pd.to_datetime(df['date'])
        #X['holyday'] = df['date'].isin(feries)
        X['holiday'] = df['date'].isin(holidays_qc)

        return X

#NOTE: Look at this piece of code
def ajouter_jours_feries(donnees_calendaires, types_donnees):
    r"""Ajoute un indicateur 0-1 de date fériée

    >>> echantillon = preparer_donnees_brutes('test/echantillon_3.csv')
    Ajout de 50 dates manquantes
    >>> echantillon2 = echantillon.copy()
    >>> echantillon.shape, echantillon2.shape
    ((53, 29), (53, 29))
    >>> ajouter_jours_feries(echantillon, echantillon2, ['ferie'])
    1
    >>> echantillon.shape, echantillon2.shape
    ((53, 30), (53, 29))
    """
    if 'ferie' not in types_donnees:
        return 0

    feries = holidays.Canada(prov='QC')   
    #donnees['ferie'] = donnees_calendaires['date'].apply(lambda date: 1 if date in feries else 0)
    
    return 1

def transformer_donnees_onehot(x, types_donnees_onehot):
    colonnes_onehot = [c for c in x.columns if c in types_donnees_onehot]
    
    # transformation des données one hot
    onehot = OneHotEncoder(sparse=False, categories='auto')
    x_onehot = onehot.fit_transform(x[colonnes_onehot])
    
    # normalisation pour les autres données standard
    min_norm = -1
    max_norm = 1
    x_onehot = np.where(np.equal(x_onehot, 0), min_norm, max_norm)

    # construction du nouveau dataframe
    x_onehot_df = pd.DataFrame(index=x.index, data=x_onehot, columns=[f'onehot_{i}' for i in range(x_onehot.shape[1])])
    x_transforme = pd.concat([x.drop(columns=colonnes_onehot), x_onehot_df], axis=1)
    
    # on renvoie aussi le nombre de variables créées, pour la normalisation
    return x_transforme, x_onehot.shape[1] 

    # convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

class TimeSeriesTensor(UserDict):
    """A dictionary of tensors for input into the RNN model.
    
    Use this class to:
      1. Shift the values of the time series to create a Pandas dataframe containing all the data
         for a single training example
      2. Discard any samples with missing values
      3. Transform this Pandas dataframe into a numpy array of shape 
         (samples, time steps, features) for input into Keras

    The class takes the following parameters:
       - **dataset**: original time series
       - **target** name of the target column
       - **H**: the forecast horizon
       - **tensor_structures**: a dictionary discribing the tensor structure of the form
             { 'tensor_name' : (range(max_backward_shift, max_forward_shift), [feature, feature, ...] ) }
             if features are non-sequential and should not be shifted, use the form
             { 'tensor_name' : (None, [feature, feature, ...])}
       - **freq**: time series frequency (default 'H' - hourly)
       - **drop_incomplete**: (Boolean) whether to drop incomplete samples (default True)
    """
    
    def __init__(self, dataset, target, H, tensor_structure, freq='H', drop_incomplete=True):
        self.dataset = dataset
        self.target = target
        self.tensor_structure = tensor_structure
        self.tensor_names = list(tensor_structure.keys())
        
        self.dataframe = self._shift_data(H, freq, drop_incomplete)
        self.data = self._df2tensors(self.dataframe)
    
    def _shift_data(self, H, freq, drop_incomplete):
        
        # Use the tensor_structures definitions to shift the features in the original dataset.
        # The result is a Pandas dataframe with multi-index columns in the hierarchy
        #     tensor - the name of the input tensor
        #     feature - the input feature to be shifted
        #     time step - the time step for the RNN in which the data is input. These labels
        #         are centred on time t. the forecast creation time
        df = self.dataset.copy()
        
        idx_tuples = []
        for t in range(1, H+1):
            df['t+'+str(t)] = df[self.target].shift(t*-1, freq=freq)
            idx_tuples.append(('target', 'y', 't+'+str(t)))

        for name, structure in self.tensor_structure.items():
            rng = structure[0]
            dataset_cols = structure[1]
            
            for col in dataset_cols:
            
            # do not shift non-sequential 'static' features
                if rng is None:
                    df['context_'+col] = df[col]
                    idx_tuples.append((name, col, 'static'))

                else:
                    for t in rng:
                        sign = '+' if t > 0 else ''
                        shift = str(t) if t != 0 else ''
                        period = 't'+sign+shift
                        shifted_col = name+'_'+col+'_'+period
                        df[shifted_col] = df[col].shift(t*-1, freq=freq)
                        idx_tuples.append((name, col, period))
                
        df = df.drop(self.dataset.columns, axis=1)
        idx = pd.MultiIndex.from_tuples(idx_tuples, names=['tensor', 'feature', 'time step'])
        df.columns = idx

        if drop_incomplete:
            df = df.dropna(how='any')

        return df
    
    def _df2tensors(self, dataframe):
        
        # Transform the shifted Pandas dataframe into the multidimensional numpy arrays. These
        # arrays can be used to input into the keras model and can be accessed by tensor name.
        # For example, for a TimeSeriesTensor object named "model_inputs" and a tensor named
        # "target", the input tensor can be acccessed with model_inputs['target']
    
        inputs = {}
        y = dataframe['target']
        y = y.values
        inputs['target'] = y

        for name, structure in self.tensor_structure.items():
            rng = structure[0]
            cols = structure[1]
            tensor = dataframe[name][cols].values
            if rng is None:
                tensor = tensor.reshape(tensor.shape[0], len(cols))
            else:
                tensor = tensor.reshape(tensor.shape[0], len(cols), len(rng))
                tensor = np.transpose(tensor, axes=[0, 2, 1])
            inputs[name] = tensor

        return inputs
       
    def subset_data(self, new_dataframe):
        
        # Use this function to recreate the input tensors if the shifted dataframe
        # has been filtered.
        
        self.dataframe = new_dataframe
        self.data = self._df2tensors(self.dataframe)