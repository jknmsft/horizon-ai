import datetime as dt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D, Flatten, LSTM,GRU, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from standalone.algo.utils.data import TimeSeriesTensor



class RecurrentNeuralNet():

    def __init__(self, lr, T, alpha, encoder_dim_1, encoder_dim_2, decoder_dim_1, decoder_dim_2, architecture='LSTM', optimizer='adam', loss='mae', num_stack_layers=1, units=100, dropout=0, hidden_dense_layers=[]):
        self.lr = lr
        self.T = T
        self.alpha = alpha
        self.encoder_dim_1 = encoder_dim_1
        self.encoder_dim_2 = encoder_dim_2
        self.decoder_dim_1 = decoder_dim_1
        self.decoder_dim_2 = decoder_dim_2
        self.architecture  = architecture
        self.optimizer = optimizer
        self.loss = loss
        self.num_stack_layers = num_stack_layers
        self.units = units
        self.dropout = dropout
        self.hidden_dense_layers = hidden_dense_layers
        #self.return_sequences = num_stack_layers>1

    def get_model(self, length, HORIZON=24, architecture='LSTM'):
        self.architecture = architecture
        return_sequences = self.num_stack_layers>1
        model = Sequential()
        if self.architecture == "LSTM":
            n_hours = self.T + HORIZON
            n_obs = n_hours*length - HORIZON + 1
            model.add(LSTM(units=100, input_shape=(self.T, length)))
            model.add(Dense(units=100, activation='relu', input_dim=n_obs))
            model.add(Dense(units=HORIZON))
            model.compile(loss=self.loss, optimizer=self.optimizer)
            #print("LSTM")
        if self.architecture == "GRU":
            if self.encoder_dim_2:
                model.add(
                    GRU(
                        units=self.encoder_dim_1,
                        input_shape=(self.T, length),
                        return_sequences=True,
                        kernel_regularizer=regularizers.l2(self.alpha),
                        bias_regularizer=regularizers.l2(self.alpha),
                    )
                )
                model.add(
                    GRU(
                        ubits=self.encoder_dim_2,
                        kernel_regularizer=regularizers.l2(self.alpha),
                        bias_regularizer=regularizers.l2(self.alpha),
                    )
                )
            else:
                model.add(
                    GRU(
                        units=self.encoder_dim_1,
                        input_shape=(self.T, length),
                        kernel_regularizer=regularizers.l2(self.alpha),
                        bias_regularizer=regularizers.l2(self.alpha),
                    )
                )

            model.add(RepeatVector(HORIZON))

            model.add(
                GRU(
                    units=self.decoder_dim_1,
                    return_sequences=True,
                    kernel_regularizer=regularizers.l2(self.alpha),
                    bias_regularizer=regularizers.l2(self.alpha),
                )
            )
            if self.decoder_dim_2:
                model.add(
                    GRU(
                        units=self.decoder_dim_2,
                        return_sequences=True,
                        kernel_regularizer=regularizers.l2(self.alpha),
                        bias_regularizer=regularizers.l2(self.alpha),
                    )
                )

            model.add(TimeDistributed(Dense(units=1)))
            model.add(Flatten())
            optimizer = RMSprop(lr=self.lr)
            model.compile(optimizer=optimizer, loss=self.loss)
        
        return  model

class CNN():
    def __init__(self, lr, T, alpha, latent_dim_1, latent_dim_2, kernel_size, padding='causal', activation='relu', dilation_rate=1):
        self.lr = lr
        self.T = T
        self.alpha = alpha
        self.latent_dim_1 = latent_dim_1
        self.latent_dim_2 = latent_dim_2
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dilation_rate = dilation_rate

    def get_model(self, length, HORIZON=24):
        model = Sequential()
        model.add(
            Conv1D(
                filters=self.latent_dim_1,
                kernel_size=self.kernel_size,
                padding=self.padding,
                activation=self.activation,
                dilation_rate=self.dilation_rate,
                input_shape=(self.T, length),
                kernel_regularizer=regularizers.l2(self.alpha),
                bias_regularizer=regularizers.l2(self.alpha),
            )
        )
        if self.latent_dim_2:
            model.add(
                Conv1D(
                    filters=self.latent_dim_2,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    activation=self.activation,
                    dilation_rate=2,
                    kernel_regularizer=regularizers.l2(self.alpha),
                    bias_regularizer=regularizers.l2(self.alpha),
                )
            )

        model.add(Flatten())
        model.add(
            Dense(
                units=HORIZON,
                activation='linear',
                kernel_regularizer=regularizers.l2(self.alpha),
                bias_regularizer=regularizers.l2(self.alpha),
            )
        )

        optimizer = RMSprop(lr=self.lr)
        model.compile(optimizer=optimizer, loss="mse")

        return model
    
    def create_input(self, data, T, HORIZON, cols=["consumption", "temperature"], valid_start_dt="2014-01-01 00:00:00", test_start_dt="2014-02-01 00:00:00"):
        # Get training data
        train = data.copy()[data.index < valid_start_dt][cols]

        # Normalize training data
        y_scaler = MinMaxScaler()
        y_scaler.fit(train[["consumption"]])
        X_scaler = MinMaxScaler()
        train[cols] = X_scaler.fit_transform(train)

        tensor_structure = {"X": (range(-T + 1, 1), cols)}
        train_inputs = TimeSeriesTensor(train, "consumption", HORIZON, tensor_structure)

        look_back_dt = dt.datetime.strptime(valid_start_dt, "%Y-%m-%d %H:%M:%S") - dt.timedelta(hours=T - 1)
        valid = data.copy()[(data.index >= look_back_dt) & (data.index < test_start_dt)][cols]
        valid[cols] = X_scaler.transform(valid)
        valid_inputs = TimeSeriesTensor(valid, "consumption", HORIZON, tensor_structure)

        look_back_dt = dt.datetime.strptime(test_start_dt, "%Y-%m-%d %H:%M:%S") - dt.timedelta(hours=T - 1)
        test = data.copy()[test_start_dt:][cols]
        test[cols] = X_scaler.transform(test)
        test_inputs = TimeSeriesTensor(test, "consumption", HORIZON, tensor_structure)

        return train_inputs, valid_inputs, test_inputs, y_scaler
    




