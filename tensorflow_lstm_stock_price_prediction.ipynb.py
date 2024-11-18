import keras_tuner as kt
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def calculate_sma(column, window):
    # calculates the simple moving average of a given column.

    sma_column_name = f"{column.name}_sma_{window}"
    return column.rolling(window=window).mean().rename(sma_column_name)


# define start and end dates
start_date = '2022-01-01'
end_date = (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")

# define attribute tickers which are used as support sequences to help predict price of target tickers
attribute_tickers = ['^GSPC', 'USDPLN=X', 'CL=F', 'GC=F', 'ETFBW20TR.WA', 'BND', 'EEM',
                     '^HSI', '^BVSP', '^BSESN', 'EWW', 'ETFBTBSP.WA']
# define target tickers which price will be predicted by the models
target_tickers = ['ALE.WA', 'CDR.WA']
# create a map to store dataframes
df_map = {}

for ticker in attribute_tickers + target_tickers:
    df_map[ticker] = yf.download(ticker, start=start_date, end=end_date)

raw_df = pd.DataFrame()

# create new df with close price of each ticker only
for ticker, df in df_map.items():
    df = df[['Close']].copy()
    df.columns = [f'{ticker}_Close']
    raw_df = pd.merge(raw_df, df, how='outer', left_index=True, right_index=True)

# fill null values
raw_df = raw_df.ffill()
raw_df = raw_df.bfill()

# calculate the SMA for
for column in raw_df.columns:
    raw_df[f'{column}_sma'] = calculate_sma(raw_df[column], 60)

print(raw_df.info())
print(raw_df.describe())

# create df for modified data
modified_df = pd.DataFrame()
# calculate the percentage above the SMA for each ticker to capture similar trends despite stock price level
# also to help with normalization
for column in raw_df.columns:
    if 'Close' in column and '_sma' not in column:  # check if it's a closing price column and not already an SMA
        modified_df[f'{column}_percent_above_sma'] = ((raw_df[column] - raw_df[f'{column}_sma']) / raw_df[
            f'{column}_sma']) * 100

# calculate percentage change to capture similar trends despite stock price level
# also to help with normalization
for column in raw_df.columns:
    if 'Close' in column and '_sma' not in column:
        # Calculate percentage change
        modified_df[f'{column}_pct_change'] = raw_df[column].pct_change() * 100

# clip values in modified_df to the range [-10, 10] to help with normalization when model will be in use
modified_df = modified_df.clip(lower=-10, upper=10)


# normalize and scale linearly from -10 to 10 to 0 to 1
def normalize_and_scale(column):
    min_val = -10
    max_val = 10
    return (column - min_val) / (max_val - min_val)


modified_df = modified_df.apply(normalize_and_scale)

modified_df = modified_df.dropna()
print(modified_df.info())
print(modified_df.describe())

target_to_df_map = {}
# create separate dfs for each target ticker, without other target tickers data
for ticker in target_tickers:
    ticker_df = modified_df.copy()

    columns_to_drop = [ticker_col for ticker_col in ticker_df.columns if
                       any(target_ticker in ticker_col and ticker not in ticker_col for target_ticker in
                           target_tickers)]

    ticker_df = ticker_df.drop(columns_to_drop, axis=1)

    # store the dataframe in the map
    ticker_df.dropna(inplace=True)
    target_to_df_map[ticker] = ticker_df

for key, value in target_to_df_map.items():
    print(key)
    print(value.info())
    print(value.describe())


def make_sequences(df, target, horizon, seq_length):
    # create sequnces of seq_length length where horizon is defining how many timesteps should be
    # beetween last X datapoint and y
    X = []
    y = []
    for i in range(len(df) - seq_length - horizon + 1):
        X.append(df.iloc[i:i + seq_length].values)
        y.append(df[f'{target}_Close_pct_change'].iloc[i + seq_length + horizon - 1])
    return np.array(X), np.array(y)


def create_model(hp):
    model = Sequential()

    # define tunable hyper parameters
    hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
    hp_lstm_layers = hp.Int('lstm_layers', min_value=300, max_value=512, step=5)
    hp_dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])

    model.add(Input(shape=input_shape))

    for _ in range(hp_lstm_layers):
        model.add(LSTM(units=hp_units, return_sequences=True))  # return_sequences=True for all but the last LSTM layer
        model.add(Dropout(hp_dropout_rate))  # dropout layer for regularization
    model.add(LSTM(units=hp_units))  # last LSTM layer
    model.add(Dense(units=1))  # Output layer

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mse')
    return model


ticker_to_model = {}
# train the model
for target, df in target_to_df_map.items():
    X, y = make_sequences(df, target, 1, 60)
    # shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    print(f"Sequences created for {target}")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # define the EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    input_shape = (X.shape[1], X.shape[2])
    tuner = kt.Hyperband(create_model,
                         objective='val_loss',
                         max_epochs=10,
                         factor=3,
                         directory='keras_tuner',
                         project_name=target)
    tuner.search(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    print(tuner.get_best_hyperparameters()[0].values)
    best_model = tuner.get_best_models(num_models=1)[0]
    ticker_to_model[target] = best_model
