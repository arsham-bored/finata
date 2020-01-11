import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def convert_btc_data_to_df(data):
    time, hight, low = [], [], []
    columns = ['Time', 'High', 'Low']

    for obj in data:
        time.append( obj['time'] )
        hight.append( obj['high'] )
        low.append( obj['low'] )

    data = { 'Time': time, 'High': hight, 'Low': low }
    df = pd.DataFrame(data, columns=columns)

    return df

scaler = MinMaxScaler()
scaled_data = lambda df: scaler.fit_transform(df)
inverse_scaled_data = lambda data: scaler.inverse_transform(data)

def turn_into_sequence (data: list, sequence_length: int):
    x_data, y_data = [], []

    for item in range(sequence_length, len(data)):
        x_item = data[item]
        y_item = data[item - sequence_length : item]
        x_data.append(x_item)
        y_data.append(y_item)

    return x_data, y_data

def transform_data( df, sequence_length = 10 ):

    print('converting df data into min max scale..')

    # format prices into mix-max scaler
    data = scaled_data(df)
    # turn them into sequence of x and y
    x_data, y_data = turn_into_sequence(data, sequence_length)

    return np.array(x_data), np.array(y_data)

if __name__ == "__main__":
    pass

    