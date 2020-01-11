from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import requests as r
import os

from ..utils.preprocessing import convert_btc_data_to_df

import pickle

cache_path = '../cache'
url = 'https://min-api.cryptocompare.com/data/v2/histohour?fsym=BTC&tsym=USD'

total_data = []

if not os.path.isdir(cache_path):
    os.mkdir(cache_path)

splitted = lambda data: train_test_split(data)

def request_data(url):
    
    print('start downloading ..')

    timestamp = int(datetime.now().timestamp())
    request_limit = 2000

    _url = lambda url, limit, timestamp: f'{url}&limit={limit}&toTs={timestamp}'
    request = lambda url: r.get(url).json()

    while True:
        generated_url = _url( url, request_limit, timestamp )
        data = request(generated_url)

        data = data['Data']
        new_time = data['TimeFrom']
        timestamp = new_time

        data = data['Data']

        if data[0]['high'] != 0:
            total_data.extend(data)
        else:
            break

    print('donwload completed .')
    return total_data

def save(data):
    path = f'{cache_path}/{datetime.now().timestamp()}.pickle'

    with open(path, 'wb') as file:
        print(f'saving data to {path}')
        pickle.dump(data, file)

def load(path):

    dates = []

    for file in os.listdir(path):
        file_name = file.split('.pickle')[0]
        dates.append(file_name)

    newest_file = sorted(dates, key=lambda d: map(int, d.split('-')))[0]

    print(f'importing {newest_file} ...')
    with open(f'{cache_path}/{newest_file}.pickle', 'rb') as file:
        data = pickle.load(file)
        
    return data

def load_data():

    data_exist = os.listdir(cache_path) != []

    if data_exist:
        data = load(cache_path)

        return splitted(data)

    else:
        data = request_data(url)
        data = save(data)

        return splitted(data)


def load_as_DataFrame():

    # loading data
    x_train, x_test = load_data()

    # convert data to data frame
    df = convert_btc_data_to_df(x_train)
    #df_test = convert_btc_data_to_df(x_test)

    # convert time stamp to readable date time
    date = lambda time : datetime.fromtimestamp(time).ctime()

    # add readable time date to columns
    #df['ctime'] = df.Time.apply(lambda time: date(time))
    #df_test['ctime'] = df_test.Time.apply(lambda time: date(time))

    # sort data by date (up to year, 2020)
    df.set_index('Time', drop=True, append=False, inplace=True, verify_integrity=False)
    #df_test.set_index('Time', drop=True, append=False, inplace=True, verify_integrity=False)

    df = df.sort_index()
    #df_test = df_test.sort_index()

    df_test = df.iloc[50000:]
    df = df.iloc[:50000]

    return (df, df_test)

if __name__ == "__main__":
    load_as_DataFrame()
