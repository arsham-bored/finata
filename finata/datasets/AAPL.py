import yfinance as yf
from ..utils.preprocessing import *
from ..utils import scaler

container = 'AAPL/'
filename = 'result.pickle'

download_data = lambda: yf.download('AAPL','2016-01-01','2019-08-01')

def load_data(**arg) -> tuple:
    '''>>> return (train_data, test_data, generated_train, generated_test)'''

    sequence = arg.get('sequence', 100)
    batch_size = arg.get('batch_size', 50)

    df = download_data()['Close'].values
    df = df.reshape((-1, 1))

    train, test = split_train_test( df )

    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    generated_train, generated_test = time_series_generation(
        train, test,
        sequence=sequence,
        batch_size=batch_size
    )

    return (train, test, generated_train, generated_test)


if __name__ == "__main__":

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    data = download_data()
    data = data['Close'].values

    from matplotlib import pyplot as plt

    data = data.reshape((-1, 1))

    from ..utils import TimeseriesGenerator as TG

    split_percent = 0.80
    split = int(split_percent*len(data))

    train, test = data[:split], data[split:]

    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    look_back = 20

    train = TG(train, train, length=look_back, batch_size=20)     
    test_g = TG(test, test, length=look_back, batch_size=1)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential()
    model.add(
        LSTM(10,
            activation='relu',
            input_shape=(look_back,1),
             return_sequences=True
    ))
    model.add(Dropout(0.5)),
    model.add(
        LSTM(20,
            activation='relu',
             return_sequences=True
    ))
    model.add(Dropout(0.5)),
    model.add(
        LSTM(30,
            activation='relu',
            return_sequences=False
    ))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    num_epochs = 1
    model.fit_generator(train, epochs=20, verbose=1)

    prediction = model.predict_generator(test_g)

    #train = train.reshape((-1))
    test = test.reshape((-1))
    prediction = prediction.reshape((-1))

    plt.plot(test)
    plt.plot(prediction)

    plt.show()

"""

close_data = close_data.reshape((-1))

def predict(num_prediction, model):
    prediction_list = close_data[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list
    
def predict_dates(num_prediction):
    last_date = df['Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

num_prediction = 30
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)

"""