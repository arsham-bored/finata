from finata.datasets import bitcoin
from finata.utils.plots import plot_bitcoin_dataset
from finata.utils.plots import plot_bitcoin_dataset

# load data, lates 10'000 samples for test data.
df, df_test = bitcoin.load_as_DataFrame()
# save plot as image:
#plot_bitcoin_dataset(df, path='./result.png')

#from finata.brain.lstm_sample import simple_lstm_model

from finata.utils.preprocessing import transform_data
from finata.brain.lstm_sample import simple_lstm_model

x, y = transform_data(df[['High']])

model = simple_lstm_model(
    x, y,
    input_shape=(10, 1),
    optimizer='adam',
    loss= 'mae',
    epochs=10,
    log_dir='./result',
    save_to='./result.h5'
)

real_data = df_test[['High']]
predictions = model.predict(real_data)

