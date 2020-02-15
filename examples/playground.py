from finata.datasets import Bitcoin
from finata.brain.LSTM import LSTM_net
import numpy as np

train, test, gen_train, gen_test = \
Bitcoin.load_data(sequence=100, batch_size=1)


lstm = LSTM_net(input_shape=(100, 1), epochs=1)
lstm.train_on_sample_model()
lstm.train(gen_train)

y_predic = lstm.predict(gen_test)
y_predic = np.array(y_predic)


from finata.utils.preprocessing import flatten

y_real, y_predic = flatten(gen_test, y_predic)

from finata.utils.graph import plot

plot(
    [
        (y_real, 'real data'),
        (y_predic, 'predicted data')
    ],
    save='result.png'
)
