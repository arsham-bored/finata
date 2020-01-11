from tensorflow import keras
from tensorflow.keras import layers

def simple_lstm_model ( x_train, y_train, **arg ) -> None:

    # hyperprameters
    input_shape = arg['input_shape']
    optimizer = arg['optimizer']
    loss = arg['loss']
    epochs = arg['epochs']

    # define model
    #Input = layers.Input(shape=input_shape)
    #model = layers.LSTM(20)(Input)
    #model = layers.LSTM(10)(model)
    #model = layers.Dense(model)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional

    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, input_shape=(10, 1), activation='relu')),
        Bidirectional(LSTM(50, return_sequences=True, activation='relu')),
        Bidirectional(LSTM(25, return_sequences=False, activation='relu')),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1, activation='softmax')
    ])

    # compile, fit, validate
    log_dir = arg['log_dir']
    save_to = arg['save_to']

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,  
          write_graph=True, write_images=True)

    model.compile( loss=loss, optimizer=optimizer, metrice=['accuracy'] )
    history = model.fit(y_train, x_train, epochs=epochs)

    model.save(save_to)




class LSTM():
    def __init__( be_bidirectional: bool = True, speed_matter: bool = True ):
        self.be_bidirectional = be_bidirectional
