import pandas as pd
import numpy as np
from ...utils.preprocessing import *
from ...utils import scaler

def load_data(**arg):

    sequence = arg.get('sequence', 100)
    batch_size = arg.get('batch_size', 50)
    path = arg.get('path', 'data.csv')

    df = pd.read_csv(path)
    df = df.iloc[::-1]
    df = df['Price'].values

    df = [i.replace(',', '') for i in df]
    df = [float(price) for price in df]
    
    df = np.array(df).reshape(-1, 1) 

    train, test = split_train_test( df )

    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    generated_train, generated_test = time_series_generation(
        train, test,
        sequence=sequence,
        batch_size=batch_size
    )

    return (train, test, generated_train, generated_test)