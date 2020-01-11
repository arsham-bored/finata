import matplotlib as m
from matplotlib import pyplot as plt

m.use('Agg')

def plot_bitcoin_dataset(df, **arg):
        
        # we'll plot only high price
        high = df.High.values

        plt.plot( high )
        print('plottig ..')
        plt.savefig(arg['path'])