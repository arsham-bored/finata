import matplotlib as m
from matplotlib import pyplot as plt

m.use('Agg')

def plot_linear_price(data: list, **arg):
        # if wanna save or only plot data
        save = arg['save']
        plt.plot( data )
        
        if save:
                
                print(f'save plot to .. {arg['path']}')
                plt.savefig(arg['path'])

        else:
                print('plotting ..')
                plt.show()