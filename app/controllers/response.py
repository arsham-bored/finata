from .Price import Price
import logging

class Response():
    def __init__(self, request):
        self.Price = Price()
        self.request = request

    def home(self):
        return 'not implemented yet'

    def bitcoin(self):
        logging.warn("tommorows's bitcoin price requested")
        data = [i[0] for i in self.Price.bitcoin()]
        return {'price': list(data)}

    def historical_bitcoin(self):
        logging.warn('historical data requested')

        prediction = self.Price.historical_bitcoin()
        prediction = [float(i[0]) for i in prediction]

        return {
            'prediction': list(prediction),
            'real': self.Price.bitcoin_model.pure_data[100:]
        }