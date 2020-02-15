from finata.applications.bitcoin_application import App as bit_app
from dotenv import load_dotenv
import os

class Price():
    def __init__(self):

        # get env values ..
        config = load_dotenv()
        get = lambda name: os.getenv(name)

        # load bitcoin application
        bitcoin_model = get('bitcoin_model_path')
        self.bitcoin_model = bit_app( bitcoin_model )


    def bitcoin(self):
        return self.bitcoin_model.next(1)

    def historical_bitcoin(self):
        return self.bitcoin_model.predict(inverse=True)

