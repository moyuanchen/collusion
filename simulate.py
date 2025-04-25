from util import simulate
from config import Config


c = Config()

simulate(T = 1000, config = c, save_path='./simulated_data.pkl')

