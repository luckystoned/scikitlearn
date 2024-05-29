from utils import Utils
from models import Models

if __name__ == "__main__":

    utils = Utils()
    models = Models()

    data = utils.load_from_csv()
    x, y = utils.feature_target(data, ['country', 'rank', 'score'], 'score')

    models.grid_training(x, y)

    # print(data.head(5))



