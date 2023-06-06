from eve_v2.model.model import create_model
from utils.mnist_util import load_mnist_dataset


def get_train_image():
    # Load 37800 train images and 4200 val images
    X_train, X_val, Y_train, Y_val, test = load_mnist_dataset()
    model = create_model()

def main():
    get_train_image()


if __name__ == '__main__':
    main()
