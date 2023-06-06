import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

def get_MNIST_train_example():
    X_train, _, _, _, _ = load_mnist_dataset()
    return X_train[0][:, :, 0]

def load_mnist_dataset():
    np.random.seed(2)
    train = pd.read_csv("input/train.csv")
    test = pd.read_csv("input/test.csv")
    Y_train = train["label"]

    # Drop 'label' column
    X_train = train.drop(labels=["label"], axis=1)

    # free some space
    del train

    # sns.countplot(Y_train)
    # plt.show()

    Y_train.value_counts()
    # Check the data
    X_train.isnull().any().describe()
    test.isnull().any().describe()

    # Normalize the data
    X_train = X_train / 255.0
    test = test / 255.0

    # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
    X_train = X_train.values.reshape(-1, 28, 28, 1)
    test = test.values.reshape(-1, 28, 28, 1)

    # Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    Y_train = to_categorical(Y_train, num_classes=10)

    # Set the random seed
    random_seed = 2
    # Split the train and the validation set for the fitting
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)
    # Some examples
    # plt.imshow(X_train[0][:, :, 0])
    # plt.show()

    return X_train, X_val, Y_train, Y_val, test
