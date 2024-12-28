import matplotlib.pyplot as plt
import tensorflow as tf

from utils.mnist_util import get_input_train_example

IMG_PATH = "../eve_v1/train_images/mnist_example.jpg"

def visualize(model):
    layer_outputs = [layer.output for layer in model.layers[1:11]]
    activation_model = tf.keras.Model(inputs=model.inputs, outputs=layer_outputs)
    img_tensor = get_input_train_example()
    activations = activation_model.predict(img_tensor)

    for layer_activation in activations:
        print(layer_activation.shape)
        draw(layer_activation)

    # Getting Activations of first layer
    # first_layer_activation = activations[0]

    # shape of first layer activation
    # print(first_layer_activation.shape)
    # draw(first_layer_activation)
def draw(imgs, format=None):
    plt.figure(figsize=(20, 40))
    for i in range(min(imgs.shape[-1], 100)):
        plt_idx = i + 1
        plt.subplot(20, 16, plt_idx)
        plt.imshow(imgs[0, :, :, i], format)
    plt.show()
