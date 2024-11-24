import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# The dimensions of our input image
img_width = 28
img_height = 28


def get_layer_features(model):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_name = "conv2d_1"
    layer_output = layer_dict[layer_name].output
    return layer_output

def draw(imgs, format=None, gray=True):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        plt_idx = i + 1
        plt.subplot(32, 16, plt_idx)
        plt.imshow(imgs[i][:, :, 0], format)
    plt.show()

def visualize(model):
    model.summary()
    layer_output = get_layer_features(model)
    feature_extractor = keras.Model(inputs=model.inputs, outputs=layer_output)

    """
    ## Visualize filters in the target layer
    """
    # Compute image inputs that maximize per-filter activations
    all_imgs = []
    for filter_index in range(layer_output.shape[3]):
        print("Processing filter %d" % (filter_index,))
        loss, img = visualize_filter(filter_index, feature_extractor)
        all_imgs.append(img)

    draw(all_imgs)

def compute_loss(input_image, filter_index, feature_extractor):
    activation = feature_extractor(input_image)
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)

@tf.function
def gradient_ascent_step(img, filter_index, learning_rate, feature_extractor):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index, feature_extractor)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img

def initialize_image():
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 1))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return img * 0.125

def visualize_filter(filter_index, feature_extractor):
    # We run gradient ascent for 30 steps
    iterations = 30
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate, feature_extractor)
    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img

def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")

    return img