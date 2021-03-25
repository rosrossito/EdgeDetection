import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize and reshape images to [28,28,1]
train_images = np.expand_dims(train_images.astype(np.float32) / 255.0, axis=3)
test_images = np.expand_dims(test_images.astype(np.float32) / 255.0, axis=3)

plt.figure(dpi=100)
plt.imshow(np.squeeze(train_images[0]), cmap='gray')
plt.title('Number: {}'.format(train_labels[0]))
plt.show()



with tf.compat.v1.Session() as sess:
    batch_size = 128
    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    dataset = dataset.batch(batch_size)
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    # iterator = dataset.repeat().batch(batch_size).make_initializable_iterator()
    data_batch = iterator.get_next()

    sess.run(iterator.initializer)

    batch_images, batch_labels = sess.run(data_batch)

    print('Images shape: {}'.format(batch_images.shape))
    print('Labels shape: {}'.format(batch_labels.shape))

    # Get the first batch of images and display first image
    batch_images, batch_labels = sess.run(data_batch)
    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(batch_images)[0], cmap='gray')

    # Get a second batch of images and display first image
    batch_images, batch_labels = sess.run(data_batch)
    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(batch_images)[0], cmap='gray')
    plt.show()