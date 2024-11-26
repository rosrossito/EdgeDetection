import matplotlib.pyplot as plt

def visualize(model):
    # retrieve weights from the second hidden layer
    filters, biases = model.layers[1].get_weights()

    print(filters.shape)

    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    # plot first few filters
    n_filters = 12
    for i in range(n_filters):
        # get the filter
        plt_idx = i + 1
        plt.subplot(4, 3, plt_idx)
        plt.imshow(filters[:, :, 0, i], cmap='gray')

    # show the figure
    plt.show()