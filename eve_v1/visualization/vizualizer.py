from math import ceil

import matplotlib.pyplot as plt
import numpy as np

from eve_v1.visualization.conversion import get_coord, convert_third_level_feature, convert_second_level_feature, \
    convert_first_level_feature


def viz_layer(layer, n_filters=4):
    for k in range(ceil(n_filters / 4)):
        fig = plt.figure(figsize=(20, 20))
        end = n_filters if k * 4 + 4 > n_filters else k * 4 + 4
        for i in range(k * 4, end):
            ax = fig.add_subplot(1, end, i + 1)
            ax.imshow(np.squeeze(layer[0, i].data.numpy()), cmap='gray')
            ax.set_title('Output %s' % str(i + 1))
        plt.show()


def viz_filter(filters):
    # Check the Filters
    fig = plt.figure(figsize=(10, 5))
    for i in range(len(filters)):
        ax = fig.add_subplot(1, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(filters[i], cmap='gray')
        ax.set_title('Filter %s' % str(i + 1))
        width, height = filters[i].shape
        for x in range(width):
            for y in range(height):
                ax.annotate(str(filters[i][x][y]), xy=(y, x),
                            color='white' if filters[i][x][y] < 0 else 'black')
    plt.show()

    # fig = plt.figure(figsize=(12, 6))
    # fig.subplots_adjust(left=0, right=1.5, bottom=0.8, top=1, hspace=0.05, wspace=0.05)
    # for i in range(len(filters)):
    #     ax = fig.add_subplot(1, 4, i + 1, xticks=[], yticks=[])
    #     ax.imshow(filters[i], cmap='gray')
    #     ax.set_title('Filter %s' % str(i + 1))
    # plt.show()


def get_pixel_value_pic(img):
    fig = plt.figure(figsize=(20, 20))
    draw_kernels(fig, img)
    plt.show()


def draw_kernels(fig, img, n_filters=1, i=0):
    ax = fig.add_subplot(2, n_filters, i + 1)
    ax.imshow(img, cmap='gray')
    ax.set_title('Feature number: %s' % str((img > 0).sum()))

    width, height = img.shape
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y], 2) if img[x][y] != 0 else 0
            ax.annotate(str(val), xy=(y, x),
                        color='white' if img[x][y] < thresh else 'black', fontsize=6, ha='center', va='center')


# def get_pixel_value_layer(layer, icons, n_filters=4):
#     total_feature_number = 0
#     for k in range(ceil(n_filters / 4)):
#         fig = plt.figure(figsize=(20, 20))
#         end = n_filters if k * 4 + 4 > n_filters else k * 4 + 4
#         for i in range(k * 4, end):
#             img = layer[0, i].data.numpy()
#             draw_kernels(fig, img, 4, i - k * 4)
#
#             # draw feature
#             ax_1 = fig.add_subplot(2, 4, 5)
#             ax_1.imshow(icons[k])
#
#             total_feature_number = total_feature_number + (img > 0).sum()
#         plt.show()
#     print("Total feature number in layer: " + str(total_feature_number))


def get_pixel_value_layer_with_icon(layer, icons, n_filters=4):
    total_feature_number = 0
    for k in range(ceil(n_filters / 4)):
        fig = plt.figure(figsize=(20, 20))
        end = n_filters if k * 4 + 4 > n_filters else k * 4 + 4
        for i in range(k * 4, end):
            img = layer[0, i].data.numpy()
            draw_kernels_with_icon(fig, img, icons[i], 4, i - k * 4)
            total_feature_number = total_feature_number + (img > 0).sum()
    plt.show(block=True)
    print("Total feature number in layer: " + str(total_feature_number))


def draw_kernels_with_icon(fig, img, icon, n_filters=1, i=0):
    ax = fig.add_subplot(2, n_filters, i + 1)
    ax.imshow(img, cmap='gray')
    ax.set_title('Feature number: %s' % str((img > 0).sum()))

    width, height = img.shape
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y], 2) if img[x][y] != 0 else 0
            ax.annotate(str(val), xy=(y, x),
                        color='white' if img[x][y] < thresh else 'black', fontsize=6, ha='center', va='center')

    ax_1 = fig.add_subplot(2, n_filters, n_filters + i + 1)
    ax_1.imshow(icon)


def get_total_picture(layer):
    depth_img, height, width = layer[0].shape
    img = np.zeros((height, width))
    for feature in layer.detach().numpy()[0][0:depth_img]:
        img = np.add(img, feature)
    get_pixel_value_pic(img)


def get_converted_picture(binary_conv_layer, manually_created_features, second_level_manually_created_features):
    coords = get_coord(binary_conv_layer)
    img = np.zeros((17, 17), dtype="float32")
    next_level_conversion_data = convert_third_level_feature(coords, manually_created_features)
    next_level_conversion_data = convert_second_level_feature(next_level_conversion_data,
                                                              second_level_manually_created_features)
    pixels_coords = convert_first_level_feature(next_level_conversion_data)
    # raw, column
    for pixels_coord in pixels_coords:
        img[pixels_coord[1], pixels_coord[0]] = 1
    plt.imshow(img, cmap='gray')
    plt.show()

def get_converted_picture_first_layer(binary_conv_layer):
    input = []
    counter = 1
    coords = get_coord(binary_conv_layer)
    img = np.zeros((20, 20), dtype="float32")
    for coord_raw in coords:
        for coord in coord_raw:
            input.append([coord, counter])
        counter = counter + 1
    pixels_coords = convert_first_level_feature(input)
    for pixels_coord in pixels_coords:
        img[pixels_coord[1], pixels_coord[0]] = 1
    plt.imshow(img, cmap='gray')
    plt.show()
