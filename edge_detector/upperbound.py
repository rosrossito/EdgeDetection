# как улучшить
# 0.отрисовать фильтры - done
# 1.уйти от смесей?
# 2.едж детектор не из коробки
# 3.отрезать поле предварительно - done
# 4.добавить цвет! - !!
# 5.добавить кернелы - тогда можно повысить порог - done
# 6.убирать шум - трибуны и тени, зная что у нас должна быть форма близкая к прямоугольнику

# find upper/lower bound
# if key == 159:
#     for edge, kernel in kernelEdgeBank[key]:
#         if edge < 27 and edge > 21:
#             max_result = 0
#             max_edge = 0
#             output_conv = np.zeros((round(iH / 8), round(iW / 8)), dtype="float32")
#             kernel_out = kernel[~(kernel == 0).all(1)]
#             (iHs, iWs) = output_conv.shape[:2]
#             (iHk, iWk) = kernel_out.shape[:2]
#             print("Edge: " + str(edge) + ", kernel_sum: " + str(np.ndarray.sum(kernel)))
#
#             # output_kernel(kernel_out, edge)
#
#             for n in range(0, iHs):
#
#                 if n + iHk < iHs:
#                     roi_image = output[n:n+iHk, 0:iWk]
#                     mask  =  np.multiply(kernel_out, roi_image)
#                     result = sum((sum(mask)))
#
#                     max_result, max_edge = get_max_result(result, edge, max_result, max_edge)
#                     percentage = round(result / 255 / np.ndarray.sum(kernel) * 100)
#
#                     if percentage > 65:
#                         print("Bingo! " + "Edge: " + str(edge) + ", column: " + str(n) + " - " + str(result) +
#                                   ", active pixels: " + str(result/255) + ", percentage: " + str(percentage))
#                         output_conv[n:n+iHk, 0:iWk] = cv2.add(mask, kernel_out.astype('float64')*255)
#
#             output_conv = cv2.copyMakeBorder(output_conv, 0, 0, 45, 45, cv2.BORDER_CONSTANT)
#             output_conv_total = cv2.add(output_conv_total, output_conv)
#
#             print("Maximum. " + "Edge: " + str(edge) + " - " + str(max_result) +
#                   ", active pixels: " + str(max_result / 255) + ", percentage: " + str(
#                 round(max_result / 255 / np.ndarray.sum(kernel) * 100)))


# blur
# smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
# largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
# gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
# image_rescaled = cv2.filter2D(output_conv_total[:,45:205], -1, smallBlur)

# output_conv_total = output_conv_total[:, 45:205]
# output_conv_total[output_conv_total == 255] = 255
# moreSmoothenedImage = image.filter(ImageFilter.SMOOTH_MORE)


#
# def convolve_with_custom_kernel(output, output_conv):
#     kernelEdgeBank = create_edge_kernels()
#     for key in kernelEdgeBank:
#         if key == 89:
#             for edge, kernel in kernelEdgeBank[key]:
#                 print(str(edge))
#                 print(np.ndarray.sum(kernel))
#                 opencvOutput = cv2.filter2D(output, -1, kernel / np.ndarray.sum(kernel), borderType=cv2.BORDER_ISOLATED)
#                 opencvOutput[opencvOutput < 180] = 0
#                 output_conv = cv2.add(opencvOutput, output_conv)
#                 # frame = cv2.filter2D(frame, cv2.CV_32F, log_kernel)
#                 # frame *= 255
#                 # # remove near 0 floats
#                 # frame[frame < 0] = 0
#     return output_conv