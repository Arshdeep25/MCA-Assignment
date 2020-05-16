import time
import cv2
import numpy as np
from PIL import Image


def returnCorrelogram(image):
    map_color_freq = dict()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            array_to_tupple = (image[i, j, 0], image[i, j, 1], image[i, j, 2])
            distance = [1, 3, 5, 7]
            count_same_color_list = []
            for d in distance:
                count_same_color = 0
                for y in range(d):
                    if i + d < image.shape[0] and j + y < image.shape[1]:
                        color_tupple = (image[i + d, j + y, 0], image[i + d, j + y, 1], image[i + d, j + y, 2])
                        if color_tupple == array_to_tupple:
                            count_same_color += 1
                    if i + d < image.shape[0] and j - y > -1 and y != 0:
                        color_tupple = (image[i + d, j - y, 0], image[i + d, j - y, 1], image[i + d, j - y, 2])
                        if color_tupple == array_to_tupple:
                            count_same_color += 1
                    if i - d > -1 and j + y < image.shape[1]:
                        color_tupple = (image[i - d, j + y, 0], image[i - d, j + y, 1], image[i - d, j + y, 2])
                        if color_tupple == array_to_tupple:
                            count_same_color += 1
                    if i - d > -1 and j - y > -1 and y != 0:
                        color_tupple = (image[i - d, j - y, 0], image[i - d, j - y, 1], image[i - d, j - y, 2])
                        if color_tupple == array_to_tupple:
                            count_same_color += 1
                for x in range(d):
                    if i + x < image.shape[0] and j + d < image.shape[1]:
                        color_tupple = (image[i + x, j + d, 0], image[i + x, j + d, 1], image[i + x, j + d, 2])
                        if color_tupple == array_to_tupple:
                            count_same_color += 1
                    if i - x > -1 and j + d < image.shape[1] and x != 0:
                        color_tupple = (image[i - x, j + d, 0], image[i - x, j + d, 1], image[i - x, j + d, 2])
                        if color_tupple == array_to_tupple:
                            count_same_color += 1
                    if i + x < image.shape[0] and j - d > -1:
                        color_tupple = (image[i + x, j - d, 0], image[i + x, j - d, 1], image[i + x, j - d, 2])
                        if color_tupple == array_to_tupple:
                            count_same_color += 1
                    if i - x > -1 and j - d > -1 and x != 0:
                        color_tupple = (image[i - x, j - d, 0], image[i - x, j - d, 1], image[i - x, j - d, 2])
                        if color_tupple == array_to_tupple:
                            count_same_color += 1
                if i + d < image.shape[0] and j + d < image.shape[1]:
                    color_tupple = (image[i + d, j + d, 0], image[i + d, j + d, 1], image[i + d, j + d, 2])
                    if color_tupple == array_to_tupple:
                        count_same_color += 1
                if i + d < image.shape[0] and j - d > -1:
                    color_tupple = (image[i + d, j - d, 0], image[i + d, j - d, 1], image[i + d, j - d, 2])
                    if color_tupple == array_to_tupple:
                        count_same_color += 1
                if i - d > -1 and j + d < image.shape[1]:
                    color_tupple = (image[i - d, j + d, 0], image[i - d, j + d, 1], image[i - d, j + d, 2])
                    if color_tupple == array_to_tupple:
                        count_same_color += 1
                if i - d > -1 and j - d > -1:
                    color_tupple = (image[i - d, j - d, 0], image[i - d, j - d, 1], image[i - d, j - d, 2])
                    if color_tupple == array_to_tupple:
                        count_same_color += 1
                count_same_color_list.append(count_same_color / (8 * d))
            if array_to_tupple not in map_color_freq:
                map_color_freq[array_to_tupple] = [count_same_color_list]
            else:
                map_color_freq[array_to_tupple].append(count_same_color_list)
    for key in map_color_freq:
      map_color_freq[key] = np.mean(np.array(map_color_freq[key]), axis=0)
    return map_color_freq


scale_percent = 15.0

time_1 = time.time()
image_1 = np.array(Image.open('images/all_souls_000000.jpg'))
dim = (int(image_1.shape[1] * scale_percent/100), int(image_1.shape[0] * scale_percent/100))
image_1 = cv2.resize(image_1, dim)
map_color_freq_1 = returnCorrelogram(image_1)
print("Image 1 ", time.time() - time_1)

time_2 = time.time()
image_2 = np.array(Image.open('images/all_souls_000001.jpg'))
dim = (int(image_2.shape[1] * scale_percent/100), int(image_2.shape[0] * scale_percent/100))
image_2 = cv2.resize(image_2, dim)
map_color_freq_2 = returnCorrelogram(image_2)
print("Image 2", time.time() - time_2)

time_3 = time.time()
denominator = 1
numerator = 0

map_color_freq = dict()
for key in map_color_freq_1:
    map_color_freq[key] = [map_color_freq_1[key]]

for key in map_color_freq_2:
    if key in map_color_freq:
        map_color_freq[key].append(map_color_freq_2[key])
    else:
        map_color_freq[key] = [map_color_freq_2[key]]

distance_score = 0

for key in map_color_freq:
    if np.array(map_color_freq[key]).shape[0] > 1:
        for k in range(4):
            distance_score += abs(map_color_freq[key][0][k] - map_color_freq[key][1][k])/(1+map_color_freq[key][0][k] + map_color_freq[key][1][k])
    else:
        for k in range(4):
            distance_score += (map_color_freq[key][0][k])/(1+map_color_freq[key][0][k])

unique_colors = len(list(map_color_freq.keys()))

print("Distance ", distance_score/unique_colors)
print("Time Taken : ", time.time() - time_3)
