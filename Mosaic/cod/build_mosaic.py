import os
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import pdb

from cod.add_pieces_mosaic import *
from cod.parameters import *
import os


def load_pieces(params: Parameters):
    # citeste toate cele N piese folosite la mozaic din directorul corespunzator
    # toate cele N imagini au aceeasi dimensiune H x W x C, unde:
    # H = inaltime, W = latime, C = nr canale (C=1  gri, C=3 color)
    # functia intoarce pieseMozaic = matrice N x H x W x C in params
    # pieseMoziac[i, :, :, :] reprezinta piesa numarul i
    images = []

    for file in os.listdir(params.small_images_dir):
        if params.gray == True:
            img = cv.imread(os.path.join(params.small_images_dir, file), cv.IMREAD_GRAYSCALE)
        else:
            img = cv.imread(os.path.join(params.small_images_dir, file))
        img = np.array(img, dtype=np.int16)
        images.append(img)

    # citeste imaginile din director

    if params.show_small_images:
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, i * 10 + j + 1)
                # OpenCV reads images in BGR format, matplotlib reads images in RBG format
                im = images[i * 10 + j].copy()
                # BGR to RGB, swap the channels
                im = im[:, :, [2, 1, 0]]
                plt.imshow(im)
        plt.show()

    images = np.asarray(images)
    params.small_images = images


def compute_dimensions(params: Parameters):
    # calculeaza dimensiunile mozaicului
    # obtine si imaginea de referinta redimensionata avand aceleasi dimensiuni
    # ca mozaicul

    height = 0
    width = 0
    if params.gray:
        height, width = params.image.shape
    else:
        height, width, _ = params.image.shape
    width_small_image = params.small_images.shape[2]
    height_small_image = params.small_images.shape[1]

    # redimensioneaza imaginea
    new_w = width_small_image * params.num_pieces_horizontal
    aspect_ratio = height / width
    new_h = aspect_ratio * new_w
    params.num_pieces_vertical = int(round(new_h / height_small_image))
    params.image_resized = cv.resize(params.image, (new_w, params.num_pieces_vertical*height_small_image))


def build_mosaic(params: Parameters):
    # incarcam imaginile din care vom forma mozaicul
    load_pieces(params)
    # calculeaza dimensiunea mozaicului
    compute_dimensions(params)

    img_mosaic = None
    if params.layout == 'caroiaj':
        if params.hexagon is True:
            img_mosaic = add_pieces_hexagon(params)
        else:
            img_mosaic = add_pieces_grid(params)
    elif params.layout == 'aleator':
        img_mosaic = add_pieces_random(params)
    else:
        print('Wrong option!')
        exit(-1)

    return img_mosaic


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def load_pieces_cifar(path, category):
    label_names = unpickle(path + "/batches.meta")["label_names"]
    label_digit = label_names.index(category)

    contor = 0
    try:
        os.mkdir("./../photo" + category)
    except OSError:
        print("Creation of the directory %s failed" % path)

    for i in range(1, 6):
        data = unpickle(path + "/data_batch_" + str(i))
        matrix = data['data']
        Channels = 3
        no_pixels = round(math.sqrt(matrix.shape[1] / Channels))
        images = matrix.reshape(matrix.shape[0], Channels, no_pixels, no_pixels).transpose(0, 2, 3, 1)

        labels = np.asarray(data['labels'])
        indices = np.where(labels == label_digit)[0]
        images_useful = images[indices]
        print(images_useful.shape)

        for image in images_useful:
            cv.imwrite('./../photo' + category + "/" + category + str(contor) + ".png", image)
            contor += 1

    data = unpickle(path + "/test_batch")
    matrix = data['data']
    Channels = 3
    no_pixels = round(math.sqrt(matrix.shape[1] / Channels))
    images = matrix.reshape(matrix.shape[0], Channels, no_pixels, no_pixels).transpose(0, 2, 3, 1)

    labels = np.asarray(data['labels'])
    indices = np.where(labels == label_digit)[0]
    images_useful = images[indices]
    print(images_useful.shape)

    for image in images_useful:
        cv.imwrite('./../photo' + category + "/" + category + str(contor) + ".png", image)
        contor += 1