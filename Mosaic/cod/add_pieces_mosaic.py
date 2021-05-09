from cod.parameters import *
import numpy as np
import copy
import pdb
import timeit
import random


def add_pieces_grid(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    if params.gray:
        N, H, W = params.small_images.shape
    else:
        N, H, W, C = params.small_images.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                index = np.random.randint(low=0, high=N, size=1)
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    elif params.criterion == 'distantaCuloareMedie' and not params.neigh:
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                if not params.gray:
                    patch = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W, :].copy()
                    index_sorted = get_sorted_indices(params.small_images, patch, False)
                    index = index_sorted[0]
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                else:
                    patch = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W].copy()
                    index_sorted = get_sorted_indices(params.small_images, patch, True)
                    index = index_sorted[0]
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))
    elif params.criterion == 'distantaCuloareMedie' and params.neigh:
        neigh = np.empty(shape=(params.num_pieces_vertical, params.num_pieces_horizontal))
        neigh.fill(-1)
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                if not params.gray:
                    patch = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W, :].copy()
                    index_sorted = get_sorted_indices(params.small_images, patch, False)
                else:
                    patch = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W].copy()
                    index_sorted = get_sorted_indices(params.small_images, patch, True)

                if i == 0 and j == 0:
                    index = index_sorted[0]
                    neigh[i, j] = index
                elif i == 0 and j > 0:
                    if index_sorted[0] == neigh[i, j - 1]:
                        index = index_sorted[1]
                    else:
                        index = index_sorted[0]
                    neigh[i, j] = index
                elif i > 0 and j == 0:
                    if index_sorted[0] == neigh[i - 1, j]:
                        index = index_sorted[1]
                    else:
                        index = index_sorted[0]
                    neigh[i, j] = index
                else:
                    if index_sorted[0] != neigh[i - 1, j] and index_sorted[0] != neigh[i, j - 1]:
                        index = index_sorted[0]
                    elif index_sorted[1] != neigh[i - 1, j] and index_sorted[1] != neigh[i, j - 1]:
                        index = index_sorted[1]
                    else:
                        index = index_sorted[2]
                    neigh[i, j] = index

                if not params.gray:
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                else:
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W] = params.small_images[index]

                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))


    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_random(params: Parameters):
    start_time = timeit.default_timer()
    if params.gray:
        N, H, W = params.small_images.shape
    else:
        N, H, W, C = params.small_images.shape

    new_h = params.image_resized.shape[0] + H
    new_w = params.image_resized.shape[1] + W

    if params.gray:
        img_mosaic = np.zeros(shape=(new_h, new_w))
    else:
        img_mosaic = np.zeros(shape=(new_h, new_w, C))

    bigger_image = np.zeros(img_mosaic.shape, np.uint8)
    bigger_image[0:params.image_resized.shape[0], 0: params.image_resized.shape[1]] = params.image_resized.copy()

    indexes = np.arange(params.image_resized.shape[0] * params.image_resized.shape[1]).reshape(
        params.image_resized.shape[0], params.image_resized.shape[1])

    if params.criterion == "distantaCuloareMedie":
        while True:
            print(len(indexes[indexes != -1]))
            if len(indexes[indexes != -1]) == 0:
                break

            positions = random.choice(indexes[indexes != -1])
            height = positions // params.image_resized.shape[1]
            width = positions % params.image_resized.shape[1]
            if not params.gray:
                patch = bigger_image[height: height + H, width: width + W, :].copy()
                index_sorted = get_sorted_indices(params.small_images, patch, False)
            else:
                patch = bigger_image[height: height + H, width: width + W].copy()
                index_sorted = get_sorted_indices(params.small_images, patch, True)

            index = index_sorted[0]
            if not params.gray:
                img_mosaic[height: height + H, width: width + W, :] = params.small_images[index]
            else:
                img_mosaic[height: height + H, width: width + W] = params.small_images[index]
            indexes[height: height + H, width: width + W] = -1
    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)
    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic[0:params.image_resized.shape[0], 0:params.image_resized.shape[1]]


def get_mask(params: Parameters):
    h = params.small_images.shape[1]
    w = params.small_images.shape[2]
    if not params.gray:
        mask = np.ones((h, w, 3))
    else:
        mask = np.ones((h, w))
    for i in range(h // 2):
        for j in range(w // 3 - i):
            mask[i, j] = 0
        for j in range(w - 1, w - 1 - w // 3+i, -1):
            mask[i, j] = 0

    for i in range(h // 2, h):
        for j in range(i - h // 2):
            mask[i, j] = 0
        for j in range(w - 1, w - 1 - (i - h // 2), -1):
            mask[i, j] = 0
    return mask


def add_pieces_hexagon(params: Parameters):
    # print(params.image_resized.shape[0], params.image_resized.shape[1])
    start_time = timeit.default_timer()
    if params.gray:
        N, H, W = params.small_images.shape
    else:
        N, H, W, C = params.small_images.shape

    new_h = params.image_resized.shape[0] + H
    new_w = params.image_resized.shape[1] + W

    if params.gray:
        img_mosaic = np.zeros(shape=(new_h, new_w))
    else:
        img_mosaic = np.zeros(shape=(new_h, new_w, C))

    bigger_image = np.zeros(img_mosaic.shape, np.uint8)
    bigger_image[H//2:params.image_resized.shape[0]+H//2, W//2: params.image_resized.shape[1]+W//2] = params.image_resized.copy()

    mask = get_mask(params)
    if params.criterion == "distantaCuloareMedie" and not params.neigh:
        for i in range(H//2, bigger_image.shape[0] - H//2, H):
            for j in range(0, bigger_image.shape[1]-W, int(np.ceil(W + W / 3))):
                if not params.gray:
                    patch = bigger_image[i:i+H, j:j+W, :].copy()
                    index_sorted = get_sorted_indices(params.small_images, patch, False)
                else:
                    patch = bigger_image[i:i+H, j:j+W].copy()
                    index_sorted = get_sorted_indices(params.small_images, patch, True)
                index = index_sorted[0]
                if not params.gray:
                    bigger_image[i:i + H, j:j + W, :] = (1 - mask) * bigger_image[i:i + H, j:j + W, :] + mask * params.small_images[index]
                else:
                    bigger_image[i:i + H, j:j + W] = (1 - mask) * bigger_image[i:i + H, j:j + W] + mask * \
                                                        params.small_images[index]

        for i in range(0, bigger_image.shape[0] - H//2, H):
            for j in range(2*W//3+1, bigger_image.shape[1]-W, int(np.ceil(W + W / 3))):
                if not params.gray:
                    patch = bigger_image[i:i+H, j:j+W, :].copy()
                    index_sorted = get_sorted_indices(params.small_images, patch, False)
                else:
                    patch = bigger_image[i:i+H, j:j+W].copy()
                    index_sorted = get_sorted_indices(params.small_images, patch, True)
                index = index_sorted[0]
                if not params.gray:
                    bigger_image[i:i + H, j:j + W, :] = (1 - mask) * bigger_image[i:i + H, j:j + W, :] + mask * params.small_images[index]
                else:
                    bigger_image[i:i + H, j:j + W] = (1 - mask) * bigger_image[i:i + H, j:j + W] + mask * \
                                                        params.small_images[index]
    elif params.criterion == "distantaCuloareMedie" and params.neigh:
        neigh = np.empty(shape=(params.num_pieces_vertical*2+2, params.num_pieces_horizontal*2))
        neigh.fill(-1)
        row = 1
        for i in range(H // 2, bigger_image.shape[0] - H // 2, H):
            col = 0
            for j in range(0, bigger_image.shape[1] - W, int(np.ceil(W + W / 3))):
                if not params.gray:
                    patch = bigger_image[i:i+H, j:j+W, :].copy()
                    index_sorted = get_sorted_indices(params.small_images, patch, False)
                else:
                    patch = bigger_image[i:i+H, j:j+W].copy()
                    index_sorted = get_sorted_indices(params.small_images, patch, True)
                if row == 1:
                    index = index_sorted[0]
                else:
                    if index_sorted[0] == neigh[row - 2, col]:
                        index = index_sorted[1]
                    else:
                        index = index_sorted[0]
                if not params.gray:
                    bigger_image[i:i + H, j:j + W, :] = (1 - mask) * bigger_image[i:i + H, j:j + W, :] + mask * params.small_images[index]
                else:
                    bigger_image[i:i + H, j:j + W] = (1 - mask) * bigger_image[i:i + H, j:j + W] + mask * \
                                                        params.small_images[index]
                neigh[row, col] = index
                col = col + 2
            row = row + 2

        row = 0
        for i in range(0, bigger_image.shape[0] - H // 2, H):
            col = 1
            for j in range(int(np.ceil(2 * W / 3)), bigger_image.shape[1] - W, int(np.ceil(W + W / 3))):
                if not params.gray:
                    patch = bigger_image[i:i+H, j:j+W, :].copy()
                    index_sorted = get_sorted_indices(params.small_images, patch, False)
                else:
                    patch = bigger_image[i:i+H, j:j+W].copy()
                    index_sorted = get_sorted_indices(params.small_images, patch, True)
                if row == 0:
                    if neigh[row + 1, col - 1] != index_sorted[0] and neigh[row + 1, col + 1] != index_sorted[0]:
                        index = index_sorted[0]
                    elif neigh[row + 1, col - 1] != index_sorted[1] and neigh[row + 1, col + 1] != index_sorted[1]:
                        index = index_sorted[1]
                    else:
                        index = index_sorted[2]
                else:
                    if neigh[row + 1, col - 1] != index_sorted[0] and neigh[row + 1, col + 1] != index_sorted[0] and neigh[row - 2, col] != index_sorted[0]\
                            and neigh[row - 1, col - 1] != index_sorted[0] and neigh[row - 1, col + 1] != index_sorted[0]:
                        index = index_sorted[0]
                    elif neigh[row + 1, col - 1] != index_sorted[1] and neigh[row + 1, col + 1] != index_sorted[1] and neigh[row - 2, col] != index_sorted[1]\
                            and neigh[row - 1, col - 1] != index_sorted[1] and neigh[row - 1, col + 1] != index_sorted[1]:
                        index = index_sorted[1]
                    elif neigh[row + 1, col - 1] != index_sorted[2] and neigh[row + 1, col + 1] != index_sorted[2] and neigh[row - 2, col] != index_sorted[2]\
                            and neigh[row - 1, col - 1] != index_sorted[2] and neigh[row - 1, col + 1] != index_sorted[2]:
                        index = index_sorted[2]
                    elif neigh[row + 1, col - 1] != index_sorted[3] and neigh[row + 1, col + 1] != index_sorted[3] and neigh[row - 2, col] != index_sorted[3]\
                            and neigh[row-1, col-1] != index_sorted[3] and neigh[row - 1, col + 1] != index_sorted[3]:
                        index = index_sorted[3]
                    elif neigh[row + 1, col - 1] != index_sorted[4] and neigh[row + 1, col + 1] != index_sorted[4] and neigh[row - 2, col] != index_sorted[4]\
                            and neigh[row - 1, col - 1] != index_sorted[4] and neigh[row - 1, col + 1] != index_sorted[4]:
                        index = index_sorted[4]
                    else:
                        index = index_sorted[5]
                if not params.gray:
                    bigger_image[i:i + H, j:j + W, :] = (1 - mask) * bigger_image[i:i + H, j:j + W, :] + mask * params.small_images[index]
                else:
                    bigger_image[i:i + H, j:j + W] = (1 - mask) * bigger_image[i:i + H, j:j + W] + mask * \
                                                        params.small_images[index]
                neigh[row, col] = index
                col = col + 2
            row = row + 2
    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))
    return bigger_image[H // 2:params.image_resized.shape[0]+H // 2, W // 2: params.image_resized.shape[1] + W // 2]


def get_sorted_indices(small_photos, patch, gray):
    mean_small_photos = np.mean(small_photos, axis=(1, 2))
    mean_patch = np.mean(patch, axis=(0, 1))
    if not gray:
        euclidian_distance = np.sum((mean_small_photos - mean_patch) * (mean_small_photos - mean_patch), axis=1)
    else:
        euclidian_distance = (mean_small_photos - mean_patch) * (mean_small_photos - mean_patch)

    return euclidian_distance.argsort()
