import sys
import cv2 as cv
import numpy as np
import copy

from cod.parameters import *
from cod.select_path import *

import pdb


def compute_energy(img):
    E = np.zeros((img.shape[0], img.shape[1]))
    img = np.uint8(img)
    image_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(image_gray, cv.CV_64F, 1, 0)
    sobely = cv.Sobel(image_gray, cv.CV_64F, 0, 1)
    E = np.array(np.abs(sobelx) + np.abs(sobely), dtype=np.int32)
    return E


def show_path(img, path, color):
    new_image = img.copy()
    for row, col in path:
        new_image[row, col] = color

    E = compute_energy(img)
    new_image_E = img.copy()
    new_image_E[:, :, 0] = E.copy()
    new_image_E[:, :, 1] = E.copy()
    new_image_E[:, :, 2] = E.copy()

    for row, col in path:
        new_image_E[row, col] = color
    cv.imshow('path img', np.uint8(new_image))
    cv.imshow('path E', np.uint8(new_image_E))
    cv.waitKey(1000)


def delete_path(img, path):
    updated_img = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]), np.uint8)
    for i in range(img.shape[0]):
        col = path[i][1]
        # copiem partea din stanga
        updated_img[i, :col] = img[i, :col].copy()
        # copiem partea din dreapta
        updated_img[i, col:] = img[i, col + 1:].copy()

    return updated_img


def decrease_width(params: Parameters, num_pixels):
    img = params.image.copy()  # copiaza imaginea originala
    for i in range(num_pixels):
        print('Eliminam drumul vertical numarul %i dintr-un total de %d.' % (i + 1, num_pixels))

        # calculeaza energia dupa ecuatia (1) din articol
        E = compute_energy(img)
        path = select_path(E, params.method_select_path)
        if params.show_path:
            show_path(img, path, params.color_path)
        img = delete_path(img, path)

    cv.destroyAllWindows()
    return img


def decrease_height(params: Parameters, num_pixels):
    img = params.image.copy()
    rotated_image = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    new_params = copy.deepcopy(params)
    new_params.image = rotated_image
    new_image = decrease_width(new_params, num_pixels)
    return cv.rotate(new_image, cv.ROTATE_90_COUNTERCLOCKWISE)


def delete_object(params: Parameters, x0, y0, w, h):
    x1 = x0 + w
    y1 = y0 + h
    new_img = params.image.copy()
    num_pixels_width = x1 - x0 + 1
    num_pixels_height = y1 - y0 + 1
    print(num_pixels_height, num_pixels_width)
    if num_pixels_width < num_pixels_height:
        for i in range(num_pixels_width):
            print('Eliminam drumul vertical numarul %i dintr-un total de %d.' % (i + 1, num_pixels_width))
            E = compute_energy(new_img)
            E[y0:y1 + 1, x0:x1 + 1] = -1e12
            path = select_path(E, params.method_select_path)
            if params.show_path:
                show_path(new_img, path, params.color_path)
            new_img = delete_path(new_img, path)
            x1 = x1 - 1

    else:
        for i in range(num_pixels_height):
            print('Eliminam drumul orizontal numarul %i dintr-un total de %d.' % (i + 1, num_pixels_height))
            E = compute_energy(new_img)
            E[y0:y1+1, x0:x1+1] = -1e12
            rotated_E = cv.rotate(E, cv.ROTATE_90_CLOCKWISE)
            path = select_path(rotated_E, params.method_select_path)
            rotated_img = cv.rotate(new_img, cv.ROTATE_90_CLOCKWISE)
            if params.show_path:
                show_path(rotated_img, path, params.color_path)
            rotated_img = delete_path(rotated_img, path)
            new_img = cv.rotate(rotated_img, cv.ROTATE_90_COUNTERCLOCKWISE)
            y1 = y1 - 1
    cv.destroyAllWindows()
    return new_img


def amplify_content(params: Parameters):
    img = params.image.copy()
    resized_image = cv.resize(img, (0, 0), fx=params.factor_amplification, fy=params.factor_amplification)
    number_pixels_height = resized_image.shape[0] - img.shape[0]
    number_pixels_width = resized_image.shape[1] - img.shape[1]

    new_params_width = copy.deepcopy(params)
    new_params_width.image = resized_image
    img_width = decrease_width(new_params_width, number_pixels_width)

    new_params_height = copy.deepcopy(params)
    new_params_height.image = img_width
    new_image = decrease_height(new_params_height, number_pixels_height)
    return new_image


def resize_image(params: Parameters):
    if params.resize_option == 'micsoreazaLatime':
        # redimensioneaza imaginea pe latime
        resized_image = decrease_width(params, params.num_pixels_width)
        return resized_image

    elif params.resize_option == 'micsoreazaInaltime':
        resized_image = decrease_height(params, params.num_pixel_height)
        return resized_image

    elif params.resize_option == 'amplificaContinut':
        resized_image = amplify_content(params)
        return resized_image

    elif params.resize_option == 'eliminaObiect':
        x0, y0, w, h = cv.selectROI(np.uint8(params.image))
        resized_image = delete_object(params, x0, y0, w, h)
        return resized_image



    else:
        print('The option is not valid!')
        sys.exit(-1)
