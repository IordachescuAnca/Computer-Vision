import cv2
import numpy as np
from module import *
import sys

input_file = raw_input("Enter the full path of the color image you want to a mosaic photo: ")

image = cv2.imread(input_file)
if image is None:
	print("The image path is wrong.")
	sys.exit()


new_image = mosaic_photo(image)
cv2.imwrite('after.jpg', image)
print("Transformation done succesfully.")
