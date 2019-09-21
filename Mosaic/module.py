import numpy as np
import cv2

INF =  200000

def resize_image(image, template):
	i_height, i_width, i_channels = image.shape
	t_height, t_width, t_channels = template.shape

	i_height -= i_height % t_height
	i_width -= i_width % t_width
	return i_height, i_width


def average_color(image):
	avg_color_per_row = np.average(image, axis=0)
	avg_color = np.average(avg_color_per_row, axis=0)
	return avg_color


def best_matching_photo(img):
	b, g, r = average_color(img)
	ans = 0
	best_ed = INF
	for number in range(1,500):
		path = str(number) + ".png"
		template = cv2.imread(path)
		new_b, new_g, new_r = average_color(template)
		ed = (new_b - b) ** 2 + (new_g - g) ** 2 + (new_r -r) ** 2
		if ed < best_ed:
			ans = number
			best_ed = ed
	return ans 


def mosaic_photo(image):
	template = cv2.imread("1.png")
	t_height, t_width, t_channels = template.shape

	new_height, new_width = resize_image(image, template)
	new_image = image[0:new_height, 0:new_width]

	i = 0
	while i <= new_height - t_height - 1:
		j = 0
		while j <= new_width - t_width - 1:
			region_image = image[i:i+t_height, j:j+t_width]
			number = best_matching_photo(region_image)
			path = str(number) + ".png"
			template = cv2.imread(path)
			new_image[i:i+t_height, j:j+t_width] = template
			j = j + t_width
		i = i + t_height
	return new_image





