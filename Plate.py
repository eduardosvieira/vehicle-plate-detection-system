import flask
from flask import Flask, session
from sklearn.externals import joblib
import numpy as np
from scipy import misc
from flask import send_from_directory

from skimage.io import imread, imshow, show
from skimage.filters import threshold_otsu, gaussian
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import regionprops
import matplotlib
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import resize
import simplejson as json
import cv2
import os
from werkzeug.utils import secure_filename

from glob import glob

import sys

sys.setrecursionlimit(1500)

#Variaveis Globais

i = 0
s = 0

plate_obj = []
column_list = []
characters = []

def deteccaoDePlaca(path):
	global plate_obj
	global i
	img = imread(path)

	car_image = imread(path, as_gray=True)

	gray_car_image = car_image * 255

	filtered_img = gaussian(gray_car_image, sigma=1, multichannel=True)

	fig, (ax1, ax2) = plt.subplots(1, 2)

	threshold_value = threshold_otsu(filtered_img)

	binary_car_image = gray_car_image > threshold_value

	label_image = measure.label(binary_car_image)

	plate_dimensions = (0.03*label_image.shape[0], 0.08*label_image.shape[0], 0.15*label_image.shape[1], 0.3*label_image.shape[1])
	plate_dimensions2 = (0.08*label_image.shape[0], 0.2*label_image.shape[0], 0.15*label_image.shape[1], 0.4*label_image.shape[1])

	min_height, max_height, min_width, max_width = plate_dimensions

	plate_objects_cordinates = []
	plate_like_objects = []

	fig, (ax1) = plt.subplots(1)
	ax1.imshow(gray_car_image, cmap="gray")
	flag = 0

	for region in regionprops(label_image):
		if region.area < 50:
		    continue

		min_row, min_col, max_row, max_col = region.bbox

		region_height = max_row - min_row
		region_width = max_col - min_col

		if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:

		    flag = 1
		    plate_like_objects.append(binary_car_image[min_row:max_row,
		                              min_col:max_col])
		    plate_objects_cordinates.append((min_row, min_col,
		                                     max_row, max_col))
		    rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
		                                   linewidth=2, fill=False)

		    ax1.add_patch(rectBorder)
	if(flag == 1):
		plt.axis('off')

		plt.savefig('result/car' + str(i) + '.png',bbox_inches='tight')
	if(flag==0):
		min_height, max_height, min_width, max_width = plate_dimensions2
		plate_objects_cordinates = []
		plate_like_objects = []

		for region in regionprops(label_image):
		    if region.area < 50:
		        continue
		    min_row, min_col, max_row, max_col = region.bbox

		    region_height = max_row - min_row
		    region_width = max_col - min_col

		    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
		        plate_like_objects.append(binary_car_image[min_row:max_row,
		                                  min_col:max_col])
		        plate_objects_cordinates.append((min_row, min_col,
		                                         max_row, max_col))
		        rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
		                                       linewidth=2, fill=False)
		        ax1.add_patch(rectBorder)

		plt.axis('off')
		plt.savefig('result/car' + str(i) + '.png',bbox_inches='tight')

	i = i + 1

	if len(plate_like_objects) > 0:
		for p in plate_like_objects:
		    p = p.tolist()
		    plate_obj.append(p)

		return True

	plate_obj = []
	colum_list = []
	characters = []
	return False

def segmetacaoDeCaracteres():
	global plate_obj
	global colum_list
	global characters
	global s

	for i in plate_obj:

		license_plate = np.invert(i)

		labelled_plate = measure.label(license_plate)

		fig, ax1 = plt.subplots(1)
		ax1.imshow(license_plate, cmap="gray")

		character_dimensions = (0.35*license_plate.shape[0], 0.60*license_plate.shape[0], 0.05*license_plate.shape[1], 0.15*license_plate.shape[1])

		min_height, max_height, min_width, max_width = character_dimensions

		for regions in regionprops(labelled_plate):
			y0, x0, y1, x1 = regions.bbox
			region_height = y1 - y0
			region_width = x1 - x0

			if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
				roi = license_plate[y0:y1, x0:x1]

				rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
				                               linewidth=2, fill=False)

				ax1.add_patch(rect_border)

				resized_char = resize(roi, (20, 20), mode='constant')
				characters.append(resized_char)

				column_list.append(x0)

		if len(characters) > 0:
			if existeCaracteres():
				plt.savefig('segment/car' + str(s) + '.png',bbox_inches='tight')
				s = s + 1

				predicao()

def predicao():
	try:
		global characters
		global column_list
		classification_result = []

		filename = 'finalized_model.sav'
		model = pickle.load(open(filename, 'rb'))

		for each_character in characters:
			each_character = each_character.reshape(1, -1)
			result = model.predict(each_character)
			classification_result.append(result)

		plate_string = ''
		for eachPredict in classification_result:
			plate_string += eachPredict[0]

		column_list_copy = column_list[:]
		column_list.sort()
		placa_string = ''
		for each in column_list:
			placa_string += plate_string[column_list_copy.index(each)]

		if len(placa_string) > 0:
			print('Placa')
			print(placa_string)

			file = open("result.txt", "a")
			file.write("Placa: " + placa_string + "\n")
			file.close()

		colum_list = []
		characters = []
		return True

	except:
		return False
		colum_list = []
		characters = []

def existeCaracteres():
	try:
		global characters
		global column_list
		classification_result = []

		filename = 'finalized_model.sav'
		model = pickle.load(open(filename, 'rb'))

		for each_character in characters:
			# converts it to a 1D array
			each_character = each_character.reshape(1, -1);
			result = model.predict(each_character)
			classification_result.append(result)

		plate_string = ''
		for eachPredict in classification_result:
			plate_string += eachPredict[0]

		column_list_copy = column_list[:]
		column_list.sort()
		placa_string = ''
		for each in column_list:
			placa_string += plate_string[column_list_copy.index(each)]

		if len(placa_string) > 0:
			return True
		return False
	except:
		return False


if __name__ == '__main__':

	file = open("result.txt", "w")
	file.write("RESULTADO\n\n")
	file.close()

	files_name = glob('dataset/**')
	for file in files_name:
		if deteccaoDePlaca(file):
			print(file)
			segmetacaoDeCaracteres()

		characters = []
		column_list = []
		plate_obj = []
