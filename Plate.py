import flask
from flask import Flask, session
from sklearn.externals import joblib
import numpy as np
from scipy import misc
from flask import send_from_directory

from skimage.io import imread, imshow, show
from skimage.filters import threshold_otsu
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


#Variaveis Globais

plate_obj = None
column_list = []
characters = []

def detectPlate(path):
	global plate_obj
	img = imread(path)

	car_image = imread(path, as_gray=True)

	gray_car_image = car_image * 255

	fig, (ax1, ax2) = plt.subplots(1, 2)
	ax1.imshow(gray_car_image, cmap="gray")
	threshold_value = threshold_otsu(gray_car_image)
	binary_car_image = gray_car_image > threshold_value
	ax2.imshow(binary_car_image, cmap="gray")
	label_image = measure.label(binary_car_image)

	plate_dimensions = (0.03*label_image.shape[0], 0.08*label_image.shape[0], 0.15*label_image.shape[1], 0.3*label_image.shape[1])
	plate_dimensions2 = (0.08*label_image.shape[0], 0.2*label_image.shape[0], 0.15*label_image.shape[1], 0.4*label_image.shape[1])
	min_height, max_height, min_width, max_width = plate_dimensions
	plate_objects_cordinates = []
	plate_like_objects = []

	fig, (ax1) = plt.subplots(1)
	ax1.imshow(gray_car_image, cmap="gray")
	flag = 0

	# regionprops creates a list of properties of all the labelled regions
	for region in regionprops(label_image):
		if region.area < 50:
		    #if the region is so small then it's likely not a license plate
		    continue
		    # the bounding box coordinates
		min_row, min_col, max_row, max_col = region.bbox

		region_height = max_row - min_row
		region_width = max_col - min_col

		# ensuring that the region identified satisfies the condition of a typical license plate
		if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:

		    flag = 1
		    plate_like_objects.append(binary_car_image[min_row:max_row,
		                              min_col:max_col])
		    plate_objects_cordinates.append((min_row, min_col,
		                                     max_row, max_col))
		    rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
		                                   linewidth=2, fill=False)
		    ax1.add_patch(rectBorder)
		    # let's draw a red rectangle over those regions
	if(flag == 1):
		plt.axis('off')
		#imshow(gray_car_image)
		#show()

		plt.savefig('car.png',bbox_inches='tight')
	if(flag==0):
		min_height, max_height, min_width, max_width = plate_dimensions2
		plate_objects_cordinates = []
		plate_like_objects = []

		# regionprops creates a list of properties of all the labelled regions
		for region in regionprops(label_image):
		    if region.area < 50:
		        #if the region is so small then it's likely not a license plate
		        continue
		        # the bounding box coordinates
		    min_row, min_col, max_row, max_col = region.bbox

		    region_height = max_row - min_row
		    region_width = max_col - min_col

		    # ensuring that the region identified satisfies the condition of a typical license plate
		    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
		        plate_like_objects.append(binary_car_image[min_row:max_row,
		                                  min_col:max_col])
		        plate_objects_cordinates.append((min_row, min_col,
		                                         max_row, max_col))
		        rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
		                                       linewidth=2, fill=False)
		        ax1.add_patch(rectBorder)
		        # let's draw a red rectangle over those regions

		plt.axis('off')
		plt.savefig('car.png', bbox_inches='tight')

	p = plate_like_objects[0]
	p = p.tolist()
	plate_obj= p


def segmentCharacters():
	global plate_obj
	global colum_list
	global characters

	license_plate = np.invert(plate_obj)

	labelled_plate = measure.label(license_plate)

	fig, ax1 = plt.subplots(1)
	ax1.imshow(license_plate, cmap="gray")
	character_dimensions = (0.35*license_plate.shape[0], 0.60*license_plate.shape[0], 0.05*license_plate.shape[1], 0.15*license_plate.shape[1])
	min_height, max_height, min_width, max_width = character_dimensions

	counter=0

	for regions in regionprops(labelled_plate):
	    y0, x0, y1, x1 = regions.bbox
	    region_height = y1 - y0
	    region_width = x1 - x0

	    if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
	        roi = license_plate[y0:y1, x0:x1]

	        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
	                                       linewidth=2, fill=False)
	        ax1.add_patch(rect_border)

	        resized_char = resize(roi, (20, 20))
	        characters.append(resized_char)

	        column_list.append(x0)
	plt.axis('off')
	plt.savefig('segmented_car.png', bbox_inches='tight')
	print('characters',characters)
	print('column_list',column_list)
	print(type(characters))
	print(type(column_list))
	


def make_prediction():
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

	print('Classification result')
	print(classification_result)

	plate_string = ''
	for eachPredict in classification_result:
	    plate_string += eachPredict[0]

	print('Predicted license plate')
	print(plate_string)
	column_list_copy = column_list[:]
	column_list.sort()
	rightplate_string = ''
	for each in column_list:
	    rightplate_string += plate_string[column_list_copy.index(each)]

	print('License plate')
	print(rightplate_string)
	
	characters = []
	column_list = []


if __name__ == '__main__':
	detectPlate("car.jpg")
	segmentCharacters()
	make_prediction()
