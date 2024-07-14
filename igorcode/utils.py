import os
import tensorflow.keras.preprocessing.image as image 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelBinarizer 
import numpy as np
def load_data(data_dir, input_shape):
	'''
	:param data_dir:
	:param input_shape:
	:return:
	'''

	image_data = [] 
	class_labels = []
	# Walk the domain dataset and collect the images, and the class/domain labels based on the appropriate directory
	# name
	for root, dirs, files in os.walk(data_dir): 
		for file in files:
			_, class_label =  os.path.split(os.path.normpath(root))
			image_path = os.path.join(root, file)
# Load image and convert it to an array of float
			img = image.load_img(image_path, target_size=input_shape) 
			img_arr = image.img_to_array(img, dtype='float')
# Rescale pixel values to be between 0 and 1 
			norm_image_arr = img_arr / 255 
			image_data.append(norm_image_arr) 
			class_labels.append(class_label)
	return image_data, class_labels

def preprocess_data(img_data, class_labels):
	x, x_test, y, y_test =	train_test_split(img_data,
		class_labels, stratify=class_labels, test_size=.2)
# Transform class and domain labels into one-hot encoded arrays
	class_encoder = LabelBinarizer() 
	y = class_encoder.fit_transform(y)
	y_test = class_encoder.fit_transform(y_test)
	return np.array(x), np.array(y), np.array(x_test), np.array(y_test)
