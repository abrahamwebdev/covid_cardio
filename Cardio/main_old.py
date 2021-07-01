import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
		print(path)
		#print('upload_image filename: ' + filename)
		import numpy as np
		import pandas as pd
		import tensorflow as tf
		import matplotlib.pyplot as plt
		import keras
		import time
		import keras.backend as K
		import os
		import shutil
		import random

		from tensorflow.keras.optimizers.schedules import InverseTimeDecay,ExponentialDecay
		from tensorflow.keras.applications.resnet50 import ResNet50
		from tensorflow.keras.applications import VGG16
		from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
		from tensorflow.keras.utils import Progbar

		from keras.losses import CategoricalCrossentropy,BinaryCrossentropy
		from keras.optimizers import Adam,RMSprop
		from keras.applications.densenet import DenseNet121
		# from keras.applications import InceptionV3
		from keras.preprocessing.image import ImageDataGenerator
		from keras.models import Model, Sequential
		from keras.metrics import CategoricalAccuracy, Precision, Recall
		from keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, AveragePooling2D, Flatten, Input,GlobalAveragePooling2D
		from keras.utils.vis_utils import plot_model
		from keras.optimizers.schedules import InverseTimeDecay,ExponentialDecay

		from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
		from google.colab import drive
		drive.mount("/content/gdrive")
		model = keras.models.load_model('/content/gdrive/MyDrive/Colab Notebooks/cardio_models/CNN_Deep_vgg8.h5')
		from keras.models import model_from_json
		# serialize model to json
		json_model = model.to_json()
		#save the model architecture to JSON file
		with open('CNN_Deep_vgg8.json', 'w') as json_file:
			json_file.write(json_model)

		from keras.initializers import glorot_uniform
		#Reading the model from JSON file
		with open('CNN_Deep_vgg8.json', 'r') as json_file:
			json_savedModel= json_file.read()
		#load the model architecture 
		model_j = tf.keras.models.model_from_json(json_savedModel)
		import numpy as np
		from keras.preprocessing import image
		test_image = image.load_img('/content/sample_data/cardio1.png', target_size = (100, 100))
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis = 0)
		result = model.predict(test_image)
		if result[0][0] >= result[0][1]:
			if result[0][0] >= result[0][2]:
				flash("Covid Positive chances is the highest")
			else:
				flash("No signs of Covid effects or Viral Pneumonia")
		else:
			print("Chances of Viral Pneumonia is highest")
		# flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)

	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)