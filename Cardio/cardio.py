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

from keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import InverseTimeDecay,ExponentialDecay
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.utils import Progbar

from keras.losses import CategoricalCrossentropy,BinaryCrossentropy
from keras.optimizers import Adam,RMSprop
from keras.applications.densenet import DenseNet121
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from keras.metrics import CategoricalAccuracy, Precision, Recall
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, AveragePooling2D, Flatten, Input,GlobalAveragePooling2D
from keras.utils.vis_utils import plot_model
from keras.optimizers.schedules import InverseTimeDecay,ExponentialDecay

from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
model = keras.models.load_model('CNN_Deep_vgg8.h5')