import cv2 
from keras.models import model_from_json 

import numpy as np


def modelRetriever():

    with open('model.json' , 'r') as f:
        model = model_from_json(f.read())


    model.load_weights('cnn_model.h5')

    return model


def image_preprocess(image_array):
    img = np.reshape(image_array, (-1, 28, 28, 1))
    return img


def predict_live(image , model):
    model = modelRetriever()
    img = image_preprocess(image)
    prediction = model.predict(img)
    return(prediction)


