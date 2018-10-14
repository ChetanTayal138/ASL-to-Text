This repo is our first implementation of a ASL To Text converter which can be utilized by mute people to talk to 
smart assistants such as Alexa or Google Assistant.

The ipnyb file contains the CNN model implementation which was trained over 50 epochs and achieved a validation accuracy of 100% and test accuracy of 97.9%.

The hand_detector.py module contains our rough implementation of grabbing a certain part of the frame as an image to be fed into our prediction module.


The keras_pred.py module is fed with the image taken by the above module and this image is then used by our saved model in the model.json file to make a prediction. The weights of the model have been saved in the cnn_model.h5 file.

Finally the predicted letter is displayed on to the screen.