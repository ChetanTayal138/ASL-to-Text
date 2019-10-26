import tensorflow as tf 
import numpy as np 
from utils import load_data 
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def get_model():
        model = keras.Sequential([
        keras.layers.Conv2D(29, kernel_size = (3,3) , activation = 'relu' , input_shape = (200,200,3)),
        keras.layers.MaxPooling2D(pool_size = (2,2)),
        keras.layers.Conv2D(32, kernel_size = (3,3) , activation = 'relu'),
        keras.layers.MaxPooling2D(pool_size = (2,2)),
        keras.layers.Conv2D(64, kernel_size = (3,3) , activation = 'relu'),
        keras.layers.MaxPooling2D(pool_size = (2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation = 'relu'),
        keras.layers.Dense(128, activation = 'relu'),
        keras.layers.Dense(29 , activation = 'softmax')
        ])

        return model





def train_model(model, x_train , y_train):
    model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])
    datagen = ImageDataGenerator(
            featurewise_center = True,
            featurewise_std_normalization = True,
            rotation_range = 20,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            horizontal_flip = False)

    datagen.fit(x_train)
    #model.fit_generator(datagen.flow(x_train , y_train , batch_size = 32), steps_per_epoch = len(x_train) / 32 , epochs = 50)



    model.fit(x_train , y_train , shuffle = True , epochs = 5)
    return model 




if __name__ == "__main__":

    """

    checkpoint_path = "training_checkpoints/cp.cpkt"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path , save_weights_only = True , verbose = 1 )
 """

   
    

    

    
    x_train, y_train , x_test , y_test = load_data()
    model = get_model()
    model = train_model(model , x_train , y_train)
    model.summary()
    model.save("../src/my_model.h5")
    
    test_loss, test_acc = model.evaluate(x_test, y_test , verbose = 2)
    print(test_loss)
    print("ACCURACY IS " + str(test_acc))
    



























