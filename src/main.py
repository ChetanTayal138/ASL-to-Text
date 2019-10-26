import cv2 
import numpy as np
import time 
import matplotlib.pyplot as plt 
import tensorflow as tf 








labels_dict = {
        0:"A",
        1:"B",
        2:"C",
        3:"D",
        4:"E",
        5:"F",
        6:"G",
        7:"H",
        8:"I",
        9:"J",
        10:"K",
        11:"L",
        12:"M",
        13:"N",
        14:"O",
        15:"P",
        16:"Q",
        17:"R",
        18:"S",
        19:"T",
        20:"U",
        21:"V",
        22:"W",
        23:"X",
        24:"Y",
        25:"Z",
        26:"d",
        27:"n",
        28:"s"
        }





def retrieve_model():
    new_model = tf.keras.models.load_model("../src/my_model.h5")
    new_model.summary()
    return new_model 




def funcmain():
    count= 0 
    model = retrieve_model()
    x,y,w,h = 0 , 0 , 200 , 200
    mirror = False
    cam = cv2.VideoCapture(0)

    while True:

        ret_val, img = cam.read()
        img[: , : , 0] , img[: , : , 2] = img[: , : , 2] , img[: , : , 0] #Converting from BGR to RGB 
        reading_frame = img[:200 , :200 , :]
        reading_frame = np.flip(reading_frame , axis = 1)
    
        if(count == 50):
            plt.imshow(reading_frame)
            plt.show()
            prediction = model.predict(reading_frame.reshape(-1,200,200,3).astype(np.float64), batch_size = 1 , verbose = 1)

            print(labels_dict[np.argmax(prediction)])
            count = 0
        else:
            count = count + 1 


        if mirror: 
            img = cv2.flip(img, 1)

        cv2.rectangle(img, (x,y), (x+w,y+h) , (255,0,0) ,2)
        cv2.imshow('Lol' , img)




        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()




if __name__ == "__main__":

    funcmain()






