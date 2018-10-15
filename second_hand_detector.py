import cv2 
import numpy as np 
import sys 
import imutils 
from PIL import Image
from keras_pred import *
import time






letter_dictionary = {
0:'a',
1:'b',
2:'c',
3:'d',
4:'e',
5:'f',
6:'g',
7:'h',
8:'i',
9:'k',
10:'l',
11:'m',
12:'n',
13:'o',
14:'p',
15:'q',
16:'r',
17:'s',
18:'t',
19:'u',
20:'v',
21:'w',
22:'x',
23:'y',
24:'z'
}





model = modelRetriever()
video = cv2.VideoCapture(0)
x,y,w,h = 0,55,200, 200

counter = 0 

while True:
    _ , frame = video.read()
    reading_frame = frame[:200 , :200 , :]
  

    im = Image.fromarray(reading_frame , 'RGB')
    im = im.convert('L')
  

  
  
    im = im.resize((28,28))
  
    img_array = np.array(im)
    img_array = image_preprocess(img_array)
    cv2.rectangle(frame , (x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Live Feed" , frame.copy())
    
    if counter == 100:

        prediction = predict_live(img_array , model)
        prediction = prediction[0, :]
        prediction_list = np.array(prediction).tolist()
        predicted_letter = letter_dictionary[prediction_list.index(max(prediction_list))]
        print(predicted_letter)
        counter = 0

    else:
        counter += 1


    

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


video.release()
cv2.destroyAllWindows()





