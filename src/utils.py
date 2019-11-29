from skimage.io import imread_collection
from sklearn import preprocessing 
import numpy as np 
import matplotlib.pyplot as plt 
import os 




path = ""
os.chdir(path)


label_dict = {
        "A" : 0 ,
        "B" : 1 , 
        "C" : 2 , 
        "D" : 3 , 
        "E" : 4 , 
        "F" : 5 , 
        "G" : 6 ,
        "H" : 7 , 
        "I" : 8 , 
        "J" : 9 , 
        "K" : 10, 
        "L" : 11,
        "M" : 12,
        "N" : 13, 
        "O" : 14,
        "P" : 15, 
        "Q" : 16,
        "R" : 17, 
        "S" : 18,
        "T" : 19,
        "U" : 20,
        "V" : 21,
        "W" : 22,
        "X" : 23,
        "Y" : 24,
        "Z" : 25,
        "d" : 26,
        "n" : 27,
        "s" : 28 
        }



def load_train_images():
    col_dir = '../train_data/*/*'
    col = imread_collection(col_dir)
    return col

def load_test_images():
    col_dir = '../test_data/*/*'
    col = imread_collection(col_dir)
    return col 



"""Generating the labels for the training images"""

def generate_labels():

    labels = [] 
    final_labels = []
    for r,d,f in os.walk(os.getcwd()):
        for x in f:
            labels.append(x[0])

    for i in labels:
        final_labels.append(label_dict[i])
    
    return np.array(sorted(final_labels))


"""Generating the data"""
 
def generate_data(data):

    values = []
    if(data == "train"):
        images = load_train_images()
    elif(data == "test"):
        images = load_test_images()

    for i in range(len(images)):
        values.append(images[i])
    return np.stack(values)

""" Function for loading the data """ 

def load_data():
    x_train = generate_data("train")
    y_train = generate_labels()
    x_test  = generate_data("test")
    y_test  = generate_labels()

    return (x_train,y_train,x_test,y_test)





if __name__ == "__main__":


    
    x_train, y_train, x_test ,y_test = load_data()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)







