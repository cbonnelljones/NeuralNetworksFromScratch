import os
import urllib
import urllib.request
from zipfile import ZipFile
import cv2
import numpy as np
import matplotlib.pyplot as plt

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = URL.split('/')[-1]
FOLDER = FILE.split('.')[0] #'fashion_mnist_images'
FOLDER_TRAIN = FOLDER + "/train"

if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE}...')
    urllib.request.urlretrieve(URL, FILE)

    print('Unzipping images...')
    with ZipFile(FILE) as zip_images:
        zip_images.extractall(FOLDER)

    print('Done!')

labels = os.listdir(FOLDER_TRAIN)

X = []
y = []

for label in labels:
    #print(FOLDER_TRAIN + "/" + label)
    FOLDER_TRAIN_CLASS = FOLDER_TRAIN + "/" + label
    for file in os.listdir(FOLDER_TRAIN_CLASS):
        image = cv2.imread(FOLDER_TRAIN_CLASS + "/" + file, cv2.IMREAD_UNCHANGED)

        X.append(image)
        y.append(label)

##print(labels)
#files = os.listdir(FOLDER_TRAIN + "/0")
#print(files[:10])
#print(len(files))

#image_data = cv2.imread(FOLDER_TRAIN + "/7/0002.png", cv2.IMREAD_UNCHANGED)
#print(np.set_printoptions(linewidth=200))
#print(image_data)

#plt.imshow(image_data, cmap='gray')
#plt.show()