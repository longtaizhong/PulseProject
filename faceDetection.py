import numpy as np 
import pandas as pd 
import cv2
import time
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.feature import hog
import os
import re
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report , accuracy_score


t1=time.time()
folderpath="../input/pins-face-recognition/pins/PINS/"
cascade = "../input/haarcascadefrontalfaces/haarcascade_frontalface_default.xml"
height=128
width=64
data=[]
labels=[]
Celebs=[]

for dirname,_, filenames in tqdm(os.walk(folderpath)):
    for filename in filenames:
        image = cv2.imread(os.path.join(dirname, filename))
        image= cv2.resize(image , (width,height))
        labels.append(dirname.split("/")[-1])
        data.append(image)


fig = plt.figure(figsize=(20,15))

for i in range(1,10):
    index = random.randint(0,10769) 
    plt.subplot(3,3,i)
    plt.imshow(data[index])
    plt.xlabel(labels[index].split("_")[1])
plt.show()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Labels= le.fit_transform(labels)

data_gray = [cv2.cvtColor(data[i] , cv2.COLOR_BGR2GRAY) for i in range(len(data))]


fig = plt.figure(figsize=(20,15))

for i in range(1,10):
    index = random.randint(1,10770) #https://www.pythoncentral.io/how-to-generate-a-random-number-in-python/
    plt.subplot(3,3,i)
    plt.imshow(data_gray[index])
    plt.xlabel(Labels[index])
plt.show()



Labels = np.array(Labels).reshape(len(Labels),1)

ppc =8
cb=4
hog_features=[]
hog_image=[]
for image in tqdm(data_gray):
    fd , hogim = hog(image , orientations=9 , pixels_per_cell=(ppc , ppc) , block_norm='L2' , cells_per_block=(cb,cb) , visualize=True)
    hog_image.append(hogim)
    hog_features.append(fd)


fig = plt.figure(figsize=(20,15))

for i in range(1,10):
    index = random.randint(1,10770)
    plt.subplot(3,3,i)
    plt.imshow(hog_image[index])
    plt.xlabel(Labels[index])
plt.show()


hog_features = np.array(hog_features)
df = np.hstack((hog_features,Labels))


X_train , X_test , Y_train , Y_test = train_test_split(df[:,:-1] ,
                                                       df[:,-1], 
                                                       test_size=0.3 , 
                                                       random_state=0 , 
                                                       stratify=df[:,-1])


from sklearn.decomposition import PCA
t= time.time()
pca = PCA(n_components=150 , svd_solver='randomized' , whiten=True).fit(X_train)
print("Time evolved", time.time()-t)

print("Projecting the input data on the orthonormal basis")
t0 = time.time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time.time() - t0))


t3=time.time()
svm = SVC(kernel='rbf' , class_weight='balanced' , C=1000 , gamma=0.0082)
svm.fit(X_train_pca , Y_train)
print(svm.score(X_test_pca , Y_test))
print("done in %0.3fs" % (time.time() - t3))

print("total time evolved", (time.time()-t1))



