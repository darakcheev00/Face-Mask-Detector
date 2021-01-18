import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



DIRECTORY = r"./photos2"
CATEGORIES = ["with_mask", "without_mask"]
IMG_SIZE = 100
EPOCHS = 20

photos = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img))
            img_array_resized = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
            photos.append(img_array_resized)
            labels.append(class_num)
        except Exception as e:
            pass
            

X_train, X_test, y_train, y_test = train_test_split(photos, labels, test_size=0.1,random_state=42, stratify=labels)

X_train = np.array(X_train).reshape(-1,IMG_SIZE,IMG_SIZE,3)
X_test = np.array(X_test).reshape(-1,IMG_SIZE,IMG_SIZE,3)

y_train = np.array(y_train)


X_train = X_train/255.0
X_test = X_test/255.0

#Model
model = Sequential()

model.add(Conv2D(64,(3,3),input_shape=X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),input_shape=X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),input_shape=X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

#output layer
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

M = model.fit(X_train, y_train, batch_size=32, validation_split=0.2, epochs=EPOCHS,callbacks=[early_stop])


import pandas as pd
model_loss = pd.DataFrame(model.history.history)
#model_loss.plot()

predictions = model.predict(X_test)

model.save("mask_detector.model", save_format="h5")


# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), M.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), M.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), M.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), M.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig("plot.png")


#calculate accuracy
masks_correct = 0
no_mask_correct = 0
no_mask__mask = 0
mask__no_mask = 0

length = len(predictions)
for i in range (length):
    pred = int(round(predictions[i,0]))
    if y_test[i] == pred:
        if y_test[i] == 1:
            masks_correct += 1
        else:
            no_mask_correct += 1
    else:
        if y_test[i] == 1:
            mask__no_mask += 1
        else:
            no_mask__mask += 1
            
print ("masks_correct = "+str(masks_correct))
print ("no_mask_correct = "+str(no_mask_correct))
print ("no_mask__mask = "+str(no_mask__mask))
print ("mask__no_mask = "+str(mask__no_mask))
accuracy = (masks_correct+no_mask_correct)/(length)
print ('\n')
print ("accuracy = "+str(accuracy))
