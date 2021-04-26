import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2

#from keras.models import Sequential 
#from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
#from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

#---base data paths-------------------------
DATA_BASE = 'COVID-19_Radiography_Dataset'
COVID_IMG = 'COVID-19_Radiography_Dataset/COVID'
NORMAL_IMG = 'COVID-19_Radiography_Dataset/Normal'
COVID_META = 'COVID-19_Radiography_Dataset/COVID.metadata.xlsx'
NORMAL_META = 'COVID-19_Radiography_Dataset/Normal.metadata.xlsx'

#build images df for training and test (with image preprocessing)
def build_df(img_dir, ref_df):
    img_list = []
    colname = ref_df.columns[0]
    for i in range(1,len(ref_df.index)-1):
        img = cv2.imread(img_dir + '/' + ref_df[colname][i] + '.png',0)
        if img is not None:
            re_img = cv2.resize(img, (64,64))
            img_arr = tf.keras.preprocessing.image.img_to_array(re_img)/255
            img_list.append(img_arr)

    img_nd = np.asarray(img_list)
    return img_nd

covid_meta_df = pd.read_excel(COVID_META, usecols=[0])
normal_meta_df = pd.read_excel(NORMAL_META, usecols=[0])
covid_meta_df['TARGET'] = "1"
#covid_meta_df['FILE NAME'] = COVID_IMG + '/' + covid_meta_df['FILE NAME'] + '.png'
normal_meta_df['TARGET'] = "0"
#normal_meta_df['FILE NAME'] = NORMAL_IMG + '/' + normal_meta_df['FILE NAME'] + '.png'

covid_train_df = covid_meta_df.iloc[:3000,:]
covid_test_df = covid_meta_df.iloc[3001:3600,:]
covid_test_df = covid_test_df.reset_index(drop=True)

covid_train_img_df = build_df(COVID_IMG,covid_train_df)
#print(covid_train_img_df.shape)
covid_test_img_df = build_df(COVID_IMG,covid_test_df)

normal_train_df = normal_meta_df.iloc[:1000,:]
normal_test_df = normal_meta_df.iloc[1001:1600,:]
normal_test_df = normal_test_df.reset_index(drop=True)

normal_train_img_df = build_df(NORMAL_IMG,normal_train_df)
#print(normal_train_img_df.shape)
normal_test_img_df = build_df(NORMAL_IMG,normal_test_df)

covid_train_df['FILE NAME'] = COVID_IMG + '/' + covid_train_df['FILE NAME'] + '.png'
covid_test_df['FILE NAME'] = COVID_IMG + '/' + covid_test_df['FILE NAME'] + '.png'
normal_train_df['FILE NAME'] = NORMAL_IMG + '/' + normal_train_df['FILE NAME'] + '.png'
normal_test_df['FILE NAME'] = NORMAL_IMG + '/' + normal_test_df['FILE NAME'] + '.png'

train_df = pd.concat([covid_train_df, normal_train_df], axis=0, ignore_index=True)
test_df = pd.concat([covid_test_df,normal_test_df], axis=0, ignore_index=True)
train_img = np.append(covid_train_img_df, normal_train_img_df, axis=0)
test_img = np.append(covid_test_img_df, normal_test_img_df, axis=0)

#train_img_df = covid_train_img_df.append(normal_train_img_df)
#test_img_df = covid_test_img_df.append(normal_test_img_df)

""" print(train_img.shape)
print(train_df.head())
print(train_df.tail())
print(test_df.head())
print(test_df.tail()) """



#data augmentation
train_dgen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    horizontal_flip = True,
    zoom_range=0.3
)

test_dgen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

train_generator = train_dgen.flow_from_dataframe(
        train_df, 
        directory='./',
        x_col='FILE NAME',
        y_col='TARGET',
        target_size=(64, 64),
        batch_size = 32,
        color_mode = "grayscale",
        class_mode= "binary")

test_generator = train_dgen.flow_from_dataframe(
        test_df, 
        directory='./',
        x_col='FILE NAME',
        y_col='TARGET',
        target_size=(64, 64),
        batch_size = 32,
        color_mode = "grayscale",
        class_mode= "binary")


#---model----------------------------------
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32,(3,3),input_shape = (64,64,1)))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(32,(3,3)))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(64,(3,3)))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation("sigmoid"))

model.compile(loss = "binary_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)
history = model.fit(
                    train_generator,
                    steps_per_epoch=4000//32,
                    epochs=20,
                    validation_data=test_generator,
                    validation_steps=624//32)
                    
                    #callbacks=[early_stopping])

#change to evaluate
print("Accuracy:" , model.evaluate_generator(test_generator)[1]*100 , "%")
print("Loss:" , model.evaluate_generator(test_generator)[0])

#---convert model to .tflite-----------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
