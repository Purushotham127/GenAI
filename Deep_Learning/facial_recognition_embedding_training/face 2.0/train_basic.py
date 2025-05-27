import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Activation,Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import  Flatten, Dense, Dropout, BatchNormalization

img_height,img_width=(128,128)
train_data_dir=r"face_dataset/train1"
test_data_dir=r"face_dataset/test1"
train_datagen = ImageDataGenerator(shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(img_height,img_width),batch_size=3,class_mode='categorical')
test_generator = train_datagen.flow_from_directory(test_data_dir,target_size=(img_height,img_width),batch_size=1,class_mode='categorical')
# Create a sequential model
base_model = tf.keras.applications.MobileNet(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(3, activation='sigmoid'))
#model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_generator,epochs=40,validation_data=test_generator)
model.evaluate(test_generator)
#model.save(r"Facerecognition_real_1.h5")
model.save(r"19_03.h5")
