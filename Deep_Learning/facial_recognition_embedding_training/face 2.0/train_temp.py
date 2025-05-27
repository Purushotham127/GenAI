import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Activation,Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import  Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import backend as K
img_height,img_width=(128,128)
train_data_dir=r"face_dataset/train2"
test_data_dir=r"face_dataset/test2"
train_datagen = ImageDataGenerator(shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(img_height,img_width),batch_size=3,class_mode='categorical')
test_generator = train_datagen.flow_from_directory(test_data_dir,target_size=(img_height,img_width),batch_size=1,class_mode='categorical')
# Create a sequential model
#base_model = tf.keras.applications.MobileNet(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
model = Sequential()
#model.add(base_model)
model.add(Conv2D(32, (3,3), padding="same", input_shape=(128,128,3)))		#input layer and first 
model.add(Activation("relu"))
model.add(BatchNormalization())   #removed axis=0 or -1 or 1 onwards
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25)) 							 #1 hidden layer complete
model.add(Conv2D(64, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3))) #added
model.add(Dropout(0.25))  #added
model.add(Conv2D(64, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3)))     #changed  from (2,2) onwards
model.add(Dropout(0.25))							#2
model.add(Conv2D(128, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))  #added
model.add(Conv2D(128, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))							#3
model.add(Flatten())
model.add(Dense(1024))								#Dense layer
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5)) #changed from 0.5					#Output layer with 3 neurons
model.add(Dense(3))
model.add(Activation("sigmoid"))#model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_generator,epochs=40,validation_data=test_generator)
model.evaluate(test_generator)
model.save(r"10_04.h5")
