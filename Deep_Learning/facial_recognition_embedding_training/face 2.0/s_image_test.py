import numpy as np
#import tensorflow as tf
from PIL import Image
#from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


mypath='name35.jpg'
Retainal = ['baban','chanakya','purushotham']
def upload():
            new_model = load_model('temporary_fr.h5')
            test_image = Image.open(mypath)
            #test_image=test_image.resize((128,128))
            test_image = img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = new_model.predict(test_image)
            print(result)
            print('............................')
            print(sum(result[0]))
            t=np.max(result)
            print(t)
            val = np.argmax(result[0])
            #print(val)
            preds = Retainal[val]
            print(preds)
upload()
