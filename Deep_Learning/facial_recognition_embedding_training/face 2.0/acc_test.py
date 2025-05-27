import numpy as np
import cv2
import cvlib as cv
#import tensorflow as tf
#from PIL import Image
#from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
model = load_model('Facerecognition3.h5')
webcam = cv2.VideoCapture(0)
mypath='name2.jpg'
Retainal = ['baban','chanakya','purushotham']
while webcam.isOpened():
	status, frame = webcam.read()
	face, confidence = cv.detect_face(frame)
	for idx, f in enumerate(face):
		(startX, startY) = f[0], f[1]
		(endX, endY) = f[2], f[3]
	cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
	face_crop = np.copy(frame[startY:endY,startX:endX])
	if(face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
		continue
	face_crop = cv2.resize(face_crop, (128,128))
	#face_crop=cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
	cv2.imwrite(mypath, face_crop)
	test_image = cv2.imread(mypath)
	#test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
	test_image = img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis=0)
	result = model.predict(test_image)
#	decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(result, top=5)
#	for _, label, confidence in decoded_predictions[0]:
#		print(f'{label}: {confidence * 100}%')
	#val = np.argmax(result)
	#print(val)
	#preds = Retainal[val]
	t=np.max(result)
	print(t)
	#if t < 0.1:
		#preds='unknown'
	#if t>0.1:
	val = np.argmax(result)
	print(val)
	preds=Retainal[val]
	#label = "{}: {:.2f}%".format(preds, result[val] * 100)
	Y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.putText(frame, preds, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
	cv2.imshow("face detection", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
webcam.release()
cv2.destroyAllWindows()
