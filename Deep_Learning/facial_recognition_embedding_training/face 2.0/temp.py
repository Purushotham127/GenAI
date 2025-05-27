import numpy as np
import cv2
import os
import cvlib as cv
import datetime
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
model = load_model('19_03.h5')
webcam = cv2.VideoCapture(0)
mypath='name2.jpg'
Retainal = ['chanakya','purushotham']
c=1
l=[0,0]
co=[0,0]
def fun(name,val):
	f=open('log_time.txt','a')
	if(val<4 and co[val]==0):
		f.write(name+' entered : '+str(datetime.datetime.now())+'\n')
		l[val]=str(datetime.datetime.now())
		co[val]+=1
	elif(val<4 and co[val]==1):
		l[val]=str(datetime.datetime.now())
	else:
		f.write(name+' found at '+str(datetime.datetime.now())+'\n')
	f.close()
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
	test_image = Image.open(mypath)
	test_image = img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis=0)
	result = model.predict(test_image)
	t=np.max(result[0])
	print(t)
	if(t<0.3):
	#if(cv2.waitKey(1) & 0xFF == ord('c')):				#defining accuracy
		preds='Unknown'
		#if(cv2.waitKey(1) & 0xFF == ord('q')):
			#break
		if(preds=='Unknown'):
			imgname='Unknown_User'+str(c)+'.jpg'
			path='Unknown_faces/'
			cv2.imwrite(path+imgname,frame)   #image given by the frame
			fun('Unknown User'+str(c), 4) #above function is called to write into log
			c+=1
	else:
		val = np.argmax(result)
		preds = Retainal[val]
		if(co[val]==4):
			imgname=str(preds)+'.jpg'
			path='Known_faces/'
			cv2.imwrite(path+imgname,frame)   #image given by the frame
		fun(str(preds),val) #above function is called to write into log
	#label = "{}: {:.2f}%".format(preds, result[val] * 100)
	Y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.putText(frame, preds, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
	cv2.imshow("face detection", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
webcam.release()
cv2.destroyAllWindows()
d=-1
f=open('log_time.txt','a')
for i in l:
	d+=1
	if(i==0):
		continue
	else:
		f.write(Retainal[d]+' exit : '+i+'\n')
f.close()
