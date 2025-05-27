import cv2
from PIL import Image
c=1
f='face_dataset/train/purushotham/name'
d='face_dataset/train2/purushotham'
while(c!=301):
	#f1=f+str(c)+'.jpg'
	#f2=cv2.imread(f1)
	f3=cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
	f4=d+str(c)+'.jpg'
	c+=1
	cv2.imwrite(f4, f3)
