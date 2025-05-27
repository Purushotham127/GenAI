'''import cv2
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not access the webcam")
else:  
    path = './data/train/b/'
    counter=1
    for i in range(100):
    	ret, frame = cap.read()
    	filename = 'name'+str(counter)+'.jpg'
    	counter+=1
    	cv2.imwrite(path + filename, frame)
    	print("Image saved successfully ",ret)
    cap.release()
    '''
    
import numpy as np
c=0
import cv2
import cvlib as cv
webcam = cv2.VideoCapture(0)
while webcam.isOpened():
                        status, frame = webcam.read()
                        face, confidence = cv.detect_face(frame)
                        for idx, f in enumerate(face): 
                                (startX, startY) = f[0], f[1]
                                (endX, endY) = f[2], f[3]
                        face_crop = np.copy(frame[startY:endY,startX:endX])
                        #if(face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                        #	continue
                        face_crop = cv2.resize(face_crop, (128,128))
                        #f3=cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        if(c==60):
                        	break
                        else:
                        	c+=1
                        	print(c)
                        cv2.imwrite("su/name"+str(c)+".jpg", face_crop)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                        	break
webcam.release()
cv2.destroyAllWindows()
