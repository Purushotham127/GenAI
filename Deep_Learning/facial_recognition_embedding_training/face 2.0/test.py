import numpy as np
#from keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import cvlib as cv
classes = ['baban','chanakya','purushotham']
model = load_model('Facerecognition_real.h5')
webcam = cv2.VideoCapture(0)
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
                        #cv2.imwrite("outputimage.jpg", face_crop)
                        face_crop = cv2.resize(face_crop, (128,128))
                        face_crop = face_crop.astype("float") / 255.0
                        face_crop = img_to_array(face_crop)
                        
                        #cv2.imwrite("outputimage.jpg", face_crop)
                        face_crop = np.expand_dims(face_crop, axis=0)
                        conf = model.predict(face_crop)
                        idx = np.argmax(conf)
                        label = classes[idx]
                        print(label)
                        #label = "{}: {:.2f}%".format(label, conf[idx] * 100)
                        Y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
                        cv2.imshow("face detection", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                        	break
webcam.release()
cv2.destroyAllWindows()
