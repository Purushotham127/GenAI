#code to print time log for the user to an ext file
import datetime
#print(datetime.datetime.now())
def fun(name):
	f=open('log_time.txt','a')
	f.write(name+str(datetime.datetime.now())+'\n')
fun('Unknown User1, time : ')


#code to write unknown images to a directory
c=1
if(preds=='Unknown'):
	path='Unknown_faces'
	imgname='Unknown_User'+str(c)+'.jpg'
	cv2.imwrite(path+imgname,img)   #image given by the frame 
	fun('Unknown User'+str(c)+',time : ') #above function is called to write into log
	c+=1
