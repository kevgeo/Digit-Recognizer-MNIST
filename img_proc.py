import sys		
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import network
from PIL import Image
from numpy import array
import pickle
#----------------------------------------------------------------------------------
################################## ML Part ########################################
import network
import mnist_loader

training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()
net = network.Network([784, 100, 10])
net.SGD(training_data, 40, 10, 3.0, test_data=test_data)
pickle.dump(net.biases,open("biases2.npy","wb"))
pickle.dump(net.weights,open("weights2.npy","wb"))
#------------------------------------------------------------------------------------
#path = input("ENter image path")
image = cv2.imread("images/img19.jpg")
#def upload(pat):
#	path=pat
#image = cv2.imread(path)

#-> outputs rows,columns, and channel-if 3 means RGB
print image.shape

#-> conversion to grayscale
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#-> apply Gaussian filtering to remove noisy pixels
im_gray = cv2.GaussianBlur(gray_image, (5, 5), 0)

#hist = cv2.calcHist([gray_image],[0],None,[256],[0,256])
#plt.hist(gray_image.ravel(),256,[0,256])
#plt.title('Histogram for gray scale picture')
#plt.show()


#-> threshold the image, helps to seperate or segment foreground from background
#-> output is a binary image, white pixels-foreground, black pixels - background
#ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

im_th = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 10) 
#im_th = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 7)

#-> Find countours in image,
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

#-> Getting rectangles for each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

for rect in rects:
	#print "Rect[0]={0},Rect[1]={1},Rect[2]={2},Rect[3]={3}".format(rect[0],rect[1],rect[2],rect[3])
	#-> Rect[0]-represents x coordinate,
	#   Rect[1]-represents y coordinate
	#-> To make sure that rectangles are drawn only on digits
	if rect[2] > 20:
		cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
		leng = int(rect[3] * 1.1)
		#-> // is floor division in python
		pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
		pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
		#-> For roi, we use pt1 for row since row represent y coordinates
		#   pt2 for column as column represent x coordinates
		roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
		#print roi.shape
		roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA) 
		roi = cv2.dilate(roi, (3, 3))
		img = Image.fromarray(roi)
		img.save("output.png")
		img = Image.open("output.png")
		arr = np.array(img)
		arr3 = np.reshape(arr,784)
		arr3 = np.reshape(arr3,(784,1))
		num = np.argmax( net.feedforward(arr3))
		cv2.putText(image, str(int(num)), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
		#cv2.imshow("roi",roi)
		#cv2.waitKey(0)


#cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
#cv2.imshow("digits",image)
#cv2.imshow("digits-gray",gray_image)
#cv2.imshow("digits-Gaussian blur",im_gray)
#cv2.imshow("digits-threshold1",im_th)
#cv2.imshow("digits-threshold2",im_th)
cv2.imshow("digits-rectangular roi",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
		