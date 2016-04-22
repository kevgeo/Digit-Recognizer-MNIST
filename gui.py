from Tkinter import *
from tkFileDialog import askopenfilename
import cv2
import tkFileDialog
from user import test
import network
#from user import test

top = Tk()
top.title('What\'s Thats digit')
def uploadcallback():
    root = Tk()
    root.withdraw()
    file_path = tkFileDialog.askopenfilename()
    print file_path
    User = test(file_path)
    #image = User.load_testImage()
    User.img_proc()
#    Upload(file_path)
def capturecallback():
    camera_port = 0
    camera = cv2.VideoCapture(camera_port)
    return_value, image = camera.read()
    cv2.imwrite("opencv.jpg", image)
    del (camera)
label1= Label(top,text='Whats that digit?')
top.geometry('1600x800')
label1.pack()
button1=Button(top,text='Upload',command = uploadcallback)
button1.pack()
Button2=Button(top,text='Camera',command = capturecallback)
Button2.pack()
top.mainloop()
