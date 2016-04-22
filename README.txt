-----------------------
   Digit Recognition
-----------------------

A GUI application that can process images of digits and identify them.

network.py is the code for our neural network. After training it with the standard MNIST dataset with a learning rate of 3.0 and over 70 epochs,
we achieve an accuracy of 98.1% on the dataset. Architecutre used is a 3 layer network with 784,400,10 neurons respectively in each layer. 

The updated weights and biases of the neural network are stored in a cPickle file. weights8.npy and biases8.npy, containing the weights and biases 
which gave us an accuracy of 98.1% on the dataset.

admin.py contains the code to load the dataset, decide the architecture of the neural network and train it. Weights and biases are stored in a cPickle file.

user.py does the image processing part of extracting each digit from the image and feeding it to the neural network to classify. Since the MNIST 
dataset we used contained images of the size 28x28 pixels, we resize the image of each digit with the help of numpy and PIL library to 28x28 pixels, convert it into a vector 784x1 and feed it to the neural network to classify.

gui.py is a simple interface that makes it oblivious to the user about the background processes that make the whole digit recognition work. It is easy to use and allows you to use an image from the 'images' folder or take a picture from a webcam(delay of 3 seconds is given so that the user can 
position the paper with the digits) and stored with the name 'opencv.jpg'. Move this image to the 'images folder' and then press upload to classify the digits.
