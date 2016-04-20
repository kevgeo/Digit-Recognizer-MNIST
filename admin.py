import network
import mnist_loader
import pickle

training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()
net = network.Network([784, 400, 10])
net.SGD(training_data, 70, 10, 3.0, test_data=test_data)
pickle.dump(net.biases,open("biases8.npy","wb"))
pickle.dump(net.weights,open("weights8.npy","wb"))