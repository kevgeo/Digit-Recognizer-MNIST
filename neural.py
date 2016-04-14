#Implementing Neural Network
import sys

import PIL.Image
import scipy.misc, scipy.optimize, scipy.io, scipy.special
from numpy import *

import pylab
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab



def displayData( X, theta = None ):
	width = 20
	rows, cols = 10, 10
	out = zeros(( width * rows, width*cols ))
	#out array sze is 200x200
	#numpy-returns a permuted range.
	rand_indices = random.permutation( 5000 )[0:rows * cols]

	counter = 0
	for y in range(0, rows):
		for x in range(0, cols):
			start_x = x * width
			start_y = y * width
			#in below statement, : is a delimeter. e.g 1:5 gives from 1 t0 5(5 not included)
			out[start_x:start_x+width, start_y:start_y+width] = X[rand_indices[counter]].reshape(width, width).T
			counter += 1

	img 	= scipy.misc.toimage( out )
	figure  = pyplot.figure()
	axes    = figure.add_subplot(111)
	axes.imshow( img )


	if theta is not None:
		result_matrix 	= []
		X_biased 		= c_[ ones( shape(X)[0] ), X ]
		
		for idx in rand_indices:
			result = (argmax( theta.T.dot(X_biased[idx]) ) + 1) % 10
			result_matrix.append( result )

		result_matrix = array( result_matrix ).reshape( rows, cols ).transpose()
		print result_matrix

	pyplot.show( )

def sigmoid( z ):
	return scipy.special.expit(z)
		# return 1.0 / (1.0 + exp( -z ))


def computeCost( theta, X, y, lamda ):
	m = shape( X )[0]
	#m is basically the number of rows or data
	# X size-5000*401
	# theta size-401*
	# y size- 5000*, it is our temp_y which is boolean type	
	hypo 	   = sigmoid( X.dot( theta ) )
	#hypo size- 5000*
	#Now calculating cost function
	term1 	   = log( hypo ).dot( -y )
	#term1 is just one value as by vector multiplication, result is one value	
	term2 	   = log( 1.0 - hypo ).dot( 1 - y )
	#term2 is also just one value
	left_hand  = (term1 - term2) / m
	right_hand = theta.T.dot( theta ) * lamda / (2*m)
	# right-hand is basically the regularisation term
	return left_hand + right_hand

def gradientCost( theta, X, y, lamda ):
	m = shape( X )[0]
	grad = X.T.dot( sigmoid( X.dot( theta ) ) - y ) / m
	grad[1:] = grad[1:] + ( (theta[1:] * lamda ) / m )
	return grad


def oneVsAll( X, y, num_classes, lamda ):
	m,n 		= shape( X )
	#shape returns tuple of array dimensions	
	X 			= c_[ones((m, 1)), X]
	#creates an array of ones, size m*1 where m=5000
	#This array is appended to X, so it becomes the first column of X
	#So X size- 5000*401
	all_theta 	= zeros((n+1, num_classes))
	#array of zeroes, size n(400)+1 * 10

	for k in range(0, num_classes):
		theta 			= zeros(( n+1, 1 )).reshape(-1)
		#theta size- 401*1
		temp_y 			= ((y == (k+1)) + 0).reshape(-1) 
		# (y==k+1) returns boolean array, size- 5000*1
		#baiscally in each for loop . it is checking if value in y is equal to k+1
		#if yes, then true or else false, this is how we build our output vector for each digit
		# temp_y size- 5000*, values are just zeroes

		result 			= scipy.optimize.fmin_cg( computeCost, fprime=gradientCost, x0=theta, \
												  args=(X, temp_y, lamda), maxiter=50, disp=False, full_output=True )
		#Minimize a function using the downhill simplex algorithm.
		#In the paramters, fprime is a function that returns the gradient of f at x.
		#fprime - A function that returns the gradient of f at x. In this case, f is computeCost
		all_theta[:, k] = result[0]
		print "%d Cost: %.5f" % (k+1, result[1])

	# save( "all_theta.txt", all_theta )
	return all_theta


def predictOneVsAll( theta, X, y ):
	# the theta passed in this function is the
	# optimal values for each digit
	m,n = shape( X )
	X 	= c_[ones((m, 1)), X]
	correct = 0
	for i in range(0, m ):
		prediction 	= argmax(theta.T.dot( X[i] )) + 1
		# argmax returns index, theta.T.dot( X[i] ) returns shape 10*
		# shape(theta) is 401*10, shape(X[i]) is 401*
		print "shape:", shape(theta),shape(X[i])
		print "t:",theta.T.dot( X[i] )
		actual 		= y[i]
		# print "prediction = %d actual = %d" % (prediction, actual)
		if actual == prediction:
			correct += 1
	print "Accuracy: %.2f%%" % (correct * 100.0 / m )




def part1():
	data = scipy.io.loadmat("data1.mat")
	X, y 			= data['X'], data['y']
	#print data
	#X contain the pixel values, y is the index telling which digit it is 
	#X dimensions is 5000x400
	m, n = shape(X)
	num_labels = 10
	lamda = 0.1
	theta = oneVsAll(X, y, num_labels, lamda) 
	predictOneVsAll( theta, X, y )
	#displayData( X )
	#converting mat to csv file
	#for i in data:
	#	if '__' not in i and 'readme' not in i:
	#		np.savetxt(("file.csv"),data[i],delimiter=',')

def main():
	part1()

if __name__ == '__main__':
		main()