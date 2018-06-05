#this code is to teach machine how to count binary number as we know there is no fix rule of counting binary as in 1,
#1 appears in unit place ,in 2 it appers in second place but in three it appears in both places so it is hard to find out any
#sequence among them but only in 100 iteration it got 10 out of 15 right.
#this is the power of reccurent neural network as this question cannot be slver by simple neural network 
#counting binary is not an easy task as there are rules and logic to count


#importinf numpy
import numpy as np
# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset this is binary counting from zero to 14
X = np.array([
                [0,0,0,0],
                [0,0,0,1],
                [0,0,1,0],
                [0,0,1,1],
                [0,1,0,0],
                [0,1,0,1],
                [0,1,1,0],
                [0,1,1,1],
                [1,0,0,0],
                [1,0,0,1],
                [1,0,1,0],
                [1,0,1,1],
                [1,1,0,0],
                [1,1,0,1],
                [1,1,1,0],
                [1,1,1,1]
            ])
#this is output which gives input +1 or it will try to give next number in the sequence
y=np.array([
                [0,0,0,1],
                [0,0,1,0],
                [0,0,1,1],
                [0,1,0,0],
                [0,1,0,1],
                [0,1,1,0],
                [0,1,1,1],
                [1,0,0,0],
                [1,0,0,1],
                [1,0,1,0],
                [1,0,1,1],
                [1,1,0,0],
                [1,1,0,1],
                [1,1,1,0],
                [1,1,1,1],
                [0,0,0,0]
            ])
#seeding random numbers 
np.random.seed(1)
#initalizing synaps connecting input layer to the hidden layes
syn0=np.random.random((4,4))
#initializing synaps connecting hidden layer to reccurent layer
synh=np.random.random((4,4))
#initializing a random reccurent layer value
h=np.array([
            [0,0,0,0]
            ])
#training loop
for iter in range(100):
    for i in range(len(X)):
        l0=np.array([X[i]])
        #forward propagation
        l1 = nonlin(np.dot(l0,syn0)+(np.dot(h,synh)))
        #initializing activation layer to reccurent layer
        h=l1
        print(X[i])
        print(np.round(l1))
        #calculating error
        l1_error = np.array(y[i]) - l1
        #calculating gradient
        l1_delta = l1_error * nonlin(l1,True)
        #updating weights
        syn0 += np.dot(l0.T,l1_delta)
        synh += np.dot(h.T,l1_delta)





