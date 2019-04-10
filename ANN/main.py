import random
import numpy as np

OBSERVED_VALUE = np.random.random((1, 1))  # define observed value
STOPPING_CRITERION = np.array([0.0001])  # define stopping criterion

# define the model of FFBP ANNs
NEURON_IN_INPUT_LAYER = 4  # INPUT = 28  # the number of feature of input
INPUT = np.random.random((NEURON_IN_INPUT_LAYER, 1))
OUTPUT = 1  # the number of output
HIDDEN_LAYER = 1    # the number of hidden layer
NEURON_IN_HIDDEN_LAYER = 3  # number of neutron in the hidden layer


def main():
    # initialize w
    w_ji = np.random.random((NEURON_IN_HIDDEN_LAYER, NEURON_IN_INPUT_LAYER))  # weight between hidden and input layer
    w_kj = np.random.random((OUTPUT, NEURON_IN_HIDDEN_LAYER))  # weight between output and hidden layer
    # calculate input and output of each layer
    input_j = w_ji.dot(INPUT)   # input of neuron in the hidden layer
    output_j = sigmoid(input_j)  # output of neuron in the hidden layer
    input_k = w_kj.dot(output_j)    # input of neuron in the hidden layer
    output_k = sigmoid(input_k)  # output of neuron in the output layer(predicted value)
    c = cost(output_k)  # difference between predict and observed value
    counter = 1
    while c > STOPPING_CRITERION:
        # update weight matrix (use Back Propagation)
        delta = cost(output_k) * sigmoid_prime(input_k)
        w_kj -= delta.dot(output_j.T)  # update weight between output and hidden layer
        print("w_kj:", w_kj)
        delta = (w_kj.T.dot(delta)) * sigmoid_prime(input_j)
        print("delta_update:", delta)
        w_ji -= delta.dot(INPUT.T)  # update weight between hidden and input layer
        print("w_ji:", w_ji)
        input_j = w_ji.dot(INPUT)  # update input of neuron in the hidden layer
        output_j = sigmoid(input_j)  # update output of neuron in the hidden layer
        input_k = w_kj.dot(output_j)  # update input of neuron in the output layer
        output_k = sigmoid(input_k)  # update output of neuron in the output layer (predicted value)
        c = cost(output_k)
        print("c:", c)
        counter += 1
        print("counter:", counter, "\n")


def sigmoid(x):
    # define activation function
    return 1 / (1+np.exp(-x))


def sigmoid_prime(x):
    # calculate the derivative of sigmoid activation function
    return sigmoid(x) * (1 - sigmoid(x))


def cost(x):
    # define cost function
    return 0.5 * np.power(x - OBSERVED_VALUE, 2)


main()
