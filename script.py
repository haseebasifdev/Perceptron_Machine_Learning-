import numpy as np
import pandas as pd
import sys

# Reading data file
data = pd.read_csv("datafile.csv")
# getting input columns
inputs = np.array(data[['x1', 'x2']])
# make a list
inputs.tolist()
# getting output columns
output = np.array(data[['y']])
# output list
output.tolist()

# Calss for perceptron


class Perceptron(object):
    """Implements a perceptron network"""
    # constructer

    def __init__(self, numberOfRows, lrnRt=1, epochs=100):
        # Additind 1 for threshold or Bias value
        # And All Initialize with Zero
        self.W = np.zeros(numberOfRows+1)
        self.epochs = epochs
        # learning rate
        self.lr = lrnRt
    # Activation funtion

    def Activation(self, x):
        return 1 if x >= 0 else 0
    # predicter funtion

    def predict(self, input):
        print("Inputs ", input)
        # inserting 1 for biased input
        input = np.insert(input, 0, 1)
        print("Inputs with Bias", input)
        # Sum of weights with input
        sum = self.W.dot(input)
        # checking for Activation orthreshold
        ActivationOutput = self.Activation(sum)
        print("Activation Function output: ", ActivationOutput)
        # print("Return Value from Activation funtion ", ActivationOutput)
        return ActivationOutput
    #########################################
    # fiting weights values

    def fit(self, X, actualOutput):
        # for loop number of Epoches
        for epoch in range(self.epochs):
            print("================No of Epoches================ ", epoch)
            # Loop for number of columns
            for i in range(actualOutput.shape[0]):
                predicted = self.predict(X[i])
                # error calculation
                err = actualOutput[i] - predicted
                print("Error", err)
                # Updating weights
                self.W = self.W + self.lr * err * np.insert(X[i], 0, 1)
                print("Updated Weights: ", self.W)
    #########################################


###################################################End OF Class###################################################

# Creating object
perceptron = Perceptron(numberOfRows=2)


def learn():
    # calling function for learning
    perceptron.fit(inputs, output)
    print("Final Weights", perceptron.W)


# sys.argv for the arguments given in command line
if sys.argv[1] == "--learn":
    # calling learn funtion
    learn()
elif sys.argv[1] == "--test":
    learn()
    # testing starts
    for i in range(output.shape[0]):
        # output.shapes[0] for loop running according to number of columns in CSV file
        print("output: ", output[i])
        perceptron.predict(inputs[i])
else:
    print("No argument are given")
