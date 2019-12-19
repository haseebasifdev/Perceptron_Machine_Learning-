import numpy as np
import pandas as pd
import sys

data = pd.read_csv("datafile.csv")
inputs = np.array(data[['x1', 'x2']])
inputs.tolist()
output = np.array(data[['y']])
output.tolist()


class Perceptron(object):
    """Implements a perceptron network"""

    def __init__(self, numberOfRows, lrnRt=1, epochs=100):
        self.W = np.zeros(numberOfRows+1)
        self.epochs = epochs
        self.lr = lrnRt

    def Activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, input):
        print("Inputs ", input)
        input = np.insert(input, 0, 1)
        print("Inputs with Bias", input)
        sum = self.W.dot(input)
        ActivationOutput = self.Activation(sum)
        print("Activation Function output: ", ActivationOutput)
        # print("Return Value from Activation funtion ", ActivationOutput)
        return ActivationOutput

    def fit(self, X, actualOutput):
        for epoch in range(self.epochs):
            print("================No of Epoches================ ", epoch)
            for i in range(actualOutput.shape[0]):
                predicted = self.predict(X[i])
                err = actualOutput[i] - predicted
                print("Error", err)
                self.W = self.W + self.lr * err * np.insert(X[i], 0, 1)
                print("Updated Weights: ", self.W)


perceptron = Perceptron(numberOfRows=2)


def learn():
    perceptron.fit(inputs, output)
    print("Final Weights", perceptron.W)


# if __name__ == '__main__':

if sys.argv[1] == "--learn":

    learn()
elif sys.argv[1] == "--test":
    learn()
    print("Testing Starts here")
    for i in range(output.shape[0]):
        print("output: ", output[i])
        perceptron.predict(inputs[i])
else:
    print("No argument are given")
