import numpy as np

class MyPerceptron:

    def __init__(self, learning_rate=0.1, n_iterations=1000):

        self.lr=learning_rate
        self.epochs =n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epochs):

            for i in range(X.shape[0]):

                y_pred = self.activation_function(np.dot(self.weights,X[i]) + self.bias)

                self.weights = self.weights + self.lr * (y[i] - y_pred)*X[i]
                self.bias = self.bias + self.lr * (y[i] - y_pred)

        print("Training complete")

        print(self.weights)
        print(self.bias)

    def activation_function(self, activation):

        if activation>=0:
            return 1
        else:
            return 0

    def predict(self,X):

        y_pred=[]

        for i in range(X.shape[0]):

            y_pred.append(self.activation_function(np.dot(self.weights,X[i]) + self.bias))

        return np.array(y_pred)


