from neuron import sigmoid
from neuron import sigmoid_derivative
from neuron import relu
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size=16, output_size=1):
        #make the weights go brrr
        self.w1 = np.random.rand(input_size, hidden_size)*np.sqrt(2.0/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.rand(hidden_size, output_size)*np.sqrt(2.0/input_size)
        self.b2= np.zeros((1, output_size))

    
    def relu_derivative(self, z):

        return (z > 0).astype(float) 

    def feedforward(self, x):

        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = relu(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2
    def backprop(self, x, y, learning_rate): # Note to self : "You're shit at calculating dimensions"
        m = x.shape[0]
        #Thank me later for the assert methods as they will save yourlife if there are any dimensionality related bugs. Mustafa is the goat
        dz2 = self.a2 - y 
        assert dz2.shape == (m, 1), f"dz2 should be (m,1), got {dz2.shape}. Check w2/b2 shapes!"
        dw2 = (1/m) * np.dot(self.a1.T, dz2)
        assert dw2.shape == self.w2.shape, f"dw2 {dw2.shape} vs w2 {self.w2.shape}"
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)

        dz1 = np.dot(dz2, self.w2.T) * self.relu_derivative(self.z1)
        assert dz1.shape == self.z1.shape, f"dz1 {dz1.shape} vs z1 {self.z1.shape}"
        dw1 = (1/m) * np.dot(x.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis = 0, keepdims=True)
    
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1

        

    def compute_loss(self, y_true, y_pred):
        #MSE
        #return ((y_true-y_pred)**2).mean()

        #Binary cross entropy loss function
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1-eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1-y_pred))
    
    def predict(self, x, threshold=0.3):
        return (self.feedforward(x) > threshold).astype(int)