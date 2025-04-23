import numpy as np

# define sigmoid avtivation function
def sigmoid(x):
 return 1 / (1 + np.exp(-x))
 
 def relu(x):
    return np.maximum(0, x)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s)
 
class NeuralUnit:

    def __init__(self, params, offset):
        # Initialize the parameters (weights) and threshold (bias)
        self.__internal_params = params  
        self.__threshold_shift = offset 

    def process_inputs(self, input_vector):
        # Compute the weighted combination of inputs and parameters (each input multiplied by the weight)
       
        input_vector = np.array(input_vector)
        combined_signal = np.dot(input_vector, self.__internal_params)

        # Adjust by the threshold (bias)
        adjusted_signal = combined_signal + self.__threshold_shift

        # Apply the Sigmoid activation function and relU
        final_output = sigmoid(adjusted_signal)
        final_output = relu(activated_signal)
        
        return final_output

        
