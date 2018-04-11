import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

from testCases import *


class LogisticRegressionDNN:
    """
    define the logistic Regression DNN with one hidden layer
    """
    def __init__(self, learning_rate=0.02, num_iterations=2000, hidden_layer_sizes=4):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.hidden_layer_sizes = hidden_layer_sizes
        self.parameters = {}

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def tanh(self, z):
        return np.tanh(z)

    def initialize_parameters(self, n_x, n_h, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        
        Results:
        params -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """
        np.random.seed(2)

        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros(shape=(n_h, 1))

        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros(shape=(n_y, 1))

        assert (W1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (W2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))
        
        parameters = {
            'W1':W1,
            'b1':b1,
            'W2':W2,
            'b2':b2
        }

        self.parameters = parameters
    
    def forward_propagation(self, X):
        """
        Argument:
        X -- input data of size (n_x, m)
        
        Returns:
        Z1, A1, Z2 and A2
        """
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']

        Z1 = np.dot(W1, X) + b1
        A1 = self.tanh(Z1)

        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)

        assert(A2.shape == (1, X.shape[1]))

        return Z1, A1, Z2, A2

    def compute_cost(self, Y, A2):
        """
        Computes the cross-entropy cost given in equation (13)
        
        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        parameters -- python dictionary containing your parameters W1, b1, W2 and b2
        
        Returns:
        cost -- cross-entropy cost given equation (13)
        """
        m = Y.shape[1]

        logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
        cost = -1 / m * np.sum(logprobs, axis=1, keepdims=True)

        cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                    # E.g., turns [[17]] into 17 
        cost = np.float(cost)
        
        assert(isinstance(cost, float))
        
        return cost
    
    def backward_propagation(self, X, Y, A1, A2, Z1):
        """
        Implement the backward propagation and update the parameters accordingly. 
        
        Arguments:
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        A1 -- The tanh output of the first activation, of shape (1, number of examples)
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
        
        results:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = X.shape[1]

        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']
        
        dZ2 = A2 - Y                                #equation: dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m                 #dW2 = 1/m * dZ2 A1.T
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(self.tanh(Z1), 2))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        grads = {
            'dW1':dW1,
            'db1':db1,
            'dW2':dW2,
            'db2':db2
        }

        return grads

    def update_parameters(self, grads):
        """
        Updates parameters using the gradient descent update rule given above
        
        Arguments:
        grads -- python dictionary containing your gradients 
        """
        # Retrieve each parameter from the dictionary "parameters"
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']
        
        # Retrieve each gradient from the dictionary "grads"
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']
        
        # Update rule for each parameter
        W1 -= self.learning_rate * dW1
        b1 -= self.learning_rate * db1
        W2 -= self.learning_rate * dW2
        b2 -= self.learning_rate * db2
        
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        
        self.parameters = parameters
        
    def fit(self, X, Y):

        np.random.seed(3)

        # intialize W, b
        n_x = X.shape[0]
        n_h = self.hidden_layer_sizes
        n_y = Y.shape[0]

        self.initialize_parameters(n_x, n_h, n_y)
        self.costs = []

        for index in range(0, self.num_iterations):
            Z1, A1, Z2, A2 = self.forward_propagation(X)
            cost = self.compute_cost(Y, A2)
            grads = self.backward_propagation(X, Y, A1, A2, Z1)
            self.update_parameters(grads)

            if (index % 1000 == 0):
                self.costs.append(cost)
                print("cost at {} is: {}".format(index, cost))

    def predict(self, X):
        """
        Using the learned parameters, predicts a class for each example in X

        Arguments:
        X -- input data of size (n_x, m)

        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """
        Z1, A1, Z2, A2 = self.forward_propagation(X)
        predictions = [A2 > 0.5]

        return predictions

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y

def load_extra_datasets():  
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure

def nn_model_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = np.random.randn(1, 3)
    return X_assess, Y_assess

def initialize_parameters_test():
    n_x, n_h, n_y = initialize_parameters_test_case()

    lrDNN = LogisticRegressionDNN()
    parameters = lrDNN.initialize_parameters(n_x, n_h, n_y)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

# **Expected Output**:
# 
#     <td>**W1**</td>
#     <td> [[-0.00416758 -0.00056267]
#  [-0.02136196  0.01640271]
#  [-0.01793436 -0.00841747]
#  [ 0.00502881 -0.01245288]] </td> 

#     <td>**b1**</td>
#     <td> [[ 0.]
#  [ 0.]
#  [ 0.]
#  [ 0.]] </td> 

#     <td>**W2**</td>
#     <td> [[-0.01057952 -0.00909008  0.00551454  0.02292208]]</td> 

#     <td>**b2**</td>
#     <td> [[ 0.]] </td> 

def forward_propagation_test():
    X_assess, parameters = forward_propagation_test_case()

    lrDNN = LogisticRegressionDNN()
    Z1, A1, Z2, A2 = lrDNN.forward_propagation(X_assess, parameters)

    # Note: we use the mean here just to make sure that your output matches ours. 
    print(np.mean(Z1) ,np.mean(A1),np.mean(Z2),np.mean(A2))

# **Expected Output**:
#     -0.000499755777742 -0.000496963353232 0.000438187450959 0.500109546852 </td> 

def compute_cost_test():
    A2, Y_assess, parameters = compute_cost_test_case()

    lrDNN = LogisticRegressionDNN()
    print("cost = " + str(lrDNN.compute_cost(Y_assess, A2)))
# **Expected Output**:
#     <td> 0.692919893776 </td> 

def backward_propagation_test():
    parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

    lrDNN = LogisticRegressionDNN()
    grads = lrDNN.backward_propagation(X_assess, Y_assess, cache['A1'], cache['A2'], cache['Z1'], parameters)
    print ("dW1 = "+ str(grads["dW1"]))
    print ("db1 = "+ str(grads["db1"]))
    print ("dW2 = "+ str(grads["dW2"]))
    print ("db2 = "+ str(grads["db2"]))
# **Expected output**:
#     <td>**dW1**</td>
#     <td> [[ 0.01018708 -0.00708701]
#  [ 0.00873447 -0.0060768 ]
#  [-0.00530847  0.00369379]
#  [-0.02206365  0.01535126]] </td> 

#     <td>**db1**</td>
#     <td>  [[-0.00069728]
#  [-0.00060606]
#  [ 0.000364  ]
#  [ 0.00151207]] </td> 

#     <td>**dW2**</td>
#     <td> [[ 0.00363613  0.03153604  0.01162914 -0.01318316]] </td> 

#     <td>**db2**</td>
#     <td> [[ 0.06589489]] </td> 

def update_parameters_test():
    parameters, grads = update_parameters_test_case()

    lrDNN = LogisticRegressionDNN(learning_rate = 1.2)
    parameters = lrDNN.update_parameters(parameters, grads)

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
# **Expected Output**:
# 
#     <td>**W1**</td>
#     <td> [[-0.00643025  0.01936718]
#  [-0.02410458  0.03978052]
#  [-0.01653973 -0.02096177]
#  [ 0.01046864 -0.05990141]]</td> 

#     <td>**b1**</td>
#     <td> [[ -1.02420756e-06]
#  [  1.27373948e-05]
#  [  8.32996807e-07]
#  [ -3.20136836e-06]]</td> 

#     <td>**W2**</td>
#     <td> [[-0.01041081 -0.04463285  0.01758031  0.04747113]] </td> 

#     <td>**b2**</td>
#     <td> [[ 0.00010457]] </td> 

def fit_test():
    X_assess, Y_assess = nn_model_test_case()

    lrDNN = LogisticRegressionDNN(num_iterations=10000, learning_rate=1.2)
    parameters = lrDNN.fit(X_assess, Y_assess)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
# **Expected Output**:
# 
#     <td>**W1**</td>
#     <td> [[-4.18494056  5.33220609]
#  [-7.52989382  1.24306181]
#  [-4.1929459   5.32632331]
#  [ 7.52983719 -1.24309422]]</td> 

#     <td>**b1**</td>
#     <td> [[ 2.32926819]
#  [ 3.79458998]
#  [ 2.33002577]
#  [-3.79468846]]</td> 

#     <td>**W2**</td>
#     <td> [[-6033.83672146 -6008.12980822 -6033.10095287  6008.06637269]] </td> 

#     <td>**b2**</td>
#     <td> [[-52.66607724]] </td> 

def predict_test():

    X_assess, Y_assess = nn_model_test_case()

    lrDNN = LogisticRegressionDNN(num_iterations=10000, learning_rate=1.2)
    lrDNN.fit(X_assess, Y_assess)

    parameters, X_assess = predict_test_case()
    predictions = lrDNN.predict(X_assess)
    print("predictions mean = " + str(np.mean(predictions)))

# expected output
# predictions mean	0.666666666667

if __name__ == '__main__':
    
    X, Y = load_planar_dataset()
    # Visualize the data:
    plt.scatter(X[0,:], X[1,:], c=Y.reshape(X[0,:].shape), s=40, cmap=plt.cm.Spectral)
    plt.show()

    lrDNN = LogisticRegressionDNN(learning_rate=1.2, num_iterations=10000)

    lrDNN.fit(X, Y)

    # Plot learning curve (with costs)
    plt.plot(lrDNN.costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(lrDNN.learning_rate))
    plt.show()