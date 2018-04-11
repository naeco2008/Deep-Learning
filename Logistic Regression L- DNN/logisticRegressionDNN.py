import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import h5py

class LogisticRegressionDNN:
    """
    define the logistic Regression DNN with L hidden layer
    """
    def __init__(self, learning_rate=0.02, num_iterations=2000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, Z):
        """
        Implements the sigmoid activation in numpy
        
        Arguments:
        Z -- numpy array of any shape
        
        Returns:
        A -- output of sigmoid(z), same shape as Z
        cache -- returns Z as well, useful during backpropagation
        """
        
        A = 1/(1+np.exp(-Z))
        cache = Z
        
        return A, cache

    def relu(self, Z):
        """
        Implement the RELU function.

        Arguments:
        Z -- Output of the linear layer, of any shape

        Returns:
        A -- Post-activation parameter, of the same shape as Z
        cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
        """
        
        A = np.maximum(0,Z)
        
        assert(A.shape == Z.shape)
        
        cache = Z 
        return A, cache

    def relu_backward(self, dA, cache):
        """
        Implement the backward propagation for a single RELU unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        
        Z = cache
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        
        assert (dZ.shape == Z.shape)
        
        return dZ

    def sigmoid_backward(self, dA, cache):
        """
        Implement the backward propagation for a single SIGMOID unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        
        Z = cache
        
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        
        assert (dZ.shape == Z.shape)
        
        return dZ

    def initialize_parameters_deep(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        np.random.seed(3)
        parameters = {}
        layer_size = len(layer_dims)  # number of layers in the network

        for layer in range(1, layer_size):
            parameters['W' + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) * 0.01
            parameters['b' + str(layer)] = np.zeros(shape=(layer_dims[layer], 1))
        
            assert(parameters['W' + str(layer)].shape == (layer_dims[layer], layer_dims[layer-1]))
            assert(parameters['b' + str(layer)].shape == (layer_dims[layer], 1))

        self.parameters = parameters
    
    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
        Z = np.dot(W, A) + b
        
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        
        return Z, cache
    
    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                stored for computing the backward pass efficiently
        """
        
        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
            ### END CODE HERE ###
        
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
            ### END CODE HERE ###
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache
        
    def forward_propagation(self, X):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        
        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        caches = []
        A = X
        L = len(self.parameters) // 2                  # number of layers in the neural network
        
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            ### START CODE HERE ### (≈ 2 lines of code)
            A, cache = self.linear_activation_forward(A_prev, self.parameters['W'+str(l)], self.parameters['b'+str(l)], 'relu')
            caches.append(cache)
            ### END CODE HERE ###
        
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        ### START CODE HERE ### (≈ 2 lines of code)
        AL, cache = self.linear_activation_forward(A, self.parameters['W'+str(L)], self.parameters['b'+str(L)], 'sigmoid')
        caches.append(cache)
        ### END CODE HERE ###
        
        assert(AL.shape == (1,X.shape[1]))
                
        return AL, caches

    def compute_cost(self, Y, AL):
        """
        Computes the cross-entropy cost given in equation (13)
        
        Arguments:
        AL -- The sigmoid output of the last activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        parameters -- python dictionary containing your parameters W1, b1, W2 and b2
        
        Returns:
        cost -- cross-entropy cost given equation (13)
        """
        m = Y.shape[1]

        cost = -1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL), axis = 1, keepdims = True)

        cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                    # E.g., turns [[17]] into 17 
        cost = np.float(cost)
        
        assert(isinstance(cost, float))
        
        return cost
    
    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]
        
        ### START CODE HERE ### (≈ 3 lines of code)
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis = 1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        ### END CODE HERE ###
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            ### END CODE HERE ###
            
        elif activation == "sigmoid":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            ### END CODE HERE ###
        
        return dA_prev, dW, db

    def backward_propagation(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        
        Returns:
        grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ...
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        ### START CODE HERE ### (1 line of code)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        ### END CODE HERE ###
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        ### START CODE HERE ### (approx. 2 lines)
        current_cache = caches[L - 1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, 'sigmoid')
        ### END CODE HERE ###
        
        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            ### START CODE HERE ### (approx. 5 lines)
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 2)], current_cache, 'relu')
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            ### END CODE HERE ###

        return grads

    def update_parameters(self, grads):
        """
        Updates parameters using the gradient descent update rule given above
        
        Arguments:
        grads -- python dictionary containing your gradients 
        """

        L = len(self.parameters) // 2 # number of layers in the neural network

        for layer in range(L):
            self.parameters['W'+str(layer + 1)] -= grads['dW'+str(layer + 1)] * self.learning_rate
            self.parameters['b'+str(layer + 1)] -= grads['db'+str(layer + 1)] * self.learning_rate
        
    def fit(self, X, Y, layers_dims, print_cost=False):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        """
        np.random.seed(3)

        # intialize W, b
        self.initialize_parameters_deep(layers_dims)
        self.costs = []

        for index in range(0, self.num_iterations):
            AL, caches = self.forward_propagation(X)
            cost = self.compute_cost(Y, AL)
            grads = self.backward_propagation(AL, Y, caches)
            self.update_parameters(grads)

            if (index % 100 == 0):
                self.costs.append(cost)
                print("cost at {} is: {}".format(index, cost))

        # plot the cost
        if print_cost:
            plt.plot(np.squeeze(self.costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(self.learning_rate))
            plt.show()

    def predict(self, X, y):
        """
        Using the learned parameters, predicts a class for each example in X

        Arguments:
        X -- input data of size (n_x, m)

        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """
        m = X.shape[1]
        A, caches = self.forward_propagation(X)
        predictions = np.zeros((1,m))

        # convert probas to 0/1 predictions
        for i in range(0, A.shape[1]):
            if A[0,i] > 0.5:
                predictions[0,i] = 1
            else:
                predictions[0,i] = 0
        
        #print results
        print("Accuracy: "  + str(np.sum((predictions == y)/m)))
        
        return predictions

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

if __name__ == '__main__':
    
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
    #layers_dims = [12288, 7, 1] #  2-layer model
    lrDNN = LogisticRegressionDNN(learning_rate=0.0075, num_iterations = 25000)
    lrDNN.fit(train_x, train_y, layers_dims, print_cost=True)

    pred_test = lrDNN.predict(test_x, test_y)