import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import h5py
import math

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
        """
        A = 1/(1+np.exp(-Z))
        
        return A

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
        
        return A

    def relu_backward(self, dA, Z):
        """
        Implement the backward propagation for a single RELU unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        
        assert (dZ.shape == Z.shape)
        
        return dZ

    def sigmoid_backward(self, dA, Z):
        """
        Implement the backward propagation for a single SIGMOID unit.

        Arguments:
        dA -- post-activation gradient, of any shape

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        
        assert (dZ.shape == Z.shape)
        
        return dZ

    def initialize_parameters(self, layer_dims):
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
            #parameters['W' + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) * 0.01
            parameters['W' + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) / np.sqrt(layer_dims[layer - 1]) #He method to initialize parameters, it is better than 0.01.
            parameters['b' + str(layer)] = np.zeros(shape=(layer_dims[layer], 1))
        
            assert(parameters['W' + str(layer)].shape == (layer_dims[layer], layer_dims[layer-1]))
            assert(parameters['b' + str(layer)].shape == (layer_dims[layer], 1))

        self.parameters = parameters
    
    def initialize_adam(self):
        """
        Initializes v and s as two python dictionaries with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        
        Returns: 
        v -- python dictionary that will contain the exponentially weighted average of the gradient.
                        v["dW" + str(l)] = ...
                        v["db" + str(l)] = ...
        s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                        s["dW" + str(l)] = ...
                        s["db" + str(l)] = ...

        """
        L = len(self.parameters) // 2 # number of layers in the neural networks
        v = {}
        s = {}
        
        # Initialize v, s. Input: "parameters". Outputs: "v, s".
        for l in range(L):
            v["dW" + str(l+1)] = np.zeros(self.parameters["W"+str(l+1)].shape)
            v["db" + str(l+1)] = np.zeros(self.parameters["b"+str(l+1)].shape)
            s["dW" + str(l+1)] = np.zeros(self.parameters["W"+str(l+1)].shape)
            s["db" + str(l+1)] = np.zeros(self.parameters["b"+str(l+1)].shape)
        
        return v, s

    def forward_propagation(self, X, dropout=False, keep_prob=0.5):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        dropout -- indicate if use dropout technology to avoid overfitting. This should be False for predict. This is only used for training set.
        keep_prob - probability of keeping a neuron active during drop-out, scalar

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        layer_size = len(self.parameters) // 2                  # number of layers in the neural network

        cache_A = {}
        cache_A[0] = X

        cache_Z = {}
        cache_D = {}

        np.random.seed(1)

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for layer in range(1, layer_size + 1):
            W, b = self.parameters['W'+str(layer)], self.parameters['b'+str(layer)]
            cache_Z[layer] = np.dot(W, cache_A[layer - 1]) + b

            if (layer == layer_size):
                cache_A[layer] = self.sigmoid(cache_Z[layer])
            else:
                cache_A[layer] = self.relu(cache_Z[layer])

            if dropout == True and layer != layer_size:  ## use dropout to avoid overfitting
                cache_D[layer] = np.random.binomial(n=1,p=keep_prob,size=cache_A[layer].shape)
                cache_A[layer] = cache_A[layer] * cache_D[layer]
                cache_A[layer] = cache_A[layer] / keep_prob

        AL = cache_A[layer_size]
        assert(AL.shape == (1,X.shape[1]))
                
        return AL, cache_D, cache_A, cache_Z

    def compute_cost(self, X, Y, AL, regularization=True, lambd = 0.7):
        """
        Computes the cross-entropy cost given in equation (13)
        
        Arguments:
        AL -- The sigmoid output of the last activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        regularization -- indicate if L2-regularization is used to avoid overfitting. The default value is True
        lambd -- the parameter for L2-regularization algorithm
        
        Returns:
        cost -- cross-entropy cost given equation (13)
        """
        m = X.shape[1]

        parameter_num = int(len(self.parameters)/2)

        cross_entropy_cost = -1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL), axis = 1, keepdims = True)
        cross_entropy_cost = np.squeeze(cross_entropy_cost)     # makes sure cost is the dimension we expect. E.g., turns [[17]] into 17 
        cross_entropy_cost = np.float(cross_entropy_cost)
        
        L2_regularization_cost = 0
        if regularization == True:
            for item in range(0, parameter_num):
                W_item = self.parameters["W"+str(item+1)]
                L2_regularization_cost += np.sum(np.square(W_item))
            L2_regularization_cost = L2_regularization_cost * lambd / (2 * m)
        
        cost = cross_entropy_cost + L2_regularization_cost
        assert(isinstance(cost, float))
        
        return cost
    
    def backward_propagation(self, AL, Y, cache_D, cache_A, cache_Z, regularization=True, lambd=0.7, dropout=False, keep_prob=0.5):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        regularization -- indicate if L2-regularization technology will be used to avoid overfitting
        lambd -- the regularization parameter
        dropout -- indicate if dropout technology will be used to avoid overfitting
        keep_prob -- threashold for dropout
        
        Returns:
        grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ...
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ...
        """
        grads = {}
        layer_size = len(self.parameters) // 2        # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)             # after this line, Y is the same shape as AL

        assert(AL.shape == cache_A[layer_size].shape)

        # Initializing the backpropagation
        dAL = - Y / AL + (1 - Y) / (1 - AL)
        grads["dA" + str(layer_size)] = dAL

        for l in range(0, layer_size):
            layer = layer_size - l

            W = self.parameters["W"+str(layer)]
            Z = cache_Z[layer]
            dA = grads["dA" + str(layer)]

            if (layer == layer_size):
                dZ = self.sigmoid_backward(dA, Z)
            else:
                dZ = self.relu_backward(dA, Z)

            grads["dA" + str(layer - 1)] = np.dot(W.T, dZ)
            if (dropout == True) and (layer != 1):
                D = cache_D[layer - 1]
                grads["dA" + str(layer - 1)] *= D
                grads["dA" + str(layer - 1)] /= keep_prob

            if regularization:
                grads["dW" + str(layer)] = 1 / m * np.dot(dZ, cache_A[layer - 1].T) + W * lambd / m
            else:
                grads["dW" + str(layer)] = 1 / m * np.dot(dZ, cache_A[layer - 1].T)

            grads["db" + str(layer)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        
        return grads

    def update_parameters(self, grads, optimizer="gd", v = None, s = None, t = None, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
        """
        Updates parameters using the gradient descent update rule given above
        
        Arguments:
        grads -- python dictionary containing your gradients 
        optimizer -- indicate what optimizer we will use to update parameter. it contains: "gd" -- normal gradient decent. "adam" -- adam technology
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        beta1 -- Exponential decay hyperparameter for the first moment estimates 
        beta2 -- Exponential decay hyperparameter for the second moment estimates 
        epsilon -- hyperparameter preventing division by zero in Adam updates

        Returns:
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        """

        L = len(self.parameters) // 2 # number of layers in the neural network
        v_corrected = {}                         # Initializing first moment estimate, python dictionary
        s_corrected = {}                         # Initializing second moment estimate, python dictionary

        for layer in range(L):
            if optimizer == "gd":
                self.parameters['W'+str(layer + 1)] -= grads['dW'+str(layer + 1)] * self.learning_rate
                self.parameters['b'+str(layer + 1)] -= grads['db'+str(layer + 1)] * self.learning_rate
            elif optimizer == "adam":
                # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
                v["dW" + str(layer+1)] = beta1 * v["dW" + str(layer+1)] + (1 - beta1) * grads["dW" + str(layer+1)]
                v["db" + str(layer+1)] = beta1 * v["db" + str(layer+1)] + (1 - beta1) * grads["db" + str(layer+1)]

                # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
                v_corrected["dW" + str(layer+1)] = v["dW" + str(layer+1)] / (1 - beta1 ** t)
                v_corrected["db" + str(layer+1)] = v["db" + str(layer+1)] / (1 - beta1 ** t)

                # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
                s["dW" + str(layer+1)] = beta2 * s["dW" + str(layer+1)] + (1 - beta2) * np.square(grads["dW" + str(layer+1)])
                s["db" + str(layer+1)] = beta2 * s["db" + str(layer+1)] + (1 - beta2) * np.square(grads["db" + str(layer+1)])

                # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
                s_corrected["dW" + str(layer+1)] = s["dW" + str(layer+1)] / (1 - beta2 ** t)
                s_corrected["db" + str(layer+1)] = s["db" + str(layer+1)] / (1 - beta2 ** t)

                # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
                self.parameters["W" + str(layer+1)] = self.parameters["W" + str(layer+1)] - self.learning_rate * v_corrected["dW" + str(layer+1)] / (epsilon + np.sqrt(s_corrected["dW" + str(layer+1)])) 
                self.parameters["b" + str(layer+1)] = self.parameters["b" + str(layer+1)] - self.learning_rate * v_corrected["db" + str(layer+1)] / (epsilon + np.sqrt(s_corrected["db" + str(layer+1)])) 

        return v, s
        
    def gradient_check(self, X, Y, layers_dims, epsilon = 1e-7):
        """
        Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

        Arguments:
        x -- input datapoint, of shape (input size, 1)
        y -- true "label"
        epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

        Returns:
        difference -- difference (2) between the approximated gradient and the backward propagation gradient
        """
        np.random.seed(3)

        # intialize W, b
        self.initialize_parameters(layers_dims)

        # get gradients
        AL, cache_D, cached_A, cached_Z = self.forward_propagation(X)
        cost = self.compute_cost(X, Y, AL, regularization=False)
        grads = self.backward_propagation(AL, Y, cache_D, cached_A, cached_Z, regularization=False)

        # Set-up variables
        parameters_values, _ = self.dictionary_to_vector(self.parameters)
        grads_values, _ = self.gradient_to_vector(grads)

        num_parameters = parameters_values.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))

        for i in range(0, num_parameters):
            # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
            # "_" is used because the function you have to outputs two parameters but we only care about the first one
            thetaplus = np.copy(parameters_values)                
            thetaplus[i][0] = thetaplus[i][0] + epsilon
            self.override_parameters(self.vector_to_dictionary(thetaplus, layers_dims))

            AL, _, _, _ = self.forward_propagation(X)
            J_plus[i] = self.compute_cost(X, Y, AL)
            
            # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
            thetaminus = np.copy(parameters_values)                          
            thetaminus[i][0] = thetaminus[i][0] - epsilon                 
            self.override_parameters(self.vector_to_dictionary(thetaminus, layers_dims))

            AL, _, _, _ = self.forward_propagation(X)
            J_minus[i] = self.compute_cost(X, Y, AL)
            
            # Compute gradapprox[i]
            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
        
        # Compare gradapprox to backward propagation gradients by computing difference.
        numerator = np.linalg.norm(grads_values - gradapprox)                              
        denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grads_values)            
        difference = numerator / denominator         

        if difference > 1e-7:
            print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
        
        return difference

    # GRADED FUNCTION: random_mini_batches

    def random_mini_batches(self, X, Y, mini_batch_size = 64, seed = 0):
        """
        Creates a list of random minibatches from (X, Y)
        
        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer
        
        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """
        
        np.random.seed(seed)            # To make your "random" minibatches the same as ours
        m = X.shape[1]                  # number of training examples
        mini_batches = []
            
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1,m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches

    def fit(self, X, Y, layers_dims, print_cost=False, dropout=False, keep_prob=0.5, optimizer="gd"):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        """
        np.random.seed(3)

        # intialize W, b
        self.initialize_parameters(layers_dims)
        v, s = self.initialize_adam()
        self.costs = []

        mini_batch_size = 64
        seed = 10
        t = 0

        for index in range(0, self.num_iterations):

            # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
            seed = seed + 1
            minibatches = self.random_mini_batches(X, Y, mini_batch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # forward propagation
                AL, cache_D, cached_A, cached_Z = self.forward_propagation(minibatch_X, dropout=dropout, keep_prob=keep_prob)
                cost = self.compute_cost(minibatch_X, minibatch_Y, AL)
                grads = self.backward_propagation(AL, minibatch_Y, cache_D, cached_A, cached_Z, regularization=True, dropout=dropout, keep_prob=keep_prob)

                t = t + 1 # Adam counter
                v, s = self.update_parameters(grads, optimizer, v, s, t)

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
        A, _, _, _ = self.forward_propagation(X, dropout=False)
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

    def dictionary_to_vector(self, parameters):
        keys = []
        count = 0
        for key in parameters.keys():
            
            # flatten parameter
            new_vector = np.reshape(parameters[key], (-1,1))
            keys = keys + [key]*new_vector.shape[0]
            
            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1

        return theta, keys

    def gradient_to_vector(self, gradients):
        keys = []
        count = 0
        for key in gradients.keys():
            
            if ((key.find("dW") == -1) and (key.find("db") == -1)):
                continue

            # flatten 
            new_vector = np.reshape(gradients[key], (-1,1))
            keys = keys + [key]*new_vector.shape[0]
            
            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1

        return theta, keys

    def vector_to_dictionary(self, theta, layer_dims):
        parameters = {}
        layer_size = len(layer_dims)
        start = 0
        for layer in range(1, layer_size):
            size = layer_dims[layer] * layer_dims[layer - 1]
            parameters['W'+str(layer)] = np.reshape(theta[start:(start+size)], (layer_dims[layer], layer_dims[layer - 1]))
            start += size
            parameters['b'+str(layer)] = np.reshape(theta[start:(start + layer_dims[layer])], (layer_dims[layer], 1))
            start += layer_dims[layer]

        return parameters

    def override_parameters(self, parameters):
        for key in self.parameters.keys():
            self.parameters[key] = parameters[key]

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

    #layers_dims = [12288, 7, 1] #  2-layer model
    lrDNN = LogisticRegressionDNN(learning_rate=0.0007, num_iterations = 10000) #0.0075

    # gradient check
    # np.random.seed(1)
    # layers_dims = [4, 5, 3, 1] #  5-layer model
    # x = np.random.randn(4,3)
    # y = np.array([1, 1, 0])
    # lrDNN.gradient_check(x, y, layers_dims)

    layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
    lrDNN.fit(train_x, train_y, layers_dims, print_cost=True, dropout=False, optimizer="adam")

    pred_train = lrDNN.predict(train_x, train_y)
    pred_test = lrDNN.predict(test_x, test_y)
