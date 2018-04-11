import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

import scipy
import h5py

class LogisticRegression:
    def __init__(self, learning_rate, num_iterations):
        """
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        """
        Compute the sigmoid of z (activation function)

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        """
        return 1/(1 + np.exp(-z))
    
    def initialize_with_zeros(self, dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
        
        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)
        
        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
        """
        w = np.zeros(shape=(dim, 1))
        b = 0

        return w, b
    
    def propagate(self, w, b, X, Y):
        """
        Implement the cost function and its gradient for the propagation

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        """

        m = X.shape[1]

        #forward propagation (from X to cost)
        Z = np.dot(w.T, X) + b
        A = self.sigmoid(Z)
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

        #backward propagation (to find gradient)
        dw = 1/m * np.dot(X, (A - Y).T)
        db = 1/m * np.sum(A - Y)

        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        return cost, dw, db

    def train(self, X, Y, print_cost = False):
        """
        This function optimizes w and b by running a gradient descent algorithm
        
        Arguments:
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        print_cost -- True to print the loss every 100 steps
        
        Tips:
        You basically need to write down two steps and iterate through them:
            1) Calculate the cost and the gradient for the current parameters. Use propagate().
            2) Update the parameters using gradient descent rule for w and b.
        """
        w, b = self.initialize_with_zeros(X.shape[0])
        costs = []

        for index in range(self.num_iterations):

            # Cost and gradient calculation
            cost, dw, db = self.propagate(w, b, X, Y)

            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db

            if (index % 100 == 0):
                costs.append(cost)
                if (print_cost):
                    print('Cost on {} iteration is {}'.format(index, cost))

        self.w = w
        self.b = b

        return costs

    def predict(self, X):

        m = X.shape[1]

        Y_prediction = np.zeros((1,m))
        w_temp = self.w.reshape(X.shape[0], 1)

        Z = np.dot(w_temp.T, X) + self.b

        A = self.sigmoid(Z)

        for i in range(A.shape[1]):
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            if A[0, i] <= 0.5:
                Y_prediction[0, i] = 0
            else:
                Y_prediction[0, i] = 1
    
        assert(Y_prediction.shape == (1, m))

        return Y_prediction

    def accurate_score(self, Y_predict, Y_True):

        score = Y_predict == Y_True
        return np.average(score)

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

def load_image(filepathname):

    print("loading image \'{}\' for prediction ...".format(filepathname))
    
    image = np.array(ndimage.imread(filepathname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T

    return my_image

# main entry
if __name__ == "__main__":
    
    train_set_x, train_set_y, test_set_x, test_set_y = load_dataset()

    # standardize the data set by simply dividing each row by 255
    train_set_x = train_set_x.reshape(train_set_x.shape[0], -1).T / 255
    test_set_x = test_set_x.reshape(test_set_x.shape[0], -1).T / 255

    regressor = LogisticRegression(0.005, 2000)
    costs = regressor.train(train_set_x, train_set_y, print_cost = True)
    
    y_predict_train = regressor.predict(train_set_x)
    y_predict_Test = regressor.predict(test_set_x)
    
    print("accurate score for training set:{}".format(regressor.accurate_score(y_predict_train, train_set_y)))
    print("accurate score for testing set:{}".format(regressor.accurate_score(y_predict_Test, test_set_y)))

    # Plot learning curve (with costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(regressor.learning_rate))
    plt.show()

    # test my own image 1
    image = load_image("images\cat_in_iran.jpg")
    image_predict = regressor.predict(image)
    print("the result is {}".format(image_predict.squeeze()))

    # test my own image 2
    image = load_image("images\my_image.jpg")
    image_predict = regressor.predict(image)
    print("the result is {}".format(image_predict.squeeze()))

    # test my own image 3
    image = load_image("images\gargouille.jpg")
    image_predict = regressor.predict(image)
    print("the result is {}".format(image_predict.squeeze()))