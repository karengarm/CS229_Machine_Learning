import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
       
        self.theta = np.linalg.solve(X.T.dot(X), X.T.dot(y)) 
            
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        
        for i in range(2, k+1):
            x_p = np.power(X[:,1], i)
            x_p = np.expand_dims( x_p , axis = 1)
            X = np.hstack([X, x_p])
        return X
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        sin_x=np.sin(X[:,1])
        sin_x=np.expand_dims(sin_x, axis=1)
        for i in range(2, k+1):
            x_p = np.power(X[:,1], i)
            x_p = np.expand_dims( x_p , axis = 1)
            X = np.hstack([X, x_p])
        X=np.hstack([X,sin_x])
        return X

        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        y_pred = X.dot(self.theta)
        return y_pred
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        linearmod = LinearModel()
        if sine == False:
            X = linearmod.create_poly(k, train_x)
            plot_pi = linearmod.create_poly(k, plot_x)
        else:
             X = linearmod.create_sin(k, train_x)
             plot_pi = linearmod.create_sin(k, plot_x)
        linearmod.fit(X,train_y)
        plot_y = linearmod.predict(plot_pi)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all experiments
    '''
    # *** START CODE HERE ***
    run_exp(train_path, False, [3], 'poly3.png')
    run_exp(train_path, False, [3, 5, 10, 20], 'polyk.png')
    run_exp(train_path, True, [0, 1, 2, 3, 5, 10, 20], 'sink.png')
    run_exp(small_path, True, [1, 2, 5, 10, 20], 'small_sinek.png')
    run_exp(small_path, False, [1, 2, 5, 10, 20], 'small_polyk.png')
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
