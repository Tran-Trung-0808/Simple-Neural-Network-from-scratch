import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_csv(r'C:\Users\Trung\OneDrive\Code\data\train.csv\train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) 

data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape


def init_params() :
    W_1 = np.random.rand(10, 784) - 0.5
    b_1 = np.random.rand(10, 1) - 0.5
    W_2 = np.random.rand(10, 10) - 0.5
    b_2 = np.random.rand(10, 1) - 0.5
    return W_1, b_1, W_2, b_2 

def ReLU(Z) :
    return np.maximum(0, Z)

def softmax(Z) :
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_propagation(X, W_1, b_1, W_2, b_2) :
    # Input Layer 
    Z_1 = W_1.dot(X) + b_1
    A_1 = ReLU(Z_1) 
    # Hidden Layer
    Z_2 = W_2.dot(A_1) + b_2
    A_2 = softmax(Z_2)
    return Z_1, A_1, Z_2, A_2

def one_hot(Y) :
    # one-hot encoding for datas
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def ReLU_deriv(Z) :
    return Z > 0

def backward_propagation(X, Y, A_1, Z_1, W_1, A_2, Z_2, W_2) :
    # Calculating derivatives
    one_hot_Y = one_hot(Y)
    dZ_2 = A_2 - one_hot_Y
    dW_2 = (1 / m) * dZ_2.dot(A_1.T)
    db_2 = (1 / m) * np.sum(dZ_2)
    dZ_1 = (W_2.T).dot(dZ_2) * ReLU_deriv(Z_2)
    dW_1 = (1 / m) * (dZ_1.dot(X.T))
    db_1 = (1 / m) * np.sum(dZ_1)
    return dZ_1, dZ_2, dW_1, dW_2, db_1, db_2

def update_params(W_1, b_1, W_2, b_2, dW_1, db_1, dW_2, db_2, alpha) :
    W_1 = W_1 - alpha * dW_1
    b_1 = b_1 - alpha * db_1
    W_2 = W_2 - alpha * dW_2
    b_2 = b_2 - alpha * db_2
    return W_1, b_1, W_2, b_2

def get_predictions(A_2) :
    return np.argmax(A_2, 0)

def get_accuracy(predictions, Y) :
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y,iteration, alpha) :
    # Main flow of the neural network
    # Randomly initialize weights/paramaters
    W_1, b_1, W_2, b_2 = init_params()

    for i in range(iteration) :
        # Going through layers - Forward propagation
        Z_1, A_1, Z_2, A_2 = forward_propagation(X, W_1, b_1, W_2, b_2)
        # Get back from the end and calculating gradients - Backward propagation
        dZ_1, dZ_2, dW_1, dW_2, db_1, db_2 = backward_propagation(X, Y, A_1, Z_1, W_1, A_2, Z_2, W_2)
        # Fitting weights
        W_1, b_1, W_2, b_2 = update_params(W_1, b_1, W_2, b_2, dW_1, db_1, dW_2, db_2, alpha)
        if i % 50 == 0 :
            print(f'Iteration {i} : ')
            predictions = get_predictions(A_2)
            print(get_accuracy(predictions, Y))
    return W_1, b_1, W_2, b_2

def make_predictions(X, W_1, b_1, W_2, b_2) :
    _, _, _, A_2 = forward_propagation(X, W_1, b_1, W_2, b_2)
    return get_predictions(A_2)

def test_prediction(index, W_1, b_1, W_2, b_2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W_1, b_1, W_2, b_2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def main() :
    W_1, b_1, W_2, b_2 = gradient_descent(X_train, Y_train, 500, 0.10)
    final_predictions = make_predictions(X_test, W_1, b_1, W_2, b_2)
    print(get_accuracy(final_predictions, Y_test))
main()

