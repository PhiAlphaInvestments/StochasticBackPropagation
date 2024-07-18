import numpy as np
import keras
from keras.utils import to_categorical



class NeuralNetworkStoBackprop:
    def __init__(self, Input_Dim_Col, Hidden_Neurons, Out_Dim, tolerance_, Learning_Rate, Beta_1, Beta_2,
                 Max_iterations, Verbose_,Neuroplastic, A_, B_):
        self.InputHyperParam = Input_Dim_Col
        self.OutputHyperParam = Out_Dim
        self.HiddenLayerParam = Hidden_Neurons
        self.Beta1 = Beta_1
        self.Beta2 = Beta_2
        self.Learning_Rate = Learning_Rate
        self.W_1 = np.random.randn(self.InputHyperParam, self.HiddenLayerParam) * .01
        self.W_2 = np.random.randn(self.HiddenLayerParam, self.OutputHyperParam) * .01
        self.alpha = tolerance_
        self.Max_iters = Max_iterations
        self.Verbose = Verbose_
        self.A = A_
        self.B = B_
        self.Derivative1 = np.zeros((self.InputHyperParam, self.HiddenLayerParam))
        self.Derivative2 = np.zeros((self.HiddenLayerParam, self.OutputHyperParam))
        self.dSx = 1
        self.Neuroplastic = Neuroplastic
        self.Lcost =[]
    def NSigmoid(self, x):
        c = np.e
        if self.Neuroplastic==True:
           c =  np.random.uniform(self.A,self.B)
        return 1 / (1 + pow(c, -x))

    def NSigmoid_Prime(self, x):
        c = np.e
        if self.Neuroplastic == True:
            c = np.random.uniform(self.A, self.B)
        return (pow(c, (-x)) * np.log(c)) / ((1 + pow(c, (-x))) ** 2)

    def NSigmoid_Double_Prime(self, x):
        c = np.e
        if self.Neuroplastic == True:
            c = np.random.uniform(self.A, self.B)
        return -(pow(c, x) * (-1 + pow(c, x)) * np.log(c) ** 2) / (
                    1 + pow(c, x)) ** 3
    def Plot_Cost(self):
        import matplotlib.pyplot as plt
        plt.plot(self.Lcost)
        plt.show()

    def FeedForward(self, Input_X):
        self.M_X2   = np.dot(Input_X, self.W_1)
        self.M_2 = self.NSigmoid(self.M_X2  )
        self.M_X3 = np.dot(self.M_2, self.W_2)
        y = self.NSigmoid(self.M_X3)
        return y

    def Cost(self, Input_X, y_Correct):
        yNew = self.FeedForward(Input_X)
        temp = (y_Correct -yNew )
        temp = temp * temp
        dSx = np.average(temp)/float(2)
        self.dSx = dSx
        return dSx

    def Compute_Dx(self, Input_X, Correct_y):


        yNew = self.FeedForward(Input_X)
        error =  (yNew-Correct_y)

        dM_3 = self.NSigmoid_Prime(self.M_X3) * self.dSx + (self.NSigmoid_Double_Prime(self.M_X3) / 2) * (
            self.dSx) ** 2
        d3 = error * dM_3
        self.Derivative2 = np.dot(self.M_2.T, d3)

        dM_2 = self.NSigmoid_Prime(self.M_X2  ) * self.dSx + (self.NSigmoid_Double_Prime(self.M_X2  ) / 2) * (
            self.dSx) ** 2
        d2 = np.dot(d3, self.W_2.T) * dM_2
        self.Derivative1 = np.dot(Input_X.T, d2)

    def Train(self, Input_X, Correct_y):

        Train_condition = True
        iters_ = 0

        self.Compute_Dx(Input_X, Correct_y)
        dSx = 0
        mt_1 = np.zeros((self.InputHyperParam, self.HiddenLayerParam))
        mt_2 = np.zeros((self.HiddenLayerParam, self.OutputHyperParam))

        while Train_condition and iters_ < self.Max_iters:

            self.Compute_Dx(Input_X, Correct_y)
            dSx = self.Cost(Input_X, Correct_y)
            self.Lcost.append(dSx)
            if self.Verbose == True:
                print(iters_, " <<<< Iters")
                print(dSx, " <<<< cost_Value")
            if J < self.alpha:
                Train_condition = False

            mt_1 = self.Beta1 * mt_1 + (1 - self.Beta1) * self.Derivative1
            mt_2 = self.Beta1 * mt_2 + (1 - self.Beta1) * self.Derivative2
            self.W_1 = self.W_1 - self.Learning_Rate * (self.Beta2 * mt_1)
            self.W_2 = self.W_2 - self.Learning_Rate * (self.Beta2 * mt_2)

            iters_ = iters_ + 1
        if self.Verbose == True:
            print(iters_, " <<<< Iters")
            print(dSx, " <<<< cost_Value")

    def Predict(self, Input_X):
        M_X2 = np.dot(Input_X, self.W_1)
        M_2 = self.NSigmoid(M_X2)
        M_X3 = np.dot(M_2, self.W_2)
        y = self.NSigmoid(M_X3)
        return y


A = np.e - .1
B = np.e + .1


def NSigmoid(x):
    ##c = np.random.uniform(A,B)
    c = np.e
    return 1 / (1 + pow(c, -x))


num_classes = 10
input_shape = (28, 28, 1)


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path='mnist.npz')
a = .5
for i in range(60000):
    if i % 2 == 0:
        x_train[i] = abs(np.random.normal(0, 1, (28, 28)))
        y_train[i] = 1
    else:
        x_train[i] = .5 * (abs(np.random.normal(0, np.sqrt(1 - a), (28, 28))) + abs(
            np.random.normal(0, np.sqrt(1 + a), (28, 28))))
        y_train[i] = 0
for i in range(10000):
    if i % 2 == 0:
        x_test[i] = abs(np.random.normal(0, 1, (28, 28)))
        y_test[i] = 1
    else:
        x_test[i] = .5 * (abs(np.random.normal(0, np.sqrt(1 - a), (28, 28))) + abs(
            np.random.normal(0, np.sqrt(1 + a), (28, 28))))
        y_test[i] = 0



x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print("y_test shape:", x_test.shape)


y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)
print(x_test.shape)
print(x_train.shape)
x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


def run_test(x_train,y_train,x_test,y_test):

    NNN = NeuralNetworkStoBackprop(784, 128, 2, .01, .001, .9, .999, 300, True,False, A, B)

    _epoch = 5
    print(y_train.shape)
    x_train = x_train
    print(NNN.FeedForward(x_train))
    for i in range(_epoch):
        x__ = x_train
        y__ = y_train

        print(i / float(_epoch))


        NNN.Train(x__, y__)
        NNN.Plot_Cost()
        print("----")
        print(np.average(NNN.Predict(x__) - y__))
    mse = 0
    acc = []
    _epoch = 1
    for i in range(_epoch):
        x__ = x_test
        y__ = y_test
        acc = []
        pred_ = NNN.Predict(x__)
        for _j in range(len(pred_)):
            if y__[_j].tolist().index(y__[_j].max()) == pred_[_j].tolist().index(pred_[_j].max()):
                acc.append(1)
            else:
                acc.append(0)
        print(sum(acc) / len(acc))


if __name__ == '__main__':
    run_test(x_train,y_train,x_test,y_test)
