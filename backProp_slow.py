import numpy as np
import matplotlib.pyplot as plt

def forward(X,w1,b1,w2,b2):
    Z =1/(1+np.exp(-X.dot(w1)-b1))
    A = Z.dot(w2) + b2
    expA = np.exp(A)
    Y = expA/expA.sum(axis = 1, keepdims = True)
    return Y, Z

def classification_rate(Y,P):
    n_correct =0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct)/n_total


def derivative_w2(Z,T,Y):
    N,K = T.shape
    # num of hidden units
    M = Z.shape[1]

    #Slow way of doing this
    # Direct from our derivation
    # ret1 = np.zeros((M,K))
    # for n in range(N):
    #     for m in range(M):
    #         for k in range(K):
    #             ret1[m,k] += (T[n,k]-Y[n,k])*Z[n,m]

    ret2 = np.zeros((M,K))
    for n in range(N):
        for k in range(K):
            ret2[:,k] += (T[n,k]-Y[n,k])*Z[n,:]
    # dot product is for summation and multiplication both at the same time
    # ret1 = Z.T.dot(T-Y)
    return ret2

def derivative_b2(T,Y):
    return (T-Y).sum(axis=0)

def derivative_w1(X, Z, T, Y, w2):
    # Slow way
    N,D = X.shape
    M,K = w2.shape

    ret1 = np.zeros((D,M))
    for n in range(N):
        for k in range(K):
            for m in range(M):
                for d in range(D):
                    ret1[d,m] += (T[n,k]-Y[n,k])*w2[m,k]*Z[n,m]*(1-Z[n,m])*X[n,d]

    
    # ret1 = (X.T.dot((T-Y).dot(w2.T)*Z*(1-Z)))
    return ret1


def derivative_b1(T,Y,w2,Z):
    return ((T-Y).dot(w2.T)*Z * (1-Z)).sum(axis=0)


def cost(T,Y):
    tot = T*np.log(Y)
    return tot.sum()

def main():
    '''
    Creating a data
    '''
    Nclass = 500
    D = 2 # Dimensions of input
    M = 3 # Hidden layer size
    K = 3 # Number of classes
    
    X1 = np.random.randn(Nclass,2) + np.array([0,-2])
    X2 = np.random.randn(Nclass,2) + np.array([2,2])
    X3 = np.random.randn(Nclass,2) + np.array([-2,2])
    X = np.vstack([X1,X2,X3])
    Y = np.array([0]*Nclass + [1]*Nclass+ [2]*Nclass)

    N = len(Y)

    # Turn the target into indicator variable
    # Indicator variable
    # it is like on hot encoding for the targets

    T = np.zeros((N,K))
    for i in range(N):
        T[i,Y[i]] = 1

    # See how data looks like

    plt.scatter(X[:,0],X[:,1], c=Y, s=100, alpha=0.5)
    plt.show()

    # The starting point is same, we randomly initialize the weights
    w1 = np.random.randn(D,M)
    b1 = np.random.randn(M)
    w2 = np.random.randn(M,K)
    b2 = np.random.randn(K)

    # And then we are going to do backpropogation

    learning_rate = 10e-7
    # Make an array for cost function, so that we can plot it after to see the progression
    costs = []

    # Let's set epochs
    for epoch in range(10000):
        output, hidden = forward(X, w1, b1, w2, b2)
        # For every 100 epochs calculate the cost
        if epoch%100==0:
            c = cost(T, output)
            # get the predictions
            p = np.argmax(output, axis = 1)
            # We'll use these to calculate classificastion rate
            r = classification_rate(Y, p)
            print('cost:{}, classification_rate:{}'.format(c,r))
            costs.append(c)

        #Now we'll do gradient ascent
        w2 += learning_rate * derivative_w2(hidden, T, output)
        b2 += learning_rate * derivative_b2(T, output)
        w1 += learning_rate * derivative_w1(X,hidden,T,output,w2)
        b1 += learning_rate * derivative_b1(T, output, w2, hidden)

    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    main()