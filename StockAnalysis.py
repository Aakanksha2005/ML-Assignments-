#Question 3:Regression
# %%
import Oracle_Assignment_2
from Oracle_Assignment_2 import q3_linear_1, q3_linear_2
import numpy as np

data_3_1 = q3_linear_1(23647)
data_3_2 = q3_linear_2(23647)

print(data_3_1)
print("Hello")
print(data_3_2)

N_1 = len(data_3_1[0])
N_2 = len(data_3_2[0])

print(N_1)
print(N_2)

# %%
import numpy as np


# OLS
X_1 = np.array(data_3_1[0])
y_1 = np.array(data_3_1[1])
X_2 = np.array(data_3_2[0])
y_2 = np.array(data_3_2[1])

w1_ols = np.linalg.inv(X_1.T @ X_1) @ X_1.T @ y_1
print("Weights for data_3_1: ", w1_ols)

w2_ols = np.linalg.inv(X_2.T @ X_2) @ X_2.T @ y_2
print("Weights for data_3_2: ", w2_ols)

# Ridge Regression
lamda = 1 
w1_rr = np.linalg.inv(X_1.T @ X_1 + lamda * np.eye(X_1.shape[1])) @ X_1.T @ y_1

w2_rr = np.linalg.inv(X_2.T @ X_2 + lamda * np.eye(X_2.shape[1])) @ X_2.T @ y_2

print("Weights for data_3_1 using Ridge Regression: ", w1_rr)
print("Weights for data_3_2 using Ridge Regression: ", w2_rr)

# MSE
def mse(y, y_pred):
    return np.mean((y - y_pred)**2)

y_pred_1 = X_1 @ w1_ols
y_pred_2 = X_2 @ w2_ols

mse_1_ols = mse(y_1, y_pred_1)
print("MSE for data_3_1: ", mse_1_ols)

mse_2_ols = mse(y_2, y_pred_2)
print("MSE for data_3_2: ", mse_2_ols)

np.savetxt("w_ols_23647.csv" , w2_ols, delimiter = ",")
np.savetxt("w_rr_23647.csv" , w2_rr, delimiter = ",")

y_pred_1_rr = X_1 @ w1_rr
y_pred_2_rr = X_2 @ w2_rr

mse_1_rr = mse(y_1, y_pred_1_rr)
print("MSE for data_3_1 using Ridge Regression: ", mse_1_rr)

mse_2_rr = mse(y_2, y_pred_2_rr)
print("MSE for data_3_2 using Ridge Regression: ", mse_2_rr)

# %%
# Support Vector Regression

from Oracle_Assignment_2 import q3_stocknet

data3_3 = q3_stocknet(23647)
print(data3_3)


# %%
import pandas as pd

df = pd.read_csv("UNH.csv")
close_prices = df["Close"].values
print(close_prices)
print(len(close_prices))

# %%
import sklearn.preprocessing as skp

sscaler = skp.StandardScaler()

close_prices_norm = sscaler.fit_transform(close_prices.reshape(-1, 1))
print(close_prices_norm)
print(len(close_prices_norm))

# %%
def createXY(t):
    N = len(close_prices_norm)
    X = np.zeros((N - t, t))
    for i in range(N - t):
        X[i] = close_prices_norm[i:i+t].T
    y = close_prices_norm[t:]
    return X, y

def create_trains_tests(X,y):
    N= len(X)
    X_train_3 = X[:int(0.5*N)]
    Y_train_3 = y[:int(0.5*N)]
    X_test_3 = X[int(0.5*N):]
    Y_test_3 = y[int(0.5*N):]
    
    return X_train_3, Y_train_3, X_test_3, Y_test_3





# %%
import cvxopt
def stock_dual(epsilon , X , y , K, C=1):
    N = len(X)
    P = np.zeros((2*N, 2*N))
    for i in range(N):
        for j in range(N):
            P[i,j] = K(X[i], X[j])
            P[i+N, j] = -K(X[i], X[j])
            P[i, j+N] = -K(X[i], X[j])
            P[i+N, j+N] = K(X[i], X[j])
    P = cvxopt.matrix(P)
    
    kernel_mat = [K(X[i], X[j]) for i in range(N) for j in range(N)]
    kernel_mat = np.array(kernel_mat).reshape(N, N)
    q = np.zeros(2*N)
    for i in range(N):
        q[i] = -y[i] + epsilon
        q[i+N] = y[i] + epsilon
    q = cvxopt.matrix(q)
    
    G = np.vstack((np.eye(2*N), -np.eye(2*N)))
    G = cvxopt.matrix(G)
    
    h = np.hstack([np.zeros(2 * N), C * np.ones(2 * N)])
    h = cvxopt.matrix(h)
    
    A=np.zeros((1,2*N))
    A[0,:N] = 1
    A[0,N:] = -1
    A = cvxopt.matrix(A)
    
    b = cvxopt.matrix(0.0)
    cvxopt.solvers.options["show_progress"] = False
    
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    
    alpha = np.array(sol['x'])
    
    a_n = alpha[:len(alpha)//2]
    a_n_hat = alpha[len(alpha)//2:]
    a_n = np.array(a_n)
    a_n_hat = np.array(a_n_hat)
    
    #support vectors 
    sv_list = []
    for i in range(len(a_n)):
        if (a_n[i] - a_n_hat[i]) > 1e-5:
            sv_list.append(i)
    print("Number of support vectors: ", len(sv_list))
    return a_n, a_n_hat
    

            

# %%
def kernel_rbf(x1,x2,gamma):
    return np.exp(-gamma*np.linalg.norm(x1-x2)**2)


# %%
def bais(Y_train, a_n, a_n_hat, X_train, K,C=1):
    """Compute the bias term."""
    N = len(X_train)
    kernel_matrix = np.array([[K(X_train[i], X_train[j]) for j in range(N)] for i in range(N)])
    b = 0
    for i in range(N):
        if 0 < a_n[i] < C or 0 < a_n_hat[i] < C:
            sv += 1
            b = Y_train[i] - np.sum((a_n - a_n_hat) * kernel_matrix[i])
            break
    return b

# %%
def linear_kernel(x1,x2):
    return np.dot(x1,x2)

# %%
import matplotlib.pyplot as plt
t_set = [7,30,90]
gamma_set = [1,0.1,0.01,0.001]

epsilon = 0.1

# Linear Kernel
for t in t_set:
    X,y = createXY(t)
    X_train, Y_train, X_test, Y_test = create_trains_tests(X,y)
    K = linear_kernel
    y_pred = []
    a_n , a_n_hat = stock_dual(epsilon, X_train, Y_train, K)
    kernel_mat = np.array([[K(X_train[i], X_train[j]) for j in range(len(X_train))] for i in range(len(X_train))])
    b = bais(Y_train, a_n, a_n_hat, X_train, K)
    for x in X_test:
        k_test = np.array([K(x, X_train[i]) for i in range(len(X_train))])
        y_pred.append(np.dot((a_n - a_n_hat).flatten(), k_test) + b)
    y_pred = np.array(y_pred)
    
    avg_y = np.convolve(Y_test.flatten(), np.ones(t)/t, mode='valid')
    
    plt.figure(figsize=(10, 5))
    plt.plot(y_pred, label="Predicted Closing Price", linestyle='dashed', marker="o")
    plt.plot(Y_test, label="Actual Closing Price", marker="x")
    plt.plot(range(len(avg_y)), avg_y, label=f"{t}-Day Avg Closing Price", linestyle='dotted')
    plt.title(f'SVR Prediction with t = {t} (Linear Kernel)')
    plt.legend()
    plt.xlabel("Days")
    plt.ylabel("Normalized Price")
    plt.show()
    

# RBF Kernel   
for t in t_set:
    X,y = createXY(t)
    X_train, Y_train, X_test, Y_test = create_trains_tests(X,y)
    for gamma in gamma_set:
        K = lambda x1, x2: kernel_rbf(x1, x2, gamma)
        y_pred = []
        a_n , a_n_hat = stock_dual(epsilon, X_train, Y_train, K)
        kernel_mat = np.array([[K(X_train[i], X_train[j]) for j in range(len(X_train))] for i in range(len(X_train))])
        b = bais(Y_train, a_n, a_n_hat, X_train, K)
        for x in X_test:
            k_test = np.array([K(x, X_train[i]) for i in range(len(X_train))])
            y_pred.append(np.dot((a_n - a_n_hat).flatten(), k_test) + b)
        y_pred = np.array(y_pred)
        
        avg_y = np.convolve(Y_test.flatten() , np.ones(t)/t, mode='valid')
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_pred, label="Predicted Closing Price", linestyle='dashed', marker="o")
        plt.plot(Y_test, label="Actual Closing Price", marker="x")
        plt.plot(range(len(avg_y)), avg_y, label=f"{t}-Day Avg Closing Price", linestyle='dotted')
        plt.title(f'SVR Prediction with t = {t} and gamma = {gamma} (RBF Kernel)')
        plt.legend()
        plt.xlabel("Days")
        plt.ylabel("Normalized Price")
        plt.show()
        
