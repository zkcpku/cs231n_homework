import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]


  f = np.dot(X,W) # N * C
  f -=np.max(f) # 防止溢出
  f_exp = np.exp(f)

  p = f_exp / np.sum(f_exp,axis = 1).reshape(-1,1) # N * C


  loss = np.mean(-np.log(p[np.arange(N),y])) + reg * np.sum(W * W)


  # dW D * C
  for n in range(0,N):
    f_n = np.dot(X[n],W)#1 * C
    f_n -= np.max(f_n)
    f_n_exp = np.exp(f_n)


    dW[:,y[n]] -= X[n] # D * 1
    for i in range(0,C):
      dW[:,i] += f_n_exp[i] / np.sum(f_n_exp) * X[n]

  dW /= N
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]


  f = np.dot(X,W) # N * C
  f -=np.max(f) # 防止溢出
  f_exp = np.exp(f) # N * C
  s = np.sum(f_exp,axis = 1).reshape(-1,1)

  p = f_exp / s # N * C


  loss = np.mean(-np.log(p[np.arange(N),y])) + reg * np.sum(W * W)

  
  broadcast = f_exp / s

  dW = np.dot(X.T,broadcast)

  flag_y = np.zeros((N,C))
  flag_y[np.arange(N),y] = 1

  dW -= np.dot(X.T,flag_y)
  dW /= N
  dW += 2 * reg * W
  
  # counts = np.exp(f) / s.reshape(N, 1)
  # counts[range(N), y] -= 1
  # dW = np.dot(X.T, counts)

  # dW = dW / N + 2* reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

