import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1] # 
  num_train = X.shape[0] # 
  # 一共N个训练图片
  loss = 0.0
  for i in range(num_train):
  	# y[i]中为正确的标签号码
    scores = X[i].dot(W) # 对每个训练图片计算scores
    correct_class_score = scores[y[i]] # 正确的分组的得分
    for j in range(num_classes): # 对于第j种分类
      if j == y[i]:
        continue # 跳过正确的答案
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0: # max(0,scores[y[i]] - ... + 1)
        loss += margin
        dW[:,j] += X[i,:].reshape(dW[:,j].shape) # 其实都变成(3073,)的维度了
        dW[:,y[i]] += (-X[i,:]).reshape(dW[:,y[i]].shape) #对于一个i，要减去好几次

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg *2* W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  '''
    loss = 0.0
    dW = np.zeros(W.shape)   # initialize the gradient as zero
    scores = X.dot(W)        # N by C
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores_correct = scores[np.arange(num_train), y]   # 1 by N
    scores_correct = np.reshape(scores_correct, (num_train, 1))  # N by 1
    margins = scores - scores_correct + 1.0     # N by C
    margins[np.arange(num_train), y] = 0.0
    margins[margins <= 0] = 0.0
    loss += np.sum(margins) / num_train
    loss += 0.5 * reg * np.sum(W * W)
    # compute the gradient
    margins[margins > 0] = 1.0
    row_sum = np.sum(margins, axis=1)                  # 1 by N
    margins[np.arange(num_train), y] = -row_sum        
    dW += np.dot(X.T, margins)/num_train + reg * W     # D by C

'''
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass

  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]
  scores = np.dot(X,W) # N * D dot D * C
  correct_scores = np.zeros((N,1))
  correct_scores = scores[np.arange(N),y]

  margins = scores - correct_scores.reshape(-1,1)
  margins += 1 # delta = 1.0
  margins[np.arange(N),y] = 0

  flag = (margins >= 0)
  # flag N * C
  margins = margins * flag # max(0,-)
  

  loss = np.sum(margins) / N
  reg_loss = (0.1 * reg * np.sum(W * W))
  loss += reg_loss
  # 这里会炸掉？RuntimeWarning: overflow encountered in double_scalars
  # loss += (reg * np.sum(W * W))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  # D * C
  margins[margins > 0] = 1
  margins[np.arange(N),y] = -np.sum(margins,axis = 1)# 因为对于每个max > 0 时都要做减法,而且对于同一个样本，每一项都应该减去相同的值
  dW = np.dot(X.T,margins) / N + 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
