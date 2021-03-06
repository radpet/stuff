import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # d(scores[j]) = d(X[i]*W[:,j]) = X[i] 
        dW[:,j] += X[i] # + d(scores[j])
        dW[:,y[i]] -=X[i] # - d(correct_class_score) = - X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
    
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg*2*W

  

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
    
  scores = X.dot(W)
  # https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html
  # np.arange() = [0 - N -1], y -> correct classes
  y_scores = scores[np.arange(X.shape[0]),y]
  margins = np.maximum(0, scores - y_scores[:,np.newaxis] + 1)
  margins[np.arange(X.shape[0]),y] = 0
    
  loss = np.sum(np.sum(margins, axis=1), axis=0) / X.shape[0]

  loss += reg*np.sum(W*W)

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

  dmargins = margins
  # we need add (num_classes - count(j ==y[i]) times x[i]) since dwj += x[i] (will use row*column rule for that later on)
  # margin = 0 for j == y[i]
  dmargins[dmargins > 0] = 1 
  # for those j == y[i] we need to - (num_classes - count(j ==y[i]) times x[i]) because dwy[i] -= x[i]
  dmargins[np.arange(X.shape[0]),y] = -np.sum(dmargins,axis=1)
  dW = X.T.dot(dmargins) # X.T -> every x[j] * how many times x[i] should be + or -
  dW /= X.shape[0]
  dW += 2*reg*W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
