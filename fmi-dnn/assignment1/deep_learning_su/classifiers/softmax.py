import numpy as np
from random import shuffle
from past.builtins import xrange

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
  y_one_hot = np.zeros((y.shape[0],W.shape[1]))
  y_one_hot[np.arange(y.shape[0]), y] = 1
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    scores = np.exp(scores) / np.sum(np.exp(scores), axis=0)
    # deriv(softmax) = scores*(1-scores)
    
    for j in range(y_one_hot.shape[1]):          
        loss += - y_one_hot[i,j]*(np.log(scores[j]))
        dW[:,j] += X[i]*(scores[j] - y_one_hot[i,j] )
        
  loss = loss / X.shape[0] 
  loss += reg* np.sum(W*W)
  dW += reg* 2 * W
  dW /= X.shape[0]
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
  scores = X.dot(W)
  scores -= np.max(scores,axis=1, keepdims=True)
  scores_exp = np.exp(scores)/ np.sum(np.exp(scores), axis=1)[:,np.newaxis]
  y_one_hot = np.zeros((y.shape[0],W.shape[1]))
  y_one_hot[np.arange(y.shape[0]), y] = 1
  
  scores_exp_fil = scores_exp * y_one_hot
  scores_exp_fil = np.ravel(scores_exp_fil)
  scores_exp_fil = scores_exp_fil [scores_exp_fil>0]
  
  loss = -np.sum(np.log(scores_exp_fil))
  loss = loss / X.shape[0]
  loss += reg* np.sum(W*W)

  dscores = scores_exp
  dscores[np.arange(X.shape[0]),y] -= 1 # this is equal to (scores[j] - y_one_hot[i,j] )

  dW = X.T.dot(dscores) / X.shape[0]
    
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

