import numpy as np
import pandas as pd
from random import seed
from random import random
import matplotlib.pyplot as plt
import networkx as nx
'''P = np.array([[0.2, 0.7, 0.1],
              [0.9, 0.0, 0.1],
              [0.2, 0.8, 0.0]])
state=np.array([[1.0, 0.0, 0.0]])
stateHist=state
dfStateHist=pd.DataFrame(state)
distr_hist = [[0,0,0]]
for x in range(50):
  state=np.dot(state,P)
  print(state)
  stateHist=np.append(stateHist,state,axis=0)
  dfDistrHist = pd.DataFrame(stateHist)
  dfDistrHist.plot()
plt.show()'''
import random
from  CreateEnvironment import Graph_util


def get_steady(Q):
  #note: the matrix is row stochastic.
  #A markov chain transition will correspond to left multiplying by a row vector.
  Q = np.array([
      [.95, .05, 0., 0.],
      [0., 0.9, 0.09, 0.01],
      [0., 0.05, 0.9, 0.05],
      [0.8, 0., 0.05, 0.15]])

  #We have to transpose so that Markov transitions correspond to right multiplying by a column vector.  np.linalg.eig finds right eigenvectors.
  evals, evecs = np.linalg.eig(Q.T)
  evec1 = evecs[:,np.isclose(evals, 1)]

  #Since np.isclose will return an array, we've indexed with an array
  #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
  evec1 = evec1[:,0]

  stationary = evec1 / evec1.sum()

  #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
  stationary = stationary.real

  return (stationary)


def TransitionsMatrix( vector, digits ):

    # defined matrix size with zeros
    MatrixResult = np.zeros((10**digits, 10**digits), dtype=float)
    
    for iterator in range( 10**digits ):

        if iterator in vector:

            index = vector.index(iterator)
            amount = vector.count(iterator)

            if ( iterator == vector[ len(vector) - 1 ] ):
                amount = vector.count(iterator) - 1

            for count in range( amount ):

                MatrixResult[ vector[index] ][ vector[index + 1] ] += 1

                if ( count < vector.count(iterator) - 1 ):
                    index = vector.index( iterator, index + 1 )

    return MatrixResult

def TransitionProbability(matrixResult, vector):
    sizeMatrix = len(matrixResult)
    for row in range(sizeMatrix):
        for col in range(sizeMatrix):
            if matrixResult[row][col] != 0:
                matrixResult[row][col] = matrixResult[row][col] / vector.count(row)
    return matrixResult
  
graph = Graph_util(10)
adjaceny_matrix = nx.to_numpy_matrix(graph.G)
#states = [1,1,2,6,8,5,5,7,8,8,1,1,4,5,5,0,0,0,1,1,4,4,5,1,3,3,4,5,4,1,1]
states = [0 for i in range(10)]
states[1] = 1
print( TransitionsMatrix( states, 1 ) )

#states = [1,1,2,6,8,5,5,7,8,8,1,1,4,5,5,0,0,0,1,1,4,4,5,1,3,3,4,5,4,1,1]
MatrixResult = TransitionProbability( TransitionsMatrix(states, 1), states )
print( MatrixResult )
