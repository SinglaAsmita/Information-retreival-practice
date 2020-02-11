# TCSS 554 A - HW2, Submitted by: Asmita Singla

import numpy as np

# Input adjacency matrix for graph
M = np.array([[0, 0, 0, 1, 0],
              [1/2, 0, 0, 0, 0],
              [0, 1/2, 0, 0, 0],
              [1/2, 1/2, 0, 0, 1],
              [0, 0, 0, 0, 0]])
print("M =", M)

# Rank vector
r = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
print("r =", r)
print("Dot Product =", M.dot(r))

# Calculate the transposed array
n = 5
beta = 0.85
e = np.array([[1, 1, 1, 1, 1]])
et = e.transpose()
T = 1/n * (np.multiply(e, et))
print("T =", T)

# Calculate the value of Matrix A
A = (beta * M) + ((1 - beta) * T)
print("A =", A)

# Perform convergence operation for M
limit = 100
count = 1
epsilon = 0.00001
while count <= limit:

    rPrevious = r
    r = M.dot(r)
    count = count + 1  # update counter
    np.set_printoptions(formatter={'float_kind': '{:f}'.format})
    rDifference = np.absolute(np.subtract(rPrevious, r))
    print("r difference", rDifference)
    print("r' =", r)
    if not (np.any(rDifference > epsilon)):
        break
print("Final Count: ", count)

# Perform convergence operation for A
limit = 100
count = 1
r = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
while count <= limit:
    rPrevious = r
    r = A.dot(r)
    count = count + 1  # update counter
    np.set_printoptions(formatter={'float_kind': '{:f}'.format})
    rDifference = np.absolute(np.subtract(rPrevious, r))
    print("r difference", rDifference)
    print("r' =", r)
    if not (np.any(rDifference > epsilon)):
        break
print("Final Count: ", count)
