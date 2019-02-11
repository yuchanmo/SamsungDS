from scipy import linalg as l
import numpy as np

def plusol(A,b):    
    P,L,U = l.lu(A)
    m,n = A.shape
    y = np.zeros((m,1))
    c = np.dot(P,b)    
    #y = c - np.dot(L,y)

    for j in range(m):
        y[j] = c[j] - np.dot(L[j,0:j],y[0:j])
    x = np.zeros((n,1))

    for j in range(n-1,-1,-1):
        x[j] = (y[j] - np.dot(U[j,j+1:],x[j+1:]))/U[j,j]
    # x = (y-np.dot(U,x)/U)
    return x

aa = np.array([[2,1,1],[4,-6,0],[-2,7,2]])
bb = np.array([5,-2,9])
res = plusol(aa,bb)
np.dot(aa,res)

def determ(A):
    P,L,U = l.lu(A)
    d = np.dot(l.det(P),np.prod(np.diag(U)))
    return d

aa = np.array([[2,1,1],[4,-6,0],[-2,7,2]])
determ(aa)
l.det(aa)

