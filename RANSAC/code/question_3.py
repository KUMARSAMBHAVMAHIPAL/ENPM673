#!/usr/bin/env python

from numpy.linalg import svd
import numpy as np

def svd(A):
    U = A.dot(A.T)
    V = A.T.dot(A)

    eigval1, eigvec1 = np.linalg.eig(U)
    idx = eigval1.argsort()[::-1]
    eigval1 = eigval1[idx]
    eigvec1 = eigvec1[idx]    

    singular1 = np.diag(np.sqrt(eigval1))
    
    eigval2, eigvec2 = np.linalg.eig(V)
    idx = eigval2.argsort()[::-1]
    eigval2 = eigval2[idx]
    eigvec2 = eigvec2[:,idx]    

    if A.shape[0]>A.shape[1]:
        new_rows = np.zeros((A.shape[0]-A.shape[1], A.shape[1]))
        singular1 = np.vstack([singular1,new_rows])
    else:
        new_cols=np.zeros((A.shape[0], A.shape[1]-A.shape[0]))
        singular1 = np.hstack([singular1,new_cols])

    return eigvec1, singular1, eigvec2

def homography(A):
	U, sigma, V = svd(A)
	return V[-1]

if __name__== "__main__":

	x1 = 5
	y1 = 5
	xp1 = 100
	yp1 = 100

	x2 = 150
	y2 = 5
	xp2 = 200
	yp2 = 80

	x3 = 150
	y3 = 150
	xp3 = 220
	yp3 = 80

	x4 = 5
	y4 = 150
	xp4 = 100
	yp4 = 200

	A = np.array([
	    [-x1,-y1,-1,0,0,0,x1*xp1,y1*xp1,xp1],
	    [0,0,0,-x1,-y1,-1,x1*yp1,y1*yp1,yp1],
	    [-x2,-y2,-1,0,0,0,x2*xp2,y2*xp2,xp2],
	    [0,0,0,-x2,-y2,-1,x2*yp2,y2*yp2,yp2],
	    [-x3,-y3,-1,0,0,0,x3*xp3,y3*xp3,xp3],
	    [0,0,0,-x3,-y3,-1,x3*yp3,y3*yp3,yp3],
	    [-x4,-y4,-1,0,0,0,x4*xp4,y4*xp4,xp4],
	    [0,0,0,-x4,-y4,-1,x4*yp4,y4*yp4,yp4],
	    ],dtype="float64")


	

	x = homography(A).reshape(3,3)
	print("The homography matrix is:\n",x)
