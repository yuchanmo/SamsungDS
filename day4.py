#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\linearalgebra\SamsungDS'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ## QR-Factorization

#%%
import scipy
from scipy import matrix
from scipy import linalg
import numpy as np


#%%
# Classical Gram-Schmidt orthogonalization
# MATLAB
# function [Q, R] = clgs(A)
# [m, n] = size(A);
# V=A; Q=eye(m,n);
# R=zeros(n,n);
# for j=1:n
# for i=1:j-1
# R(i,j)=Q(:,i)’*A(:,j);
# V(:,j)=V(:,j)-R(i,j)*Q(:,i);
# end
# R(j,j)=norm(V(:,j));
# Q(:,j)=V(:,j)/R(j,j);
# end


#%%
# Modified Gram-Schmidt orthogonalization
# MATLAB
# function [Q, R] = grams(A)
# [m, n] = size(A);
# Q = A;
# R=zeros(n,n);
# for i = 1:n-1
# R(i,i)=norm(Q(:,i));
# Q(:,i)=Q(:,i)/R(i,i);
# R(i,i+1:n)=Q(:,i)’*Q(:,i+1:n);
# Q(:,i+1:n)=Q(:,i+1:n)-Q(:,i)*R(i,i+1:n);
# end
# R(n,n)=norm(Q(:,n));
# Q(:,n)=Q(:,n)/R(n,n);


#%%
A = matrix([[-1, -1, 1], [1, 3, 3], [-1, -1, 5], [1, 3, 7]])


#%%
A


#%%
Q, R = linalg.qr(A)


#%%
Q = np.asmatrix(Q)
Q


#%%
R = np.asmatrix(R)
R


#%%
Q*R

#%% [markdown]
# ## Equation Solver

#%%
from sympy.solvers import solve
from sympy import Symbol


#%%
x = Symbol('x')
solve(x**2 - 1, x)

#%% [markdown]
# ## Eigenvalues with Equation

#%%
A = matrix([[1, 2], [3, -4]])
A


#%%
lam = Symbol('lam')


#%%
A_lam = A - lam*np.asmatrix(np.identity(2))
A_lam


#%%
equation = A_lam[0,0]*A_lam[1,1] - A_lam[0,1]*A_lam[1,0]
equation


#%%
solve(equation, lam)

#%% [markdown]
# ## Eigenvalues and Eigenvectors with Package

#%%
eigenvalue, eigenvector = linalg.eig(A)


#%%
eigenvalue


#%%
eigenvector

#%% [markdown]
# ## Eigen Value Decomposition

#%%
eigenvalue, eigenvector = linalg.eig(A)


#%%
eigenvalue.shape[0]


#%%
L = np.identity(eigenvalue.shape[0])
for i in range(eigenvalue.shape[0]) :
    L[i, i] = eigenvalue[i]
L


#%%
S= np.asmatrix(eigenvector)
S


#%%
A*S


#%%
S*L


#%%
A*S==S*L


#%%
np.allclose(A*S, S*L)

#%% [markdown]
# ## SVD

#%%
A = matrix([[3, 1, 1], [-1, 3, 1]])
A


#%%
U, s, V = linalg.svd(A, full_matrices=True)


#%%
U = np.asmatrix(U)
U


#%%
s = np.asmatrix(s)
s


#%%
V = np.asmatrix(V)
V


#%%
list(A.shape)


#%%
np.min(list(A.shape))


#%%
S = np.zeros((A.shape))
for i in range(np.min(list(A.shape))) :
    S[i, i] = s[0,i]
S


#%%
U*S*V

#%% [markdown]
# ## Image Compression with SVD
#%% [markdown]
# https://github.com/rameshputalapattu/jupyterexplore/blob/master/jupyter_interactive_environment_exploration.ipynb

#%%
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
img = mpimg.imread('c:/yuchan/linearalgebra/SamsungDS/data/sample.png')
plt.imshow(img)


#%%
from skimage.color import rgb2gray
from skimage import img_as_ubyte, img_as_float

gray_images = {
    "Pierrot":rgb2gray(img_as_float(img))
}


#%%
def compress_svd(image, k):
    U, s, V = linalg.svd(image,full_matrices=False)
    reconst_matrix = np.dot(U[:,:k],np.dot(np.diag(s[:k]),V[:k,:]))
    
    return reconst_matrix, s


#%%
reconst_matrix, s = compress_svd(rgb2gray(img_as_float(img)),50)


#%%
s[:5]


#%%
plt.plot(s[:5])


#%%
def compress_show_gray_images(img_name,k):
    
    image=gray_images[img_name]
    
    original_shape = image.shape
    reconst_img,s = compress_svd(image, k)
    
    fig,axes = plt.subplots(1,2,figsize=(8,5))
    
    axes[0].plot(s)
    
    compression_ratio =100.0* (k*(original_shape[0] + original_shape[1])+k)/(original_shape[0]*original_shape[1])
    
    axes[1].set_title("compression ratio={:.2f}".format(compression_ratio)+"%")
    axes[1].imshow(reconst_img,cmap='gray')
    axes[1].axis('off')
    
    fig.tight_layout()


#%%
from ipywidgets import interact,interactive,interact_manual
interact(compress_show_gray_images,img_name=list(gray_images.keys()),k=(1,100));


