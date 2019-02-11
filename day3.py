#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\linearalgebra\SamsungDS'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ## Least Squares

#%%
import scipy
from scipy import matrix
from scipy import linalg
import numpy as np


#%%
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
A = matrix([[1, 2], [1,3], [0,0]])
b = matrix([4,5,6]).T


#%%
x, error, _, _ = linalg.lstsq(A, b)


#%%
x


#%%
A*np.asmatrix(x)


#%%
error

#%% [markdown]
# ## Applications
#%% [markdown]
# @Solubility Data
# 
# Tetko et al. (2001) and Huuskonen (2000) investigated a set of compounds with corresponding experimental solubility values using complex sets of descriptors. They used linear regression and neural network models to estimate the relationship between chemical structure and solubility. For our analyses, we will use 1267 compounds and a set of more understandable descriptors that fall into one of three groups: 208 binary "fingerprints" that indicate the presence or absence of a particular chemical sub-structure, 16 count descriptors (such as the number of bonds or the number of Bromine atoms) and 4 continuous descriptors (such as molecular weight or surface area).
# 
# 1267 observation (951+316)
# 
# 228 mixed variables
# 
# www.rdocumentation.org/packages/AppliedPredictiveModeling/versions/1.1-6/topics/solubility

#%%
import pandas as pd
from sklearn import linear_model
import sys
import os

data = {}
f_list = ["solTestX", "solTestY", "solTrainX", "solTrainY"]

#%%
for i in f_list :    
    data[i] = pd.read_csv('C:/Yuchan/linearalgebra/SamsungDS/data/'+i+".csv", index_col=0)
    
print("Data Loaded")


#%%
train_data = data["solTrainX"]
train_label = data["solTrainY"]
test_data = data["solTestX"]
test_label = data["solTestY"]

print("train data : ", train_data.shape)
print("train label : ", train_label.shape)
print("test data : ", test_data.shape)
print("test label : ", test_label.shape)

arr = train_data.columns


#%%
data["solTestX"].head()


#%%
data["solTestY"].head()


#%%
model = linear_model.LinearRegression()
model.fit(train_data, train_label)


#%%
predict_label = model.predict(test_data)


#%%
plt.style.use('classic')
plt.scatter(predict_label, test_label.values)

plt.title('Predict & Real', fontdict={'size':15})
plt.xlabel('Real')
plt.ylabel('Predict')

plt.xlim(-10, 3) 
plt.ylim(-10, 3)

plt.plot(test_label, test_label, 'r-') 
plt.show()


#%%
# ## 0.67일경우 유의미하다고 판단
from sklearn import metrics
print("R2 Score = ", metrics.r2_score(predict_label, test_label))


#%%
model.coef_


#%%
model.intercept_


#%%
col_x = train_data.columns
col_y = train_label.columns
formul = ""
for i in range(0, len(model.coef_.reshape(-1))) :
    if model.coef_.reshape(-1)[i] != 0 :
        formul += str(model.coef_.reshape(-1)[i]) + " * " + col_x[i] + " + "

print("Formula : \n")
print(col_y[0], " = ", formul[:-3], '+', model.intercept_[0])

#%% [markdown]
# ## KNN & Linear Regression
#%% [markdown]
# https://github.com/songsuoyuan/The-Element-of-Statistical-Learning-Python-Implements

#%%
import numpy as np
from sklearn import linear_model
from sklearn import neighbors
from scipy.stats import multivariate_normal


#%%

data = np.loadtxt('C:/Yuchan/linearalgebra/SamsungDS/data/mixture_simulation_data.txt', skiprows=1)
means = np.loadtxt('C:/Yuchan/linearalgebra/SamsungDS/data/means.txt')
X = data[:,:2]
y = data[:,2]


#%%
def linear_regression(X, y):
    '''
    Linear regression model:
        y = w_0 + w_1 * x_1 + ... + w_p * x_p
        coef_ = (w_1, ..., w_p), intercept_ = w_0
    Solution:
        solve min |X_ * w - y|^2 
    where X_ = [ones, X]
    Complexity: O(np^2), based on SVD
    '''
    
    clf = linear_model.LinearRegression()
    clf.fit(X, y)
    
    line = lambda x: ((.5 - clf.intercept_ - clf.coef_[0] * x) / clf.coef_[1])
    line_x = [min(X[:,0])-.2, max(X[:,0]+.5)]
    line_y = list(map(line, line_x))
    
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(X[:,0], X[:,1], c=y, alpha=.6)
    plt.plot(line_x, line_y, color='b', alpha=.8, linewidth=3)
    #plt.plot(line_x, clf.predict(line_x), color='b', alpha=.8, linewidth=3)
    plt.xlim([-3,5])
    plt.ylim([-3,5])
    plt.show()
    
    print('(beta_0) = ' + str(clf.intercept_))
    print('(beta_1, beta_2) = ' + str(clf.coef_))
    print('Precision:', 100. * sum(list(map(round,clf.predict(X))) == y) / len(y))


#%%
linear_regression(X, y)


#%%
def k_nearest_neighbor(X, y, k):
    '''
    K nearest neighbor method:
        y(x) = 1 / k * (y_1 + ... + y_k)
    where y_i belongs to the k closest points to the point x
    Complexity: based on implement, brute force is O(Dn^2)
    '''
    clf = neighbors.KNeighborsRegressor(n_neighbors=k)
    clf.fit(X, y)
    
    delta = .1
    grid_x = np.arange(min(X[:,0])-.5, max(X[:,0])+.5, delta)
    grid_y = np.arange(min(X[:,1])-.5, max(X[:,1])+.5, delta)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_x)
    combine_XY = np.dstack((grid_X,grid_Y)).reshape(grid_X.size,2)
    Z = clf.predict(combine_XY)
    grid_Z = Z.reshape(grid_X.shape)
    
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(X[:,0], X[:,1], c=y, alpha=.6)
    plt.contour(grid_X, grid_Y, grid_Z, 1, alpha=.8,
                colors='b', linewidths=3)
    plt.show()
    
    print('Precision:', 100. * sum(list(map(round, clf.predict(X))) == y) / len(y))


#%%
k_nearest_neighbor(X, y, 15)


#%%
def optimal_bayes(X, y, means):
    '''
    First 10 means m_k are generated from bivariate Gaussian 
    N([0,1],I) and labeled as RED, another 10 means are generated
    from N([1,0],I) and labeled as BLUE. Then 100 RED observations 
    are generated by first pick an m_k at random with p=0.1, then
    generate observation by N(m_k,I/5). Another 100 BLUE 
    observations are generated by the same procedure.
    Optimal Bayes decision attribute G(x) = k-th class where
    P(Y in k-th class | X = x) is the maximum.
    Estimated runtime = 25s
    '''
    
    delta = .1
    grid_x = np.arange(min(X[:,0])-.5, max(X[:,0])+.5, delta)
    grid_y = np.arange(min(X[:,1])-.5, max(X[:,1])+.5, delta)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_x)
    combine_XY = np.dstack((grid_X,grid_Y)).reshape(grid_X.size,2)
    Z = []
    
    for p in combine_XY:
        dist_B = .0
        dist_R = .0
        covar  = [[0.2,0],[0,0.2]]
        for m in means[:10,:]:
            dist_B += multivariate_normal.pdf(p, mean=m, cov=covar)
        for m in means[10:,:]:
            dist_R += multivariate_normal.pdf(p, mean=m, cov=covar)
        Z.append(np.exp(np.log(dist_B) - np.log(dist_R)) - 1.)
    Z = np.array(Z)
    grid_Z = Z.reshape(grid_X.shape)
    
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(X[:,0], X[:,1], c=y, alpha=.6)
    plt.scatter(means[:10,0], means[:10,1], s=80, color='blue')
    plt.scatter(means[10:,0], means[10:,1], s=80, color='red')
    plt.contour(grid_X, grid_Y, grid_Z, 1, alpha=.8,
                colors='b', linewidths=3)
    plt.show()
    
    n = len(y)
    predict = []
    for i in range(n):
        dist_B = .0
        dist_R = .0
        covar  = [[0.2,0],[0,0.2]]
        for m in means[:10,:]:
            dist_B += multivariate_normal.pdf(X[i,:], mean=m, 
                                              cov=covar)
        for m in means[10:,:]:
            dist_R += multivariate_normal.pdf(X[i,:], mean=m, 
                                              cov=covar)
        if (dist_B > dist_R):
            predict.append(0)
        else:
            predict.append(1)
            
    print('Precision:', 100. * sum(predict == y) / len(y))


#%%
optimal_bayes(X,y,means)


