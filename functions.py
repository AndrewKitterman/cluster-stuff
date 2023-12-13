#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import linalg
from scipy.optimize import minimize
from scipy.optimize import least_squares
import sklearn
from sklearn.decomposition import KernelPCA
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from numpy.random import multivariate_normal


# In[2]:


def PODMM_Train_KPCA(f,g,M,ker,gam):
    f1 = np.shape(f)[0]
    f2 = np.shape(f)[1]
    fvals = []
    gvals = []
    easymeans1 = np.mean(f,axis=1)
    easymeans2 = np.mean(g,axis=1)
    easymeans1 = easymeans1.reshape(f1,1)
    easymeans2 = easymeans2.reshape(np.shape(g)[0],1)
    for i in range(np.shape(f)[0]):
        f[:][i] = f[:][i] - easymeans1[i]
    for i in range(np.shape(g)[0]):
        g[:][i] = g[:][i] - easymeans2[i]
    W = np.concatenate((f,g))
    components = KernelPCA(n_components=M,kernel=ker,gamma=gam)
    V = components.fit_transform(W)
    for i in range(M):
        fvals.append(V[:f1,i])
        gvals.append(V[f1:,i])
    fvals,gvals = np.array(fvals),np.array(gvals)
    return fvals.T,gvals.T,easymeans1,easymeans2
def PODMM_Train(f,g,M):
    f1 = np.shape(f)[0]
    f2 = np.shape(f)[1]
    fvals = []
    gvals = []
    easymeans1 = np.mean(f,axis=1)
    easymeans2 = np.mean(g,axis=1)
    easymeans1 = easymeans1.reshape(f1,1)
    easymeans2 = easymeans2.reshape(np.shape(g)[0],1)
    for i in range(np.shape(f)[0]):
        f[:][i] = f[:][i] - easymeans1[i]
    for i in range(np.shape(g)[0]):
        g[:][i] = g[:][i] - easymeans2[i]
    W = np.concatenate((f,g))
    RSV_computed = []
    U,W_svd,V = np.linalg.svd(W,compute_uv=True)
    for i in range(M):
        x = V[:,i]
        RSV_computed.append(np.dot(W,x))
    RSV_computed = np.array(RSV_computed).T
    for i in range(M):
        fvals.append(RSV_computed[:f1,i])
        gvals.append(RSV_computed[f1:,i])
    fvals,gvals = np.array(fvals),np.array(gvals)
    return fvals.T,gvals.T,easymeans1,easymeans2
def PODMM_Predict(g_test,zeta_f,zeta_g,f_bar,g_bar,M):
    alpha_PODMM = []
    g_test = g_test.reshape(-1,1)
    objective_func = lambda y:(g_test-(g_bar + np.dot(zeta_g,y.T).reshape(-1,1))).flatten()
    y0 = np.random.random(M)
    gam = least_squares(objective_func,y0)
    alph = gam.x
    return f_bar + np.dot(zeta_f,alph.T).reshape(-1,1)
def ADRSource(Lx, Nx, Source, omega, v, kappa):
    """
    Solve the advection-diffusion-reaction equation
    input:
    Lx: float, the right end of x
    Nx: int, nunber of x
    Source: 1d array of size Nx
    omega: 1d array of size Nx
    v: 1d array of size Nx+1
    kappa: 1d array of size Nx
    return:
    Solution: 1d array of size Nx
    Q: float, quantiy of interest
    """
    Source = np.full((Nx),Source)
    omega = np.full((Nx),omega)
    v = np.full((Nx),v)
    kappa = np.full((Nx),kappa)
    A = sparse.dia_matrix((Nx,Nx))   
    dx = Lx/(Nx-1)
    i2dx2 = 1.0/(dx*dx)
    #fill diagonal of A
    A.setdiag(2*i2dx2*omega + np.sign(v)*v/dx + kappa)
    #fill off diagonals of A
    A.setdiag(-i2dx2*omega[1:Nx] + 0.5*(1-np.sign(v[1:Nx]))*v[1:Nx]/dx,1)
    A.setdiag(-i2dx2*omega[0:(Nx-1)] - 0.5*(np.sign(v[0:(Nx-1)])+1)*v[0:(Nx-1)]/dx,-1)
    #solve A x = Source
    Solution = linalg.spsolve(A,Source)
    # Trapezoid rule
    Q = np.sum(Solution[1:-1]*kappa[1:-1]*dx) +         Solution[0]*kappa[0]*dx/2 + Solution[-1]*kappa[-1]*dx/2
    return Solution, Q
def lazy(mean,std):
    A = [[1,-1],[1,1]]
    e1 = np.sqrt(12)*std
    e2 = 2*mean
    b = [e1,e2]
    return np.linalg.solve(A,b)


# In[ ]:




