#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from functions import *


# In[ ]:


v_a_b = lazy(10,1)
omega_a_b = lazy(20,2)
k_l_a_b = lazy(2,.2)
k_h_a_b = lazy(.1,.01)
q_a_b = lazy(1,.1)
v_sample = np.random.uniform(v_a_b[0],v_a_b[1],200)
omega_sample = np.random.uniform(omega_a_b[0],omega_a_b[1],200)
k_l_sample = np.random.uniform(k_l_a_b[0],k_l_a_b[1],200)
k_h_sample = np.random.uniform(k_h_a_b[0],k_h_a_b[1],200)
q_sample = np.random.uniform(q_a_b[0],q_a_b[1],200)
x = np.linspace(0,1,50)
y = np.linspace(0,1,2000)
G_soln = []
for i in range(200):
    G_soln.append(ADRSource(1,50,q_sample[i]*x*(10-x),omega_sample[i],v_sample[i],k_l_sample[i])[0])
G_soln = np.array(G_soln)
F_soln = []
for i in range(200):
    F_soln.append(ADRSource(1,2000,q_sample[i]*y*(10-y),omega_sample[i],v_sample[i],k_l_sample[i])[0])
F_soln = np.array(F_soln)
v_test = np.random.uniform(v_a_b[0],v_a_b[1],1)
omega_test = np.random.uniform(omega_a_b[0],omega_a_b[1],1)
k_l_test = np.random.uniform(k_l_a_b[0],k_l_a_b[1],1)
k_h_test = np.random.uniform(k_h_a_b[0],k_h_a_b[1],1)
q_test = np.random.uniform(q_a_b[0],q_a_b[1],1)
g_test = ADRSource(1,50,q_test*x*(10-x),omega_test,v_test,k_l_test)[0]
f_test = ADRSource(1,2000,q_test*y*(10-y),omega_test,v_test,k_l_test)[0]
x1 = np.linspace(0,10,1001)
errarr = []
errarr1 = []
param_grid = []
kernels = ['sigmoid,cosine,linear,poly,rbf']
for j in range(100):
    for k in range(1,20):  
        for kernel in kernels:
            G_soln = []
            for i in range(200):
                G_soln.append(ADRSource(1,50,q_sample[i]*x*(10-x),omega_sample[i],v_sample[i],k_l_sample[i])[0])
            G_soln = np.array(G_soln)
            F_soln = []
            for i in range(200):
                F_soln.append(ADRSource(1,2000,q_sample[i]*y*(10-y),omega_sample[i],v_sample[i],k_l_sample[i])[0])
            F_soln = np.array(F_soln)
            f_kPOD,g_kPOD,fk_means,gk_means = PODMM_Train_KPCA(F_soln.T,G_soln.T,k,'sigmoid',x1[j])
            f_a, g_a, f_ma, g_ma = PODMM_Train(F_soln.T,G_soln.T,k)
            pred2 = PODMM_Predict(g_test,f_kPOD,g_kPOD,fk_means,gk_means,k)
            pred3 = PODMM_Predict(g_test,f_a,g_a,f_ma,g_ma,k)
            errarr.append(np.linalg.norm(pred2-f_test,ord=2))
            errarr1.append(np.linalg.norm(pred3-f_test,ord = 2))
            param_grid.append(np.array(j,k,kernel))
errarr = np.array(errarr)
errarr1 = np.array(errarr1)
wb = []
for i in range(len(errarr)):
    if(errarr[i]<errarr1[i]):
        wb.append(param_grid[i])
print(wb)

