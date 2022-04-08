# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 21:04:01 2021

@author: subbu
"""
#importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#Importing Data
raw_train_data = np.genfromtxt('D:\SEM 6\AV489 MAchine Learning for Signal Processing\Assignment 2\MLSP-02\P2_train.csv', delimiter=",") 
raw_test_data = np.genfromtxt('D:\SEM 6\AV489 MAchine Learning for Signal Processing\Assignment 2\MLSP-02\P2_test.csv', delimiter=",")

#Data Preprocessing and MLE estimations
#################################################
ioz = []
ioo = []
for i in range(310):
    if raw_train_data[i,2] == 0:
        ioz.append(i)
    else:
        ioo.append(i)
train_zeros = []
train_ones = []
for j in range(len(ioz)):
    train_zeros.append(raw_train_data[ioz[j],:2])
for k in range(len(ioo)):
    train_ones.append(raw_train_data[ioo[k], :2])
train_zeros = np.array(train_zeros)
train_ones = np.array(train_ones)  
#means for the two classes
muz = train_zeros.mean(axis=0)
muo = train_ones.mean(axis=0)

#Different cases of covariance matrix
#Case 1 Multiple of identity matrix
a=0
for i in range(len(ioz)):
    a = a + np.dot((train_zeros[i] - muz),(train_zeros[i] - muz))
for j in range(len(ioo)):
    a = a + np.dot((train_ones[j] - muo),(train_ones[j] - muo))
a = (1/(2*(len(ioo)+len(ioz))))*a
covz1 = a*np.eye(2,2)
covo1 = covz1

#Case 2 Diagonal but different elemants
b = 0
for i in range(len(ioz)):
    b = b + ((train_zeros[i] - muz)[1])**2
for j in range(len(ioo)):
    b = b + ((train_ones[j] - muo)[1])**2
A = (1/(len(ioo)+len(ioz)))*b
c = 0
for i in range(len(ioz)):
    c = c + ((train_zeros[i] - muz)[0])**2
for j in range(len(ioo)):
    c = c + ((train_ones[j] - muo)[0])**2
B = (1/(len(ioo)+len(ioz)))*c
covz2 = np.zeros((2,2))
covz2[0,0] = A
covz2[1,1] = B
covo2 = covz2

#Case 3 both are different and arbitrary
covz3 = np.zeros((2,2))
for i in range(len(ioz)):
    covz3 = covz3 + np.outer((train_zeros[i] - muz),(train_zeros[i] - muz))
covz3 = (1/len(ioz))*covz3
covo3 = np.zeros((2,2))
for j in range(len(ioo)):
    covo3 = covo3 + np.outer((train_ones[j] - muo),(train_ones[j] - muo))
covo3 = (1/len(ioo))*covo3

#Case 4 Arbitrary but equal
covz4 = (1/310)*(150*covz3+ 160*covo3)
covo4 = covz4
#the cases presented here and in the report dont match; Cases 3 and 4 are interchanged..
########################################################


#Finding difference of Discriminant function
def g(x,m1,c1,pr1,m2,c2,pr2):
    A = np.log(pr1)
    B = 0.5*((x-m1).dot(np.linalg.inv(c1).dot(x-m1)))
    C = 0.5*(np.log(np.linalg.det(c1)))
    g1 = A-B-C
    D = np.log(pr2)
    E = 0.5*((x-m2).dot(np.linalg.inv(c2).dot(x-m2)))
    F = 0.5*(np.log(np.linalg.det(c2)))
    g2 = D-E-F
    return g1-g2


#Importing Testing Data
test_data = []
test_label = []
for j in range(90):
    test_data.append(raw_test_data[j,:2])
for k in range(90):
    test_label.append(raw_train_data[k,2])
test_data = np.array(test_data)
test_label = np.array(test_label)

######################################
#Set of code to find compare and get the confusion matrix from the predicted data
test_labels_model = []
for i in range(90):
    if g(test_data[i],muz,covz4,150/310,muo,covo4,160/310) > 0:
        test_labels_model.append(0)
    else:
        test_labels_model.append(1)
        
test_labels_model = np.asarray(test_labels_model) 
z = np.column_stack((test_labels_model,test_label)) 
conf_mat = np.zeros((2,2))
for i in range(90):
    if z[i,0] == z[i,1] == 0:
        conf_mat[0,0] = conf_mat[0,0]+1
    elif z[i,0] == z[i,1] == 1:
        conf_mat[1,1] = conf_mat[1,1]+1
    elif z[i,0] == 0 and z[i,1] == 1:
        conf_mat[1,0] = conf_mat[1,0]+1
    else:
        conf_mat[0,1] = conf_mat[0,1]+1 
########################################        
        
        
        
#plotting isoprobability lines

#function to find the probability density
def prob_den(m,c,x):
    a = 1/((4*np.pi**2)*(np.linalg.det(c)))**0.5
    b = np.exp(-0.5*((x-m).dot(np.linalg.inv(c).dot(x-m))))
    return a*b

#Get arrays of X1,X2 for the isoprobability Lines 
def get_arrays(m,c,x,arr):
    t = prob_den(m,c,arr[x])
    #t = prob_den(m,c,train_ones[x])
    sigmainv = np.linalg.inv(c)
    k = t
    X = np.linspace(-8,8,200)
    x_1 = []
    x_2 = []
    Y1 = []
    Y2 = []
    for i in range(len(X)):
        a0 = 0.5*sigmainv[1,1]
        a1 = 0.5*(sigmainv[1,0] + sigmainv[0,1])*(X[i]-m[0]) 
        a2 = (0.5*sigmainv[0,0]*(X[i]-m[0])**2) + np.log(k) + np.log(2*np.pi) + (0.5*np.log(np.linalg.det(c))) 
        ans = np.roots([a0,a1,a2])
        if  isinstance(ans[0],complex) == False:
            Y1.append(ans[0] + m[1])
            x_1.append(i)
        if isinstance(ans[1],complex) ==False:
            Y2.append(ans[1] + m[1])
            x_2.append(i)
    return [X[x_1[0]:x_1[-1] +1],Y1,X[x_2[0]:x_2[-1] +1],Y2]

#plotting descion boundaries
#Function to get Arrays X1,X2 for the descion boundary when the COvariances are different 
def get_arrays_descion_diff_sigmas(m1,c1,m2,c2,p1,p2):
    c1i = np.linalg.inv(c1)
    c2i = np.linalg.inv(c2)
    M = 0.5*(c2i - c1i)
    Md = (np.matmul(c1i,m1) - np.matmul(c2i,m2))
    X = np.linspace(-4,4,200)
    x_1 = []
    x_2 = []
    Y1 = []
    Y2 = []
    for l in range(len(X)):
        a0 = M[1,1]
        a1 = ((M[0,1] + M[1,0])*X[l]) + Md[1]
        a2 =  np.log(p1/p2) + (0.5*np.log((np.linalg.det(c2))/(np.linalg.det(c1)))) + (0.5*(m2).dot((c2i).dot(m2))) - (0.5*(m1).dot((c1i).dot(m1))) + ((X[l]**2)*(M[0,0])) + (Md[0]*X[l])
        ans = np.roots([a0,a1,a2])
        if  isinstance(ans[0],complex) == False:
            Y1.append(ans[0])
            x_1.append(l)
        if isinstance(ans[1],complex) == False:
            Y2.append(ans[1])
            x_2.append(l)
    X1 = []
    X2 = []
    for i in range(len(x_1)):
        X1.append(X[x_1[i]])
    for i in range(len(x_2)):
        X2.append(X[x_2[i]])
    #return [X[x_1[0]:x_1[-1]+1],Y1,X[x_2[0]:x_2[-1] +1],Y2]
    return [X1,Y1,X2,Y2]

#Function to get Arrays X1,X2 for the descion boundary when the Covariances are the same
def get_arrays_descion(m1,c1,m2,c2,p1,p2):
    c1i = np.linalg.inv(c1)
    c2i = np.linalg.inv(c2)
    M = 0.5*(c2i - c1i)
    Md = (np.matmul(c1i,m1) - np.matmul(c2i,m2))
    X = np.linspace(-4,4,20)
    x_1 = []
    Y1 = []
    for l in range(len(X)):
        a0 = M[1,1]
        a1 = ((M[0,1] + M[1,0])*X[l]) + Md[1]
        a2 =  np.log(p1/p2) + (0.5*np.log((np.linalg.det(c2))/(np.linalg.det(c1)))) + (0.5*(m2).dot((c2i).dot(m2))) - (0.5*(m1).dot((c1i).dot(m1))) + ((X[l]**2)*(M[0,0])) + (Md[0]*X[l])
        ans = np.roots([a0,a1,a2])
        if  isinstance(ans[0],complex) == False:
            Y1.append(ans[0])
            x_1.append(l)
    X1 = []
    for i in range(len(x_1)):
        X1.append(X[x_1[i]])
    return [X1,Y1]



#Code to generate arrys and generate Plots
'''som = get_arrays_descion(muz,covz4,muo,covo4,150/310,160/310)
som = get_arrays_descion(muz,covz4,muo,covo4,150/310,160/310)
som1 = get_arrays(muz,covz4,149,train_zeros)
som2 = get_arrays(muz,covz4,40,train_zeros)
som3 = get_arrays(muz,covz4,88,train_zeros)
som4 = get_arrays(muz,covz4,142,train_zeros)
som9 = get_arrays(muz,covz4,100,train_zeros)
som5 = get_arrays(muo,covo4,21,train_ones)
som6 = get_arrays(muo,covo4,153,train_ones)
som7 = get_arrays(muo,covo4,141,train_ones)
som8 = get_arrays(muo,covo4,46,train_ones)
som10 = get_arrays(muo,covo4,159,train_ones)

plt.plot(som[0],som[1])
plt.plot(som1[0],som1[1])
   
#plotting given data 
f1 = []
f2 = []
f12 =[]
for i in range(len(train_zeros)):
    f1.append(train_zeros[i,0])
    f2.append(train_zeros[i,1])
    f12.append(prob_den(muz,covz4,train_zeros[i]))
f3 = []
f4 = []
f34 =[]
for i in range(len(train_ones)):
    f3.append(train_ones[i,0])
    f4.append(train_ones[i,1]) 
    f34.append(prob_den(muo,covo4,train_ones[i]))

fig = plt.figure(1, figsize=(10,20))
ax3 = fig.add_subplot(311)
ax3.scatter(som1[0],som1[1],c='r',s=2)
ax3.scatter(som1[2],som1[3],c='r',s=2)
ax3.scatter(som2[0],som2[1],c='r',s=2)
ax3.scatter(som2[2],som2[3],c='r',s=2)
ax3.scatter(som3[0],som3[1],c='r',s=2)
ax3.scatter(som3[2],som3[3],c='r',s=2)
ax3.scatter(som4[0],som4[1],c='r',s=2)
ax3.scatter(som4[2],som4[3],c='r',s=2)
ax3.scatter(som5[0],som5[1],c='b',s=2)
ax3.scatter(som5[2],som5[3],c='b',s=2)
ax3.scatter(som6[0],som6[1],c='b',s=2)
ax3.scatter(som6[2],som6[3],c='b',s=2)
ax3.scatter(som7[0],som7[1],c='b',s=2)
ax3.scatter(som7[2],som7[3],c='b',s=2)
ax3.scatter(som8[0],som8[1],c='b',s=2)
ax3.scatter(som8[2],som8[3],c='b',s=2)
ax3.scatter(som9[0],som9[1],c='r',s=2)
ax3.scatter(som9[2],som9[3],c='r',s=2)
ax3.scatter(som10[0],som10[1],c='b',s=2)
ax3.scatter(som10[2],som10[3],c='b',s=2)
ax3.scatter(f1,f2,c='g',s=5)
ax3.scatter(f3,f4,c='y',s=5)
ax3.plot(som[0],som[1],c='k')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.gca().set_aspect('equal', adjustable='box')


fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.scatter(f1, f2, f12, c=f12, cmap='autumn', linewidth=2)
ax.scatter(f3, f4, f34, c=f34, cmap='winter', linewidth=2)'''

      
