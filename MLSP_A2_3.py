# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 20:53:51 2021

@author: subbu
"""
#importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection 
from pol_reg import pol_reg

#importing datasets 
data = np.genfromtxt('D:\SEM 6\AV489 MAchine Learning for Signal Processing\Assignment 2\MLSP-02\Wage_dataset.csv', delimiter=",") 
#creating different arrays for different attributes
year = []
age = []
edu = [] 
wage = []
for i in range(3000):
    year.append(data[i,0])
    age.append(data[i,1])
    edu.append(data[i,4])
    wage.append(data[i,10])

#obtaining Means Standard Deviations Covariances and hence the correlation coefficient    
sty =np.std(year)
sta =np.std(age)
ste =np.std(edu)
y_a = np.cov(year,age)
e_a = np.cov(edu,age)
y_e = np.cov(year,edu)
rho_ya = (y_a[0,1])/(sty*sta)
rho_ea = (e_a[0,1])/(ste*sta)
rho_ye = (y_e[0,1])/(sty*ste)

#Polynomial Regression
#creating arrays to be used as x axis for plots obtained from polynomial regression
ages = np.linspace(20,80,100)
years = np.linspace(2003,2009,7)
edus = np.linspace(1,5,5)
#fitting the data into polynomials using the self defined function for regression from lin_reg.py
fit_edu = pol_reg(edu,wage,5)
fit_age = pol_reg(age,wage,5)
fit_year = pol_reg(year,wage,5)
#function to return the polynomial obtained through regression 
def attr_model(x,yc):
    ans = []
    a0 = yc[0]
    a1 = yc[1]
    a2 = yc[2]
    a3 = yc[3]
    a4 = yc[4]
    a5 = yc[5]
    for i in range(len(x)):
        ans.append(a0 + (a1*x[i]) + (a2*x[i]*x[i]) + (a3*(x[i])**3) + (a4*(x[i])**4) + (a5*(x[i])**5))
    return ans
#getting relevant plots
'''plt.scatter(year,wage)
plt.scatter(age,wage)
plt.scatter(edu,wage)
plt.plot(ages,attr_model(ages,fit_age),'r-')        
plt.plot(edus,attr_model(edus,fit_edu),'r-')    
plt.plot(years,attr_model(years,fit_year),'r-')
plt.xlabel('education')
plt.ylabel('wage')'''

# Modified data to consist only the 3 attributes used
model_data = np.genfromtxt('D:\SEM 6\AV489 MAchine Learning for Signal Processing\Assignment 2\MLSP-02\Wage_my_dataset.csv', delimiter=",")
mu = np.mean(wage)

#labelling the data based on weather it is greater or lesser than the mean
labels = np.zeros(3000)
for i in range(3000):
    if model_data[i,3] - mu > 0:
        labels[i] = 1
X = []
for i in range(3000):
    X.append(model_data[i,0:3])

#Dividing the dataset into testing and training sets and using MLE to obtain the means and covariances
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X , labels , test_size=0.20, random_state=101, shuffle=True)
ioz = []
ioo = []
for i in range(len(y_train)):
    if y_train[i] == 0:
        ioz.append(i)
    else:
        ioo.append(i)
train_zeros = []
train_ones = []
x_test = np.array(x_test)
for j in range(len(ioz)):
    train_zeros.append(x_train[ioz[j]])
for k in range(len(ioo)):
    train_ones.append(x_train[ioo[k]])
train_zeros = np.array(train_zeros)
train_ones = np.array(train_ones)  

#finding means
muz = np.mean(train_zeros)
muo = np.mean(train_ones)

#finding covariance
covz = np.zeros((3,3))
for i in range(len(ioz)):
    covz = covz + np.outer((train_zeros[i] - muz),(train_zeros[i] - muz))
covz = (1/len(ioz))*covz
covo = np.zeros((3,3))
for j in range(len(ioo)):
    covo = covo + np.outer((train_ones[j] - muo),(train_ones[j] - muo))
covo = (1/len(ioo))*covo

#function to obtain difference of the discriminant functions when three attributes are used
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
#function to obtain difference of the discriminant functions when one attributes are used
def gd(x,m1,c1,pr1,m2,c2,pr2):
    A = np.log(pr1)
    B = 0.5*(x**2)*(1/c1**2)
    C = (m1*x)/(c1**2)
    D = 0.5*(m1/c1)**2
    E = np.log(c1)
    g1 = A+B+C+D+E
    F = np.log(pr2)
    G = 0.5*(x**2)*(1/c2**2)
    H = (m2*x)/(c2**2)
    I = 0.5*(m2/c2)**2
    J = np.log(c2)
    g2 = F+G+H+I+J
    return g1-g2

#Obtaining confusion matrix when 3 attributes are used
test_labels_model = []
for i in range(600):
    if g(x_test[i],muz,covz,1384/2400,muo,covo,1016/2400) > 0:
        test_labels_model.append(0)
    else:
        test_labels_model.append(1)
        
test_labels_model = np.asarray(test_labels_model) 
z = np.column_stack((test_labels_model,y_test)) 
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
 
        
#Creating test and training datasets
train_zeros_1 = train_zeros[:,0]
train_ones_1 = train_ones[:,0]
train_zeros_2 = train_zeros[:,1]
train_ones_2 = train_ones[:,1]
train_zeros_3 = train_zeros[:,2]
train_ones_3 = train_ones[:,2]
x_test_1 = x_test[:,0]
x_test_2 = x_test[:,1]
x_test_3 = x_test[:,2]


#Function to obtain Confusion matrix when one attribute is used
def give_conf_mat(x0,x1,x_t,y_t):
    mu0 = np.mean(x0)
    mu1 = np.mean(x1)
    #finding covariance
    cov0 = 0
    for i in range(len(x0)):
        cov0 = cov0 + np.outer((x0[i] - mu0),(x0[i] - mu0))
    cov0 = (1/len(x0))*cov0
    cov1 = 0
    for j in range(len(x1)):
        cov1 = cov1 + np.outer((x1[j] - mu1),(x1[j] - mu1))
    cov1 = (1/len(x1))*cov1
    test_labels_model = []
    for i in range(600):
        if gd(x_t[i],mu0,cov0,1384/2400,mu1,cov1,1016/2400) > 0:
           test_labels_model.append(0)
        else:
           test_labels_model.append(1)
    test_labels_model = np.asarray(test_labels_model) 
    z = np.column_stack((test_labels_model,y_t)) 
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
    return conf_mat
    
year_conf_mat = give_conf_mat(train_zeros_1,train_ones_1,x_test_1,y_test)
age_conf_mat = give_conf_mat(train_zeros_2,train_ones_2,x_test_2,y_test)
edu_conf_mat = give_conf_mat(train_zeros_3,train_ones_3,x_test_3,y_test)