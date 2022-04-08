# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 20:28:09 2021

@author: subbu
"""
#importing libraries required
import numpy as np

#importing data from .csv files
raw_train_data = np.genfromtxt('D:\SEM 6\AV489 MAchine Learning for Signal Processing\Assignment 2\MLSP-02\P1_data_train.csv', delimiter=",") 
raw_train_label = np.genfromtxt('D:\SEM 6\AV489 MAchine Learning for Signal Processing\Assignment 2\MLSP-02\P1_labels_train.csv', delimiter=",")

#preprocessing and MLE estimations
#################################################
iof = [] #number of indexes of 5
ios = [] #numbe of indexes of 6
for i in range(len(raw_train_label)):
    if raw_train_label[i] == 5:
        iof.append(i)
    else:
        ios.append(i)        
train_fives = []  #arrays to store
train_sixes = []  # training data
for j in range(len(iof)):
    train_fives.append(raw_train_data[iof[j]])
for k in range(len(ios)):
    train_sixes.append(raw_train_data[ios[k]])
train_fives = np.array(train_fives)
train_sixes = np.array(train_sixes)  
muf = train_fives.mean(axis=0) #means of C5
mus = train_sixes.mean(axis=0) # and C6

# Different Cases for covariance
#Case 1 both covariance different
covf1 = np.zeros((64,64))
for i in range(len(iof)):
    covf1 = covf1 + np.outer((train_fives[i] - muf),(train_fives[i] - muf))
covf1 = (1/len(iof))*covf1
covs1 = np.zeros((64,64))
for j in range(len(ios)):
    covs1 = covs1 + np.outer((train_sixes[j] - mus),(train_sixes[j] - mus))
covs1 = (1/len(ios))*covs1

#Case 2 both covariance are same 
covf2 = (1/777)*(396*covf1+ 381*covs1)
covs2 = covf2 

#Case 3 diagonal with equal elemants
covf3 = np.zeros((64,64))
for k in range(64):
    b = 0
    for i in range(len(iof)):
        b = b + ((train_fives[i] - muf)[63-k])**2
    for j in range(len(ios)):
        b = b + ((train_sixes[j] - mus)[63-k])**2
    A = (1/(len(ios)+len(iof)))*b
    covf3[k,k] = A
covs3 = covf3
#################################################

# Function to get difference of discriminant functions 
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

#Importing test data
raw_test_data = np.genfromtxt('D:\SEM 6\AV489 MAchine Learning for Signal Processing\Assignment 2\MLSP-02\P1_data_test.csv', delimiter=",") 
raw_test_label = np.genfromtxt('D:\SEM 6\AV489 MAchine Learning for Signal Processing\Assignment 2\MLSP-02\P1_labels_test.csv', delimiter=",")    

# Finding out labels as per the designed classifier
test_labels_model = []
for i in range(333):
    if g(raw_test_data[i],muf,covf1,132/259,mus,covs1,127/259) > 0:
        test_labels_model.append(5)
    else:
        test_labels_model.append(6)
        
#Comparing with original labels of test data to get the confusion matrix
test_labels_model = np.asarray(test_labels_model) 
a = np.column_stack((test_labels_model,raw_test_label)) 
conf_mat = np.zeros((2,2))
for i in range(333):
    if a[i,0] == a[i,1] ==5:
        conf_mat[0,0] = conf_mat[0,0]+1
    elif a[i,0] == a[i,1] == 6:
        conf_mat[1,1] = conf_mat[1,1]+1
    elif a[i,0] == 5 and a[i,1] == 6:
        conf_mat[1,0] = conf_mat[1,0]+1
    else:
        conf_mat[0,1] = conf_mat[0,1]+1
        

#Repeating same thing for the 3 cases Also could be implemented as a function as done in the third question
##########################################
test_labels_model_1 = []
for i in range(333):
    if g(raw_test_data[i],muf,covf2,132/259,mus,covs2,127/259) > 0:
        test_labels_model_1.append(5)
    else:
        test_labels_model_1.append(6)
        
test_labels_model_1 = np.asarray(test_labels_model_1) 
b = np.column_stack((test_labels_model_1,raw_test_label)) 
conf_mat_1 = np.zeros((2,2))
for i in range(333):
    if b[i,0] == b[i,1] ==5:
        conf_mat_1[0,0] = conf_mat_1[0,0]+1
    elif b[i,0] == b[i,1] == 6:
        conf_mat_1[1,1] = conf_mat_1[1,1]+1
    elif b[i,0] == 5 and b[i,1] == 6:
        conf_mat_1[1,0] = conf_mat_1[1,0]+1
    else:
        conf_mat_1[0,1] = conf_mat_1[0,1]+1        
#########################################        
        
        
#########################################       
test_labels_model_2 = []
for i in range(333):
    if g(raw_test_data[i],muf,covf3,132/259,mus,covs3,127/259) > 0:
        test_labels_model_2.append(5)
    else:
        test_labels_model_2.append(6)
test_labels_model_2 = np.asarray(test_labels_model_2) 
z = np.column_stack((test_labels_model_2,raw_test_label)) 
conf_mat_2 = np.zeros((2,2))
for i in range(333):
    if z[i,0] == z[i,1] ==5:
        conf_mat_2[0,0] = conf_mat_2[0,0]+1
    elif z[i,0] == z[i,1] == 6:
        conf_mat_2[1,1] = conf_mat_2[1,1]+1
    elif z[i,0] == 5 and z[i,1] == 6:
        conf_mat_2[1,0] = conf_mat_2[1,0]+1
    else:
        conf_mat_2[0,1] = conf_mat_2[0,1]+1  
###########################################    
       
    
    