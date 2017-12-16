# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:37:59 2017

@author: bharg
"""

import numpy as np
import random

def normalize(p1,p2):
    x=p1/(p1+p2)
    y=p2/(p1+p2)
    return(np.array([x,y]))

P1=np.array([0.4,0.6])   #p(theta1)
P21={'R':np.array([[0.9],[0.2]]),'L':np.array([[0.1],[0.8]])}  #p(theta2 | theta1) evidence as seen through the bayes-net
P12R=np.diag(P21['R']*P1).T  #P(theta1 | theta2=R)
P12L=np.diag(P21['L']*P1).T  #P(theta1 | theta2=L)
P12R=np.array([normalize(P12R[0],P12R[1])]) #normalized P(theta1 | theta2=R)
P12L=np.array([normalize(P12L[0],P12L[1])]) #normalized P(theta1 | theta2=L)
def sampleprobab(iterations):
    picksample1=random.choice([0,1])   #0=tails, 1=heads 
    picksample2=random.choice(['R','L'])  #'R' is right, 'L' is left
    count1=count2=0
    for _ in range(iterations):   
        u=random.uniform(0,1)
        getsample=random.choice([picksample1,picksample2])
        if getsample=='R':
            count1+=1
            if u<=P12R[0][0]:
                picksample1=1  #heads
            else:
                picksample1=0   #tails
        elif getsample=='L':
            count2+=1
            if u<=P12L[0][0]:
                picksample1=1
            else:
                picksample1=0
            
        if getsample==0:
            if u<=P21['R'][1]:
                picksample2='R'
                count1=count1+1
            else:
                picksample2='L'
                count2+=1
        elif getsample==1:
            if u<=P21['R'][0]:
                picksample2='R'
                count1=count1+1
            else:
                picksample2='L'
                count2+=1
    return(count1,count2)
            
samples=int(input("enter the number of samples you want to run gibbs sampling for: \n"))
count1,count2=sampleprobab(samples)
print("sampled probability for left and right are {} and {}".format(count1/samples,count2/samples))
P2R=np.matmul(P21['R'].T,P1)  #P(theta2='right')
P2L=np.matmul(P21['L'].T,P1)#P(theta2='left')
P2=np.vstack((P2R,P2L))   #p(theta2)       
print("probability computed manually from conditional distribution tables available:\n {}".format(P2))