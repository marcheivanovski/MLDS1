import numpy as np


betas = np.array([
    [-0.83,-0.15,0.00,0.24,0.02,-0.04,0.31,-0.41,0.14,-0.12,0.13,2.01], #above head
    [1.69,0.14,0.02,-1.27,-0.18,-0.14,0.40,-0.37,0.13,0.11,-0.03,-0.21], #layup
    [0.06,0.05,-0.01,-1.66,0.56,-0.05,0.56,-0.64,0.14,-0.44,-0.76,1.12], #tip in
    [-0.17,-0.60,-0.01,-0.45,0.60,0.03,0.87,-0.84,-0.09,-0.77,-1.41,2.05], #hook shoot
    [0.23,0.25,-0.01,-1.37,0.43,0.90,0.35,-0.87,-0.43,-0.77,-0.92,1.54], #dunk
    [0,0,0,0,0,0,0,0,0,0,0,0]
])
angle_mean, angle_std = (45.35868232484087, 26.05601709952578)
distance_mean, distance_std = (3.886504777070071, 2.7706748536613652)

'''
intercept 
transition 
angle 
distance 
competition\_EURO
competition\_NBA 
competition\_SLO1 
competition\_U14 
playertype\_F 
playertype\_G 
movement\_dribble or cut
movement\_no
'''

xi = np.array([
    1,
    0,
    (28-angle_mean)/angle_std,
    (2-distance_mean)/distance_std,
    0,
    0,
    1,
    0,
    1,
    0,
    0,
    1
])

lin_predictors = np.dot(betas, xi.T)
denominator_sum=np.sum(np.exp(lin_predictors)) 
p = [np.exp(i)/(denominator_sum) for i in lin_predictors]


for i in p:
    print(f'{i:.2f}',end=',')