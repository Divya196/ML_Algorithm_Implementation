#Implementing ML Algorithm from scratch
#Implementing ML Algorithm from scratch
Given a data point 

1. Calculate the distance between given data point and all other data points in the dataset 
2.Get the closest K points
3.Regression : Get the average og their values
4.Classification: Get the label with majority vote 

AI- To create an AI application which can perform its own task without any human intervention.

Netflix Recommendation System ,
Self Driving Car

MACHINE LEARNING - Subset of AI , It provides stats tools to analyse,visualize,predict,forecast the data.

DEEP LEARNING - Subset of Machine learning , to memic the human brain. 

DATA SCIENCE - overlapping maths,stats,Machine learning,Deep Learning. 

EQUATION OF A LINE-

y=mx+c

y=beta 0 + beta 1 * x

ax+by+c = 0 

m= slope = with unit movement of x axis,how much movement is there in y axis

c = intercept = when x=0 , where does the straight line meet the y axis

if two features x1,x2 

w1x1 + w2x2 + b = 0 where w1,w2 are co-efficient 

wTx+b =0 

if 3D Plane , 
w1x1 + w2x2 + w3x3 + b = 0

if n Dimensional plane,
w1x1 + w2x2 + w3x3 +.....+ wnxn + b =0

Equation of a straight line passing through an origin is given by , 

w1x1 + w2x2 + b = 0 

w1x1 + w2x2 + 0  = 0

w1x1 + w2x2 = 0

wTx = 0
                                                  
EQUATION OF N DIMENSIONAL PLANE OR HYPERPLANE ,

^                     ^
||n : wTx = 0 , where || is known as pi or plane

w = w1      x=   x1
    w2           x2
    w3           x3   
    .     *       .
    .  .
    .             .
    wn           xn

  -->dot product
w * x = wTx = ||w|| * ||x|| cos teta = 0 

|__________________________________|
|w will be perpendicular to plane  |
|__________________________________|
w is a vector 


LINEAR ALGEBRA 

Distance of a point from plane,

s is a point =(x1,x2,x3....xn) where w is perpendicular to plane 

distance  = wTs / ||w|| = (||w|| ||s|| cos teta) / ||w|| , where ||w|| is magnitude of w and ||s|| is magnitude of s

(||w|| ||s|| cos teta) is going to be between 0 to 1 ie)cos 90 = 0, cos 0 =1 so it will be +ve 

s' is a point =(x1,x2,x3....xn) where w is not perpendicular to plane 

distance  = wTs' / ||w|| = (||w|| ||s'|| cos teta) / ||w|| , where ||w|| is magnitude of w and ||s'|| is magnitude of s'

(||w|| ||s'|| cos teta) is going to be >90 degree so it will be -ve 













