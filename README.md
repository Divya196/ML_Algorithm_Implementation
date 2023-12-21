
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

^                     
||n : wTx = 0 , 

      ^
where || is known as pi or plane

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


Instance Based Learning vs Model Based Learning
                               ___________
                              | usecase   |
                              |___________|
                                    |
                              ___________                             
                            |             | ----- Regression
                            |  ML Model   | ----- Classification
                             _____________ 
                                    | 
                            -----------------------          
                            |                      |
                   Instance Based              Model Based Technique 
                        |                            |
  Learning religiously from training Data           Data
                  {KNN}                              |
  memorizing the data based on surrounding data     pattern - CREATE DECISION BOUNDARIES
                                                     |
                                                  Generalised method to learn the pattern of the data
                                                


Usual/conventional/model based machine learning         Instance based machine learning

1)Train model from training data to estimate 
model parameters,discover patterns

2)store the model in suitable form,h5,pkl

3)generalise the rule in serialised format in the
form of model even before scoring instance is 
seen

4)predict for unseen scoring instance using model

5)can throw away input or trained data after model 
training

6)requires a known model form

7)storing models requires less storage

8)storing for new instance is generally fast

SIMPLE LINEAR REGRESSION: applicable in deep learning(ANN)

   In Supervised Machine learning technique,if we have regression problem,we can solve using linear regression ,if we have one input feature then it is called simple linear regression.If we have many input feature then it is called multiple linear regression.
               ________
new i/p---->  | Model  | --->o/p
               ________

   Dataset 

   Weight  Height 
    78       170 cm
    80       180 cm
    75       175 cm 
    -         -

 TECHNIQUE TO SELECT BEST FIT LINE :
 
 summation of distance between True points and predicted points should be minimal for the best fit line .

 COST FUNCTION 
 
 h0(x)= 0o + 01 x

 J(0o,01) , J(intercept,slope) = 1/2m * summation of (h0(x)i - (y)i)2 , i= 1 to m , h0 = predicted, y = true points , this is Mean Squared Error.

 Final aim what we need to solve, minimize J(0o,01) 

consider 0o(intercept) = 0 ,
______________
| h0(x) = 01 x | ---->equation 
_______________

let 01(slope) = 1 ,

h0(x) = 1 , x=1 
h0(x) = 2 , x=2 
h0(x) = 3 , x=3 

in dataset , we have 3 datapoints,

J(01) = 1/2*3 [(1-1)^2 + (2-2)^2 + (3-3)^2] = 0.5

let 01(slope)= 0.5,using equation,

h0(x) = 0.5 * 1 = 0.5 ,if x=1
h0(x) = 0.5 *2 = 1 , if x = 2
h0(x) = 0.5 *3 = 1.5 , if x = 3 

J(01) = 1/2*3 [(0.5-1)^2+(1-2)^2 + (1.5-3)^2] ~= 0.58

let 01(slope) = 0 
J(01) = 1/2*3[(0-1)^2+(0-2)^2+(0-3)^3] ~= 2.3

continuing by different 01(slope) value,we will be geting curve known as gradient desent-----> 1

where the error is minimum,that point is  known as global minima. That is the best fit line

By 1 ,we cant randomly select different different 01 value,that is not possible , so we should apply convergence algorithm.

In that we select 01 value and find out the mechanism of changing 01 value




1.
What is the main difference between supervised and unsupervised learning?
In Supervised learning,dependent feature will be there. When output is continuous variable it is regression problem,when output is fixed number of classes it is classification problem. In unsupervised learning,label or dependent feature won't be there,when finding groups it is clustering problem,when we need to lower dimensions is is dimensionality reduction problem.


3.
Explain the concept of reinforcement learning and provide an example.
Re-inforcement learning is for making better decisions. Example: Robot driving a car. Robot is the agent,car is the environment.For every correct action of agent,it will receive reward,for every incorrect action of agent,it will get feedback or observations. Reinforcement learning is kind of trail and error,main aim of the agent is to receive maximum rewards.

5.
Can you describe the working principle of k-nearest neighbors (KNN) algorithm in machine learning?

7.
Differentiate between decision tree and random forest algorithms in machine learning.
8.
What are the key characteristics of deep learning and how is it different from traditional machine learning algorithms?
9.
Explain the working of the support vector machine (SVM) algorithm and its applications in machine learning.
10.
Discuss the concept of clustering in unsupervised learning and provide an example of a clustering algorithm.
11.
What is the role of feature engineering in machine learning and how does it contribute to model performance?
12.
Define ensemble learning and discuss its significance in improving machine learning models.
13.
Can you explain the concept of dimensionality reduction in machine learning and its impact on model complexity?
14.
Describe the working of the Naive Bayes algorithm and its applications in text classification.






 

 

 

