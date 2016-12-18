"""
Instance based learning :

Input dataset : Each instance has four parameters 
                - temperature 
                - pressure 
                - accelaration given to vehicle
                - Distnace covered by vehicle under those parameters
              
* The distance is categorised into 4 classes : 1, 2, 3, 4 based on different ranges.

Test instnace : Contains three parameters 
                - temperature 
                - pressure 
                - accelaration given to vehicle
                
* Back-propogation artificial neural network is used to find the distance class of
    test instance.
"""

#%%
"""
The sigmoid activation function used .
when deriv is true , it returns the sigmoid derivative
"""
import numpy as np
import pandas

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
#%%
"""
Loading the dataset from a csv file.
Each instance has 3 inputs and 1 output
"""
 
def generate_data(): 
    df = pandas.read_csv('TPMScheckdata1.csv')
    df_numeric = df.ix[:,0:3]

    df_numeric = (df_numeric - df_numeric.mean()) / df_numeric.std()
    #df_numeric.fillna(0,inplace = True)
    inputlist = []
    outputlist = []
    
    for row in df_numeric.itertuples():
        content = list(row[0:3])
        inputlist.append(content)
         
    for row in df.itertuples():
        content = list(row[4:5])
        outputlist.append(content)
        
    return inputlist , outputlist
    
A,b = generate_data()
    
X = np.array(A)
y = np.array(b).T

#%%
"""
Generating random synaptic weight matrix (with mean 0)for the neural network 
based on input dataset dimensions.

"""
np.random.seed(1)

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

#%%
"""
Full-batch training the model with the input dataset.
     - Feed forward the instance input throught the layers
     - Adjust the synaptic weights between layer 1 and layer2 based on 
             . input l1 
             . error at l2
             . slope of ouput at l2
    - Adjust the synaptic weights between layer 0 and layer 1 based on
            . input l0
            . error at l1
            . slope of output at l1
    - Back-propogation of synaptic weights at layer 1 and layer 0
    - Add the obtained adjustements for every iteration
    
"""

for j in range(60000):
    l0 = X                           
    l1 = nonlin(np.dot(l0,syn0))     
    l2 = nonlin(np.dot(l1,syn1))         
   
    l2_error = y.T - l2  
    l2_delta = l2_error*nonlin(l2,deriv=True)

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
#%%
"""
Feeding test instnace to model .
Collecting the result value .

"""
testInstance = np.array([70,532,98])
l0 = testInstance
l1 = nonlin(np.dot(l0,syn0))
l2 = nonlin(np.dot(l1,syn1))

#%%
"""
Deciding the distance class based on the result collected.

The value of result tells the class the test belongs to . 
The result is in points in range of 0-1 where : 

    - [0    : 0.25] :    Class 1
    - [0.25 : 0.5]  :    Class 2
    - [0.5  : 0.75] :    Class 3
    - [0.75 : 1]    :    Class 4
 
"""
value = l2[0]
if(value <0.25):
    print("Test instance belongs to class 1 %f" % value)
elif(value < 0.5):
    print("Test instance belongs to class 2 %s" % value)
elif(value < 0.75):
    print("Test instance belongs to class 3 %s" % value)
else:
    print("Test instance belongs to class 4 %f" % value)






