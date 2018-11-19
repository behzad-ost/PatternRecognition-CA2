import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm
from sys import maxint

rand.seed(30)

mu1 = 1
sig1 = 0.01
mu2 = 1.4
sig2 = 0.13
mu3 = 2
sig3 = 0.05
mu4 = 3.3
sig4 = 0.02


NUM_OF_SAMPLES = 4000
I = 2

x1=np.random.normal(mu1, sig1, NUM_OF_SAMPLES/4)
x2=np.random.normal(mu2, sig2, NUM_OF_SAMPLES/4)
x3=np.random.normal(mu3, sig3, NUM_OF_SAMPLES/4)
x4=np.random.normal(mu4, sig4, NUM_OF_SAMPLES/4)

xs = np.concatenate((x1, x2, x3, x4))

labels = ([1] * (NUM_OF_SAMPLES/I)) + ([2] * (NUM_OF_SAMPLES/I))


data = {'x': xs,'label': labels}
df = pd.DataFrame(data=data)


guess = { 
	# 'mu': [1, 1.4, 2, 3.3],
	'mu': [1, 1.5],
	'sig': [float(1),float(1)],
	'lambda': [.25, 0.25]
}

def prob(val, mu, sig, lam):
	p= lam*norm.pdf(val, mu, sig)
	return p



def expectation(dataFrame, parameters):
	for i in range(dataFrame.shape[0]):
		x = dataFrame['x'][i]
		p_cluster1 = prob(x, parameters['mu'][0], parameters['sig'][0], parameters['lambda'][0] )
		p_cluster2 = prob(x, parameters['mu'][1], parameters['sig'][1], parameters['lambda'][1] )
		pmax=np.max([p_cluster1,p_cluster2])
		if p_cluster1 == pmax :
			dataFrame['label'][i] = 1
		elif p_cluster2 == pmax :
			dataFrame['label'][i] = 2
	return dataFrame



def maximization(dataFrame, parameters):
    points_assigned_to_cluster1 = dataFrame[dataFrame['label'] == 1]
    points_assigned_to_cluster2 = dataFrame[dataFrame['label'] == 2]

    percent_assigned_to_cluster1 = len(points_assigned_to_cluster1) / float(len(dataFrame))
    percent_assigned_to_cluster2 = len(points_assigned_to_cluster2) / float(len(dataFrame))

    parameters['lambda'] = [percent_assigned_to_cluster1, percent_assigned_to_cluster2]
    parameters['mu'][0] = points_assigned_to_cluster1['x'].mean()
    parameters['mu'][1] = points_assigned_to_cluster2['x'].mean()
    
    parameters['sig'][0] = points_assigned_to_cluster1['x'].std() 
    parameters['sig'][1] = points_assigned_to_cluster2['x'].std() 


    return parameters


def distance(old_params, new_params):
    dist = 0
    for param in [0,1]:
        dist += (old_params['mu'][param] - new_params['mu'][param]) ** 2
    return dist ** 0.5


diff = 10000.0
epsilon = 0.01
iters = 0
df_copy = df.copy()
df_copy['label'] = map(lambda x: x+1, np.random.choice(2, len(df)))
params = pd.DataFrame(guess)
print(params)



while diff > epsilon:

    iters += 1
    updated_labels = expectation(df_copy.copy(), params)
    updated_parameters = maximization(updated_labels, params.copy())
    diff = distance(params, updated_parameters)
    df_copy = updated_labels
    params = updated_parameters
    print(params)