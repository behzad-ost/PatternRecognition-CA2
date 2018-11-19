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

NUM_OF_SAMPLES = 1000

x1=np.random.normal(mu1, sig1, NUM_OF_SAMPLES)
x2=np.random.normal(mu2, sig2, NUM_OF_SAMPLES)
x3=np.random.normal(mu3, sig3, NUM_OF_SAMPLES)
x4=np.random.normal(mu4, sig4, NUM_OF_SAMPLES)

xs = np.concatenate((x1, x2, x3, x4))

labels = ([1] * NUM_OF_SAMPLES) + ([2] * NUM_OF_SAMPLES)+([3] * NUM_OF_SAMPLES)+([4] * NUM_OF_SAMPLES)


data = {'x': xs,'label': labels}
df = pd.DataFrame(data=data)


guess = { 
	'mu': [1, 1.5, 2, 3],
	'sig': [float(1),float(1),float(1),float(1)],
	'lambda': [.25, 0.25,.25,.25]
}

def prob(val, mu, sig, lam):
	p= lam*norm.pdf(val, mu, sig)
	return p



def expectation(dataFrame, parameters):
	for i in range(dataFrame.shape[0]):
		x = dataFrame['x'][i]
		p_cluster1 = prob(x, parameters['mu'][0], parameters['sig'][0], parameters['lambda'][0] )
		p_cluster2 = prob(x, parameters['mu'][1], parameters['sig'][1], parameters['lambda'][1] )
		p_cluster3 = prob(x, parameters['mu'][2], parameters['sig'][2], parameters['lambda'][2] )
		p_cluster4 = prob(x, parameters['mu'][3], parameters['sig'][3], parameters['lambda'][3] )
		pmax=np.max([p_cluster1,p_cluster2,p_cluster3,p_cluster4])
		if p_cluster1 == pmax :
			dataFrame['label'][i] = 1
		elif p_cluster2 == pmax :
			dataFrame['label'][i] = 2
		elif p_cluster3 == pmax :
			dataFrame['label'][i] = 3
		elif p_cluster4 == pmax :
			dataFrame['label'][i] = 4
	return dataFrame



def maximization(dataFrame, parameters):
    points_assigned_to_cluster1 = dataFrame[dataFrame['label'] == 1]
    points_assigned_to_cluster2 = dataFrame[dataFrame['label'] == 2]
    points_assigned_to_cluster3 = dataFrame[dataFrame['label'] == 3]
    points_assigned_to_cluster4 = dataFrame[dataFrame['label'] == 4]

    percent_assigned_to_cluster1 = len(points_assigned_to_cluster1) / float(len(dataFrame))
    percent_assigned_to_cluster2 = len(points_assigned_to_cluster2) / float(len(dataFrame))
    percent_assigned_to_cluster3 = len(points_assigned_to_cluster3) / float(len(dataFrame))
    percent_assigned_to_cluster4 = len(points_assigned_to_cluster4) / float(len(dataFrame))

    parameters['lambda'] = [percent_assigned_to_cluster1, percent_assigned_to_cluster2 , percent_assigned_to_cluster3 , 
                            percent_assigned_to_cluster4 ]

    parameters['mu'][0] = points_assigned_to_cluster1['x'].mean()
    parameters['mu'][1] = points_assigned_to_cluster2['x'].mean()
    parameters['mu'][2] = points_assigned_to_cluster3['x'].mean()
    parameters['mu'][3] = points_assigned_to_cluster4['x'].mean()
    
    parameters['sig'][0] = points_assigned_to_cluster1['x'].std()
    parameters['sig'][1] = points_assigned_to_cluster2['x'].std()
    parameters['sig'][2] = points_assigned_to_cluster3['x'].std()
    parameters['sig'][3] = points_assigned_to_cluster4['x'].std()

    return parameters

def distance(old_params, new_params):
    dist = 0
    for param in [0,1,2,3]:
        dist += (old_params['mu'][param] - new_params['mu'][param]) ** 2
    return dist ** 0.5

diff = 10000.0
epsilon = 0.01
iters = 0
df_copy = df.copy()
df_copy['label'] = map(lambda x: x+1, np.random.choice(4, len(df)))
params = pd.DataFrame(guess)

while diff > epsilon:
    iters += 1
    updated_labels = expectation(df_copy.copy(), params)
    updated_parameters = maximization(updated_labels, params.copy())
    diff = distance(params, updated_parameters)
    df_copy = updated_labels
    params = updated_parameters
    print(params)