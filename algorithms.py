"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE

def kl(x,y):
	if x == 0:
		return -math.log(1-y)
	elif x == 1:
		return -math.log(y)
	elif y == 0 or y == 1:
		return float('inf')
	else:
		return x*math.log(x/y) + (1-x)*math.log((1-x)/(1-y))


def findQ(p, rhs):

	if p==1:
		return 1.
	tol = 0.0001
	minq = p
	maxq = 1-0.00001
	if abs(kl(p,minq)-rhs)<=tol:
		return minq
	elif abs(kl(p,maxq)-rhs)<=tol:
		return maxq
	q = 0.5*(minq+maxq)
	if abs(kl(p,q)-rhs)<=tol:
		return q
	while abs(kl(p,q)-rhs)>tol:
		if kl(p,q)>rhs:
			maxq = q
		elif kl(p,q)<rhs:
			minq = q
		q = 0.5*(minq + maxq)
		if abs(kl(p,q)-rhs<tol):
			return q

# You can use this space to define any helper functions that you need
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE

        self.emp_mean = np.zeros(num_arms)
        self.clicks = np.zeros(num_arms)
        self.ucb = np.zeros(num_arms)
        self.round_robin = 0
        self.count = 0

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE

        
        if self.round_robin < self.num_arms:
            idx = self.round_robin
            self.round_robin += 1
            return idx
        else:
            return np.argmax(self.ucb)

        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.count+=1
        self.clicks[arm_index] += 1
        n = self.clicks[arm_index]
        value = self.emp_mean[arm_index]
        self.emp_mean[arm_index] = ((n - 1) / n) * value + (1 / n) * reward

        for i in range(0, self.num_arms):
            if i != arm_index and self.clicks[i] != 0:
                self.ucb[i] = self.emp_mean[i] + math.sqrt(2*math.log(self.count+1)/self.clicks[i])

        self.ucb[arm_index] = self.emp_mean[arm_index] + math.sqrt(2*math.log(self.count+1)/self.clicks[arm_index])
        

        # END EDITING HERE

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE

        self.emp_mean = np.zeros(num_arms)
        self.clicks = np.zeros(num_arms)
        self.ucb = np.ones(num_arms)
        self.round_robin = 0
        self.t = 0

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE

        if self.round_robin < self.num_arms:
            idx = self.round_robin
            self.round_robin += 1
            return idx % self.num_arms
        else:
            return np.argmax(self.ucb)
        
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.t += 1
        self.clicks[arm_index] += 1
        n = self.clicks[arm_index]
        value = self.emp_mean[arm_index]
        self.emp_mean[arm_index] = ((n - 1) / n) * value + (1 / n) * reward


        for i in range(self.num_arms):
            if self.round_robin < self.num_arms:
                target = (math.log(self.t+2) + 3.0*math.log(math.log(self.t+2)))/2
                self.ucb[i] = findQ(self.emp_mean[i],target)
            else:
                target = (math.log(self.t+1) + 3.0*math.log(math.log(self.t+1)))/self.clicks[i]
                self.ucb[i] = findQ(self.emp_mean[i], target)
        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE

        self.clicks = np.zeros(num_arms)
        self.means = np.zeros(num_arms)
        self.success = np.zeros(num_arms)



        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE


        for i in range(self.num_arms):
            self.means[i] = np.random.beta(1+self.success[i], 1+self.clicks[i]-self.success[i], size = None)
        
        return np.argmax(self.means)

        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE

        self.clicks[arm_index] += 1
        if reward == 1:
            self.success[arm_index] += 1
        

        # END EDITING HERE
