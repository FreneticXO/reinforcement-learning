"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the AlgorithmManyArms class. Here are the method details:
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
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
import math
# END EDITING HERE

class AlgorithmManyArms:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.clicks = np.zeros(num_arms)
        self.means = np.zeros(num_arms)
        self.success = np.zeros(num_arms)
        # Horizon is same as number of arms
    
    def give_pull(self):
        # START EDITING HERE

        for i in range(int(math.sqrt(self.num_arms))) : 
            self.means[i] = np.random.beta(1+self.success[i], 1+self.clicks[i]-self.success[i], size = None)
        
        return np.argmax(self.means)

        

        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE

        self.clicks[arm_index] += 1
        if reward == 1:
            self.success[arm_index] += 1

        # END EDITING HERE
