"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

You need to complete the following methods:
    - give_pull(self): This method is called when the algorithm needs to
        select the arms to pull for the next round. The method should return
        two arrays: the first array should contain the indices of the arms
        that need to be pulled, and the second array should contain how many
        times each arm needs to be pulled. For example, if the method returns
        ([0, 1], [2, 3]), then the first arm should be pulled 2 times, and the
        second arm should be pulled 3 times. Note that the sum of values in
        the second array should be equal to the batch size of the bandit.
    
    - get_reward(self, arm_rewards): This method is called just after the
        give_pull method. The method should update the algorithm's internal
        state based on the rewards that were received. arm_rewards is a dictionary
        from arm_indices to a list of rewards received. For example, if the
        give_pull method returned ([0, 1], [2, 3]), then arm_rewards will be
        {0: [r1, r2], 1: [r3, r4, r5]}. (r1 to r5 are each either 0 or 1.)
"""

import numpy as np

# START EDITING HERE
import math

# You can use this space to define any helper functions that you need.
# END EDITING HERE

class AlgorithmBatched:
    def __init__(self, num_arms, horizon, batch_size):
        self.num_arms = num_arms
        self.horizon = horizon
        self.batch_size = batch_size
        assert self.horizon % self.batch_size == 0, "Horizon must be a multiple of batch size"
        # START EDITING HERE
        # Add any other variables you need here

        self.explore = 0
        self.eps = math.sqrt(horizon)
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.idx = 0

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE

        default = np.arange(self.num_arms)

        def_pull = np.zeros(self.num_arms)
        for i in range(self.batch_size):
            def_pull[self.idx % self.num_arms] += 1
            self.idx += 1

        pull = def_pull.astype(int)
        #return [4], [self.batch_size]
        if self.explore < self.eps:
            self.explore += self.batch_size
            return default, pull
        else:
            return [int(np.argmax(self.values))], [self.batch_size]
        
        
        # END EDITING HERE
    
    def get_reward(self, arm_rewards):
        # START EDITING HERE
        
        for key in arm_rewards:
            for reward in arm_rewards[key]:
                self.counts[key] += 1
                n = self.counts[key]
                value = self.values[key]
                new_value = ((n - 1) / n) * value + (1 / n) * reward
                self.values[key] = new_value


        # END EDITING HERE