"""
Template for implementing QLearner  (c) 2015 Tucker Balch

Code implemented by Erika Gemzer, gth659q, Summer 2018
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        """
        The constructor QLearner() reserves space and initializes the Q table Q[s, a]
        with enough space for all actions and states. It initializes ndarray Q[] with zeros.

        Inputs / Parameters:
            num_states: int, the number of states to consider
            num_actions: int, the number of actions available
            alpha: float, the learning rate used in the update rule.
                Should range between 0.0 and 1.0 with 0.2 as a typical value
            gamma: float, the discount rate used in the update rule.
                Should range between 0.0 and 1.0 with 0.9 as a typical value.
            rar: float, random action rate. The probability of selecting a random action
                at each step. Should range between 0.0 (no random actions) to 1.0
                (always random action) with 0.5 as a typical value.
            radr: float, random action decay rate, after each update, rar = rar * radr.
               Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
            dyna: int, conduct this number of dyna updates for each regular update.
                When Dyna is used, 200 is a typical value.
            verbose: boolean, if True, your class is allowed to print debugging
                statements, if False, all printing is prohibited.
        """

        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        #initialize the state(s) and action(a) to 0
        self.s = 0
        self.a = 0

        #initialize Q table to store the reward associated with each action and each state
        self.Q = np.zeros(shape=(self.num_states, self.num_actions))

        #Dyna-Q requires establishing Transition and Reward functions

        #initialize a dictionary for T. T is the probability that in state S, taking action A, we enter new state S
        self.T = {} #Initialize T dict. T will store as such: {(s, a):{s_prime:count(s,a)}
        # R is the probability that in state S we take action A and results in reward R
        self.R = np.zeros(shape=(self.num_states, self.num_actions)) #initialize R as same shape as Qtable

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
            ie when setting initial state and when using a learned policy
        @param s: The new state
        @returns: The selected action
        """

        #set the action to take based on the random action rate if the action should be random based on random action rate parameter from the constructor
        if rand.uniform(0.0,1.0) < self.rar: #the Qlearner is exploring (random action)
            action = rand.randint(0, self.num_actions - 1)
        else: #the Qlearner is evaluating using available values (not random action). Pick the index with the maximum value
            action = self.Q[s,:].argmax()
        self.s = s # store the state
        self.a = action # store the action
        if self.verbose: print "s =", s,"a =",action
        return action


    def query(self,s_prime,r):
        """
        @summary: Use the Update rule to determine the next action to take based on state s_prime.
            Update the Q table and return an action.
        @param s_prime: The new state, an integer
        @param r: The (immediate) reward, a float
        @returns: The selected action to take in s_prime, an integer
        """

        # Non-Dyna-Q: Update the Q value of the latest state and action based on s_prime and r
        self.Q[self.s, self.a] = (1 - self.alpha) * (self.Q[self.s, self.a]) + self.alpha \
                                 * (r + self.gamma * self.Q[s_prime, self.Q[s_prime, :].argmax()])

        #Dyna-Q hallucinations
        if self.dyna > 0: #if dyna hallucinations are turned on

            # first, update the table for the reward model
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r

            # Check to see if the current pair of initial state and action are in the Ttable
            if (self.s, self.a) in self.T: #we've seen this initial state + action pair before
                if s_prime in self.T[(self.s, self.a)]: # and we've see this resulting state s_prime
                    self.T[(self.s, self.a)][s_prime] += 1 # count increases
                else: #this is a new resulting state s_prime for the s, a pair
                    self.T[(self.s, self.a)][s_prime] = 1 #add the first count for this resultant state s_prime
            else: # this is a new pair of initial state and action
                self.T[(self.s, self.a)] = {s_prime: 1} # set the s_prime and count(s,a) to map to the dict key (s,a)

            #infer the new s_prime from T model
            Q = self.Q.copy()
            for i in range(self.dyna):
                s = rand.randint(0, self.num_states - 1) # pick random state s from available states
                a = rand.randint(0, self.num_actions - 1) # pick random action a from available actions

                if (s, a) in self.T:
                    #Find the most common s_prime as a result of taking action a in state s
                    best_s_prime = max(self.T[(s, a)], key=lambda k: self.T[(s, a)][k])
                    # Update the temporary Q table
                    Q[s, a] = (1 - self.alpha) * Q[s, a] + self.alpha * (self.R[s, a] \
                                    + self.gamma * Q[best_s_prime, Q[best_s_prime,:].argmax()])
                    # Update the real Q table of the learner once Dyna-Q is complete
                    self.Q = Q.copy()

        #Determine the next step to take (dyna-Q on or off), update the state and action
        new_action = self.querysetstate(s_prime)
        self.rar = self.rar * self.rar
        if self.verbose: print "s =", s_prime,"a =", new_action,"r =",r
        return new_action


    def author(self):
        return 'gth659q'  # my Georgia Tech username.


if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
