#!/usr/bin/env python
#
#  Copyright (c) 2015 MIT. All rights reserved.
#
#   author: Pedro Santana
#   e-mail: psantana@mit.edu
#   website: people.csail.mit.edu/psantana
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#  1. Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#  3. Neither the name(s) of the copyright holders nor the names of its
#     contributors or of the Massachusetts Institute of Technology may be
#     used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
#  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
#  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
"""
RAO*: a risk-aware planner for POMDP's.

Module that handles belief states.

@author: Pedro Santana (psantana@mit.edu).
"""
from .rao_misc import hash_complex_type,limit_precision
import numpy as np

class BeliefState(object):
    """
    Class representing a discrete belief state.
    """
    def __init__(self,belief_dict,decimals=5):
        self.belief = limit_precision(belief_dict,decimals)
        self.hash_key = hash_complex_type(belief_dict)

    @property
    def belief(self):
        """Dictionary representing the belief state"""
        return self._belief_dict

    @belief.setter
    def belief(self,new_belief):
        if isinstance(new_belief,dict):
            self._belief_dict = new_belief
        else:
            raise TypeError('Belief states should be given in dictionary form.')

    @property
    def particle_prob_tuples(self):
        """List of particles and their associated probabilities"""
        return list(self.belief.values())

    @property
    def particles(self):
        """List of particles in the belief state."""
        return [p_tup[0] for p_tup in self.particle_prob_tuples]

    @property
    def probabilities(self):
        """List of probabilities in the belief state."""
        return [p_tup[1] for p_tup in self.particle_prob_tuples]

    @property
    def entropy(self):
        """Entropy associated with the belief state."""
        return np.sum([(-p*np.log2(p) if not np.isclose(p,0.0) else 0.0) for p in self.probabilities])

    def __hash__(self):
        """
        Returns the elements hash key.
        """
        return self.hash_key

    def __eq__(x,y):
        """
        Two elements are considered equal if they have the same hash key.
        """
        return isinstance(x,BeliefState) and isinstance(y,BeliefState) and (x.hash_key == y.hash_key)

def avg_func(belief,func,*args):
    """Averages the output of a function over a belief state."""
    avg_value = 0.0
    for str_state,(state,prob) in belief.items():
        #Applies function on a state, with any number of supplied arguments
        #after that, and averages with the probability.
        avg_value+= func(state,*args)*prob
    return avg_value

def bound_prob(prob):
    """Ensures that a probability value is within [0.0,1.0]"""
    #valid_prob = min(1.0,max(0.0,prob))
    # if valid_prob != prob:
    #     print("WARNING: received invalid probability value %f"%(prob))
    return min(1.0,max(0.0,prob))

def blf_indicator(op_type,belief,ind_func,*args):
    """Applies an indicator function to a belief, with the option of
    stopping at the first True or False."""
    if op_type == 'count': #How many times it returns true
        count = 0
        for str_state,(state,prob) in belief.items():
            count+= ind_func(state,*args)
        return count
    elif op_type == 'prob':#Probability of being true
        prob = 0.0
        for str_state,(state,prob) in belief.items():
            prob+= ind_func(state,*args)*prob
        return prob
    elif op_type == 'has_true': #Contains true
        for str_state,(state,prob) in belief.items():
            if ind_func(state,*args):
                return True
        return False
    elif op_type == 'has_false': #Contains false
        for str_state,(state,prob) in belief.items():
            if not ind_func(state,*args):
                return True
        return False

def is_terminal_belief(blf_state,term_fun,terminal_prob):
    """Determines if a given belief state corresponds to a terminal node,
    according to the stopping criterion."""
     #Evaluates the terminal indicator function over the whole belief state,
    #returning True only is the belief has no nonterminal state.
    if terminal_prob==1.0:
        return not blf_indicator('has_false',blf_state,term_fun)
    #Flags a state as being terminal if a large percentage of its particles
    #are terminal states.
    else:
        return blf_indicator('prob',blf_state,term_fun)>= terminal_prob

def predict_belief(belief,T,r,act,hash_state):
    """
    Propagates a belief state forward according to the state transition
    model. Also computes the *safe predicted belief*, i.e., the predicted
    belief coming from particles in non-constraint-violating paths.
    """
    pred_belief={}
    pred_belief_safe={}; sum_safe=0.0
    #For every particle of the current belief
    for str_state,(particle_state,particle_prob) in belief.items():

        if np.isclose(r(particle_state),0.0): #Safe belief state
            safe_state=True
            sum_safe+=particle_prob
        else:
            safe_state=False

        #For every possible next state (with some probability)
        for next_state,trans_prob in T(particle_state,act):
            #Probability for the next state
            next_prob = particle_prob*trans_prob

            #Ensures that impossible transitions do not 'pollute' the belief
            #with 0 probability particles.
            if next_prob > 0.0:
                str_next_state = hash_state(next_state)
                if str_next_state in pred_belief:
                    pred_belief[str_next_state][1] += next_prob
                else:
                    pred_belief[str_next_state] = [next_state,next_prob]

                if safe_state: #Safe belief state
                    if str_next_state in pred_belief_safe:
                        pred_belief_safe[str_next_state][1] += next_prob
                    else:
                        pred_belief_safe[str_next_state] = [next_state,next_prob]

    if sum_safe>0.0: #Not all particles are on violating paths
        #Normalizes the safe predicted belief
        for str_next_state,b_tuple in pred_belief_safe.items():
            pred_belief_safe[str_next_state][1]/=sum_safe

    return pred_belief,pred_belief_safe

def compute_observation_distribution(pred_belief,pred_belief_safe,O,hash_state):
    """Computes the probability of getting an observation, given some
    predicted belief state."""
    obs_distribution={} #Prob. distrib. of observations
    obs_distribution_safe={} #Prob. distrib. of 'safe' observations
    state_to_obs={} #Mapping from state to possible observations (used
                    #later as likelihood function in belief state update)

    beliefs = [pred_belief,pred_belief_safe]
    distribs = [obs_distribution,obs_distribution_safe]
    sum_probs= [0.0,0.0]

    #Iterates over the different belief and corresponding distributions
    for i,(belief,distrib) in enumerate(zip(beliefs,distribs)):

        #For every particle in the current predicted belief
        for str_state,(particle_state,particle_prob) in belief.items():

            #Ensures that 0 probability particles do not 'pollute' the
            #likelihood function.
            if particle_prob>0.0:
                if i==0:
                    state_to_obs[hash_state(particle_state)]=[]

                #For every possible observation (with some probability)
                for obs,obs_prob in O(particle_state):

                    #Ensures that impossible observations do not 'pollute' the
                    #likelihood with 0 probability observations.
                    if obs_prob>0.0:
                        if i==0:
                            state_to_obs[hash_state(particle_state)].append([obs,obs_prob])

                        str_obs = hash_state(obs)
                        if str_obs not in distrib:
                            distrib[str_obs] = [obs,obs_prob*particle_prob]
                        else:
                            distrib[str_obs][1] += obs_prob*particle_prob

                        #Accumulates the probabilities
                        sum_probs[i]+=obs_prob*particle_prob

    # #FIXME: this was added because RAO* was returning unormalized observation
    # #distributions if I didn't perform a deep copy of the belief state. I'm
    # #removing this normalization to see if going back to the deep copy resolves
    # #the issue by itself.

    # #Normalizes the probability distributions
    # for i,d in enumerate(distribs):
    #     for str_obs,obs_tuple in d.items():
    #         obs_tuple[1]/=sum_probs[i]

    return obs_distribution,obs_distribution_safe,state_to_obs

def update_belief(pred_belief,state_to_obs,obs):
    """Performs belief state update."""

    # post_belief = cp.deepcopy(pred_belief) #Creates a deep copy (was working before)
    # post_belief = cp.copy(pred_belief) #Creates a shallow copy of the belief (seems to break things)
    post_belief = copy_belief(pred_belief) #Does not copy the state objects.

    #For every particle in the current belief
    prob_sum = 0.0; zero_prob_states=[]
    for str_state,b_tuple in post_belief.items():
        #Checks if obs is a possible observation (nonzero likelihood)
        found_obs = False
        for possible_obs,obs_prob in state_to_obs[str_state]:
            if possible_obs == obs: #obs is a possible observation
                b_tuple[1] *= obs_prob #Likelihood
                prob_sum += b_tuple[1] #Prob. sum
                found_obs=True
                break
        #If obs was not found, that particle is removed (zero probability)
        if not found_obs:
            zero_prob_states.append(str_state)

    #Removes zero probability particles
    for str_state in zero_prob_states:
        del post_belief[str_state]

    #Normalizes the probabilities
    if prob_sum>0.0:
        for str_state,b_tuple in post_belief.items():
            b_tuple[1] /= prob_sum

    return post_belief

def copy_belief(belief):
    """
    Copies the necessary elements that compose a belief state
    """
    return {k:[v[0],v[1]] for k,v in belief.items()}
