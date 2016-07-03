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

Class representing the new version of the Boeing demo, which is exactly like the
Mitsubishi one. The only difference here is that, instead of reading the operator
descriptions directly from the PDDL files, they come from the modeling using
RMPyL constructs similar to the new version of RMPL-J.

@author: Pedro Santana (psantana@mit.edu).
""" 
from collections import namedtuple
from . import models

def check_containment(predicates1,predicates2):
    """
    Checks if predicates1 is contained in predicates2.
    """
    for svar,val in predicates1.items():
        if (not svar in predicates2) or predicates2[svar]!=val:
            return False        
    return True     

def compute_difference(predicates1,predicates2):
    """
    Computes the dictionary difference predicates1-predicates2
    """
    diff={}
    for svar,val in predicates1.items():
        if (not svar in predicates2) or predicates2[svar]!=val:
            diff[svar]=val
    return diff


class EpisodeAction(namedtuple('RMPyLChooseAction',['episode'])):
    """
    Simple class representing an episode action.
    """    
    __slots__= ()

    def __repr__(self):        
        return self.episode.action


class GenericManipulation(object):
    """
    Class encapsulating a generic manipulation and soldering problem, so that
    it
    """
    def __init__(self,objects,manipulators,locations,goal):               
        self.objects = objects
        self.manipulators = manipulators
        self.locations = locations
        self.goal=goal

    def applicable_activities(self,predicates):
        """
        Returns a list of applicable episodes, given a dictionary of predicates.
        """
        applicable=[]
        for manip in self.manipulators:
            for ob in self.objects: 
                for loc in self.locations:
                    activity = manip.pick(ob,loc,predicates)
                    if activity!=None:
                        applicable.append(activity)
                    activity = manip.place(ob,loc,predicates)
                    if activity!=None:
                        applicable.append(activity)
                    activity = manip.clean(ob,loc,predicates)
                    if activity!=None:
                        applicable.append(activity)   
                    for other_ob in self.objects: 
                        activity = manip.solder(ob,other_ob,loc,predicates)
                        if activity!=None:
                            applicable.append(activity) 
        return applicable    

    def goal_reached(self,predicates):
        """
        Returns if the goal has been reached.
        """
        return check_containment(self.goal,predicates)

    def remaining_subgoals(self,predicates):
        """
        Returns a dictionary of remaining predicates to be achieved.
        """
        return compute_difference(self.goal,predicates)


class BoeingRMPyLModel(models.CCHyperGraphModel):
    """
    Class representing an RMPyL program as a (PO)MDP.
    """
    def __init__(self,gen_manip,verbose=0):
        super(BoeingRMPyLModel,self).__init__()
       
        self.verbose = verbose
        self.is_maximization = False #Tries to minimize some measure of cost
        self.immutable_actions = False #Available actions depend on state
        self.gen_manip = gen_manip
                                
    def get_state(self,true_predicates):
        """
        Returns a proper state representation.
        """
        state_dict = {'true_predicates':true_predicates}
        return state_dict              

    def get_initial_belief(self,initial_state):
        """
        Proper initial representation of the initial belief state of the search.
        """        
        belief = {}
        s0 = self.get_state(initial_state)        
        belief[self.hash_state(s0)] = [s0,1.0]
        return belief
        
    def state_transitions(self,state,action):
        """
        Returns the next state, after executing an operator.
        """     
        new_pred = dict(state['true_predicates'])
        new_pred.update(action.episode.end_effects)
        next_state = self.get_state(new_pred)                          
        return [[next_state,1.0]]
           
    def actions(self,state):
        """
        Actions available at a state.
        """               
        if not self.is_terminal(state):             
            actions=[EpisodeAction(episode=ep) for ep in self.gen_manip.applicable_activities(state['true_predicates'])]            
            return actions
        else:
            return []                
                               
    def is_terminal(self,state):
        """A state is terminal if it has reached the goal."""        
        return self.gen_manip.goal_reached(state['true_predicates'])

    def value(self,state,action):
        """
        Small penalty for making humans do things.
        """    
        return 1.0

    def terminal_value(self,state):
        """
        Final value of a terminal state.
        """
        if self.gen_manip.goal_reached(state['true_predicates']):
            return 0.0 #No cost for goal state
        else:
            return 100.0 #High cost for goal state.

    def heuristic(self,state):
        """
        Heuristic estimate of the expected value associated with a state.
        """                
        return len(self.gen_manip.remaining_subgoals(state['true_predicates']))
        

    def state_risk(self,state):
        """
        There are no constraints to be violated, so no risk.
        """        
        return 0.0

    def execution_risk_heuristic(self,state):
        """
        There are no constraints to be violated, so no risk.
        """
        return 0.0

    def observations(self,state):
        """
        For a fully observable model, generates a unique observation per state.
        """
        meas = state['observation'] if 'observation' in state else '_state_'
        return [[meas,1.0]] 

    def obs_repr(self,observation):
        """
        Observation are represented by their own strings.
        """
        return observation


    
    
