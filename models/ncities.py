#!/usr/bin/env python
#
#  Copyright (c) 2014 MIT. All rights reserved.
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

Class representing the NCITIES benchmark problem from [Williams et al., 2005]
'Factored Partially Observable Markov Decision Processes for Dialogue Management'

@author: Pedro Santana (psantana@mit.edu).
""" 
from rao.models import models

import numpy as np
import copy as cp
import time
from collections import deque
import itertools as it

class NCities(models.CCHyperGraphModel):
    """
    Class representing a dialogue system system for the NCITIES flight
    booking benchmark [Williams et al., 2005].
    """
    def __init__(self,time_limit=-1,cities=['A','B','C'],perr=0.15,
                 simplified_obs=False,verbose=0):
        super(NCities,self).__init__()
       
        self._verbose = verbose
        self.is_maximization = True #Cost minimization
        self.immutable_actions = False

        if simplified_obs:
            self.observations = self._observations_simplified
        else:
            self.observations = self._observations_original

        #Prob. of error when interpreting user input                                              
        if (perr>=0.0 and perr<=1.0):
            self._perr = perr
        else: 
            raise TypeError("Error probability perr should be within [0,1].")

        self._time_limit = time_limit        
        if verbose:
            if time_limit>=0:
                print("Maximum number of plan steps: %d"%(time_limit))
            else:                
                print("Unlimited number of steps.")

        self._cities = cities                
        city_pairs = self._all_city_pairs(self._cities)
                            
        #Machine actions
        self._actions = [('greet',),('ask-from',),('ask-to',),('fail',)]
        self._actions += [('conf-to',c) for c in cities]
        self._actions += [('conf-from',c) for c in cities]
        self._actions += [('submit',src,dest) for src,dest in city_pairs]
                
        #User actions (observed during dialogue)
        self._user_actions = [('yes',),('no',),('null',)]                
        self._user_actions += [ (c,) for c in cities]
        self._user_actions += [('from',c) for c in cities]
        self._user_actions += [('to',c) for c in cities]
        self._user_actions += [('from-to',src,dest) for src,dest in city_pairs]
                 
    def get_ncities_state(self,user_goal,user_action,dialogue_state,failed,
                            submitted,time):
        """
        Returns a proper state representation.
        """        
        state_dict = {  'user_goal':user_goal, #User's desired itinerary
                        'user_action':user_action, #User's input to the system
                        'dialogue_state':dialogue_state, #Dialogue state
                        'failed':failed, #Whether the 'fail' action was executed                        
                        'submitted':submitted,#Whether a purchase was submitted
                        'time':time}     #Time instant                  
        return state_dict

    def get_initial_belief(self):
        """
        Builds belief state (list of pairs (state, prob))
        """        
        city_pairs = self._all_city_pairs(self._cities)
        b0 = {}
        #Uniform distribution over all possible destinations
        for src,dest in city_pairs:            
            state = self.get_ncities_state(user_goal=('from-to',src,dest), #User's true itinerary
                                           user_action=('null',), #Last user input
                                           dialogue_state=('n','n',0), #State of dialogue
                                           failed=False, #Failed booking
                                           submitted=(), #Submitted booking
                                           time=0)       #Time step                                  
            b0[self.hash_state(state)] = [state,1.0/len(city_pairs)]                
        return b0
      
    def value(self,state,action):
        """Value of a nonterminal state"""
        if (action[0]=='conf-from' and state['dialogue_state'][0]=='n'):
            return -3.0 #Confirming 'from' before the user mentioning it
        elif (action[0]=='conf-to' and state['dialogue_state'][1]=='n'):
            return -3.0 #Confirming 'to' before the user mentioning it
        else:
            return -1.0 #Penalty for keeping interacting with the user
 
    def terminal_value(self,state):
        """Value associated with a terminal state."""        
        if len(state['submitted'])>0:
            if state['submitted'] == state['user_goal']:
                return 10.0 #High reward submitting correct ticket
            else:
                return -10.0 #High cost for submitting wrong ticket.

        if state['failed']: #Failed to find a booking
            return -5.0

        if state['time']>=self._time_limit:#Went over time limit
            return -7.0
                                                         
    def heuristic(self,state):
        """Heuristic value of a state.""" 
        if self.is_terminal(state):
            return self.terminal_value(state) #Exact value when terminal
        else: #Incomplete booking            
            ds_st = ['c','u','n']
            ds = state['dialogue_state']
            #Optimist number of steps to confirm a booking
            optim_steps_to_conf = ds_st.index(ds[0])+ds_st.index(ds[1])+(1-ds[2])
            
            if (optim_steps_to_conf+1) > (self._time_limit-state['time']):
                return -5.0
            else:
                return 10.0-optim_steps_to_conf
            # if (optim_steps_to_conf+1) > (self._time_limit-state['time']):
            #     return 0.0
            # else:
            #     return 10.0-optim_steps_to_conf
            
        
    def state_risk(self,state):
        """
        Risk of booking the wrong flight.
        """        
        return 1.0 if (len(state['submitted'])>0)and(state['submitted']!=state['user_goal']) else 0.0

    def execution_risk_heuristic(self,state): 
        """Heuristic that estimates the risk of execution of a plan from a given
        state."""        
        return self.state_risk(state)                 
    
    def _user_goal_model(self,state,action):
        """
        Models an unchanging user goal.
        """
        return [(state['user_goal'],1.0)]        

    def _user_response_model(self,user_goal,action):        
        """
        Models how the user responds (provides user action) to machine actions.
        The output is a list of possible user actions and their associated
        probabilities. We assume that the user is always trying to be helpful, 
        but might not respond (same assumption as in [Williams et al., 2005]).
        """           
        if action[0]=='ask-from':   
            return [[(user_goal[1],),0.6],        # x 
                    [('from',user_goal[1]),0.3],  # from-x
                    [user_goal,0.09],             # from-x-to-y
                    [('null',),0.01]]             # no response

        elif action[0]=='ask-to':       
            return [[(user_goal[2],),0.6],      # y 
                    [('to',user_goal[2]),0.3],  # to-y
                    [user_goal,0.09],           # from-x-to-y
                    [('null',),0.01]]           # no response

        elif action[0]=='greet':       
            return [[('to',user_goal[2]),0.33],   # to-y
                    [('from',user_goal[1]),0.33], # from-x
                    [user_goal,0.33],             # from-x-to-y
                    [('null',),0.01]]             # no response

        elif action[0]=='conf-from':       
            yes_no = 'yes' if user_goal[1] == action[1] else 'no'
            return [[(user_goal[1],),0.19],       # x 
                    [('from',user_goal[1]),0.1],  # from-x
                    [(yes_no,),0.7],              # yes or no                                        
                    [('null',),0.01]]             # no response

        elif action[0]=='conf-to':
            yes_no = 'yes' if user_goal[2] == action[1] else 'no'                   
            return [[(user_goal[2],),0.19],       # y 
                    [('to',user_goal[2]),0.1],    # to-y
                    [(yes_no,),0.7],              # yes or no
                    [('null',),0.01]]             # no response
       
        elif (action[0]=='submit') or (action[0]=='fail'):
            return[[('null',),1.0]] #Action requires no input from user

        else:
            raise TypeError('Invalid machine action %s.'%(action[0]))            
                
    def _joint_user_goal_action_model(self,user_goal_list,action):        
        """
        Models the joint distribution of user goals and actions, given a machine
        action.
        """
        user_goal_action_list = []
        for user_goal, p_ug in user_goal_list:            
            user_actions = self._user_response_model(user_goal,action)

            for user_action, p_ua in user_actions:
                joint = (user_goal,user_action,p_ug*p_ua)            
                user_goal_action_list.append(joint)                          

        return user_goal_action_list


    def _dialog_evolution_model(self,user_action,dialog_state,action): 
        """
        Deterministic implementation of a dialogue system where the user chooses
        an itinerary and confirms it. We assume that the user is always trying
        to be helpful, but might not respond (same assumption as in 
        [Williams et al., 2005]).
        """ 
        #Only updates the dialog count (prevents a 'greet' loop)
        new_dialog_state = [dialog_state[0],dialog_state[1],1]
        
        if user_action[0]!='null':            
            if (user_action[0]=='from')or(action[0]in['ask-from','conf-from'] and user_action[0]!='no'):
                new_dialog_state[0] = 'u' if dialog_state[0]=='n' else 'c'

            #Changed value of 'to' field
            elif (user_action[0]=='to')or(action[0]in['ask-to','conf-to'] and user_action[0]!='no'):                
                new_dialog_state[1] = 'u' if dialog_state[1]=='n' else 'c'

            #Both 'from' and 'to' are changed                
            elif (user_action[0]=='from-to'):
                new_dialog_state[0] = 'u' if dialog_state[0]=='n' else 'c'
                new_dialog_state[1] = 'u' if dialog_state[1]=='n' else 'c'            
                        
        return tuple(new_dialog_state)

    def _joint_user_goal_action_dialog_model(self,user_goal_action_list,dialog_state,action):        
        """
        Models how the joint distribution of the whole dialogue system.
        """
        user_goal_action_dialog_list = []

        for user_goal,user_action, p_uga in user_goal_action_list:            
            new_dialog_state = self._dialog_evolution_model(user_action,
                                                            dialog_state,
                                                            action)
    
            joint = (user_goal,user_action,new_dialog_state,p_uga)            
            user_goal_action_dialog_list.append(joint)                          

        return user_goal_action_dialog_list

    def state_transitions(self,state,action):
        """Given a network state, returns the result of simulating the power network 
        as tuple of the possible next states, given an action. Since we are 
        assuming deterministic circuit-breakers, it will correspond to a single 
        simulation of the network."""
        
        failed = True if action[0] == 'fail' else False
        submitted = action if action[0] == 'submit' else ()
        
        ug_list = self._user_goal_model(state,action)
        uga_list = self._joint_user_goal_action_model(ug_list,action)
        ugad_list=self._joint_user_goal_action_dialog_model(uga_list,
                                                            state['dialogue_state'],
                                                            action)
        next_state_list = []
        for ug,ua,ds,p_ugad in ugad_list:            
            next_state = self.get_ncities_state(user_goal=ug,
                                                user_action=ua,
                                                dialogue_state=ds,
                                                failed=failed,
                                                submitted=submitted, 
                                                time=state['time']+1)
            next_state_list.append([next_state,p_ugad])        
                                          
        return next_state_list
    
    
    def observations(self,state):
        """ 
        Function pointer redefined at the constructor, depending on whether we
        choose the original observation model or the simplified version.
        """        
        pass        

    def _observations_simplified(self,state):
        """
        Reduced observation model with just correct or incorrect recognition.
        """
        if self._perr>0.0: #Observation error            
            meas_list =[[state['user_action'],1.0-self._perr],
                        [('inconclusive',),self._perr]]        
        else: #Perfect observation
            meas_list =[[state['user_action'],1.0]]

        return meas_list 

    def _observations_original(self,state):
        """
        Observations correspond to what we thing the user input was. Could I
        reduce this model to take into account just two observations (correct
        and incorrect)?.
        """
        if self._perr>0.0: #Observation error
            pc = 1.0-self._perr #Observe correct input from user
            pw = self._perr/(len(self._user_actions)-1.0) #Observe wrong input

            meas_list=[]
            for user_action in self._user_actions:            
                prob = pc if user_action == state['user_action'] else pw 
                meas_list.append([user_action,prob])
        
        else: #Perfect observation
            meas_list =[[state['user_action'],1.0]]

        return meas_list 
    
       
    def actions(self,state):
        """Available actions at some state."""
        if self.is_terminal(state):
            return [] #No actions available at terminal states
        else:
            if state['dialogue_state'][2]>0: #No the first dialogue action
                return self._actions[1:] #All actions, but 'greet'
            else: #Initial time step
                return [self._actions[0]] #'greet'            
    
    def is_terminal(self,state):
        """
        Signals a terminal state.
        """        
        if (state['time']>=self._time_limit) or state['failed'] or len(state['submitted'])>0:            
            return True
        else:                        
            return False

    def _all_city_pairs(self,cities):
        """
        Returns all pairs of origin-destination cities.
        """            
        pairs=[]
        for i in range(len(cities)-1):
            for j in range(i+1,len(cities)):
                pairs.append([cities[i],cities[j]])
                pairs.append([cities[j],cities[i]])
        return pairs            

    