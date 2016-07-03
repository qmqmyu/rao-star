#!/usr/bin/env python
#
#  Copyright (c) 2016 MIT. All rights reserved.
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

This module contains classes that allow RAO* and PARIS to solve the RSS demo
scenario described in PDDL with durative actions. It was originally intended to
be solved by a combination of tBurton and Picard.

@author: Pedro Santana (psantana@mit.edu).
"""
from .pddlmodel import DurativePDDL
from rmpyl.constraints import TemporalConstraint

class RSSDurativePDDL(DurativePDDL):
        """
        Class encapsulating the planning model used for the RSS demo presented on
        March 18th, 2016. It consists of PDDL model with durative actions that must
        be used to generate an execution policy for a Mars rover operating under
        uncertainty.
        """
        def __init__(self,domain_file,prob_file,pddl_pickle='',maximization=False,
                     perform_scheduling=True,duration_func=None,time_window_func=None,
                     paris_params={},max_steps=50,verbose=0):
            #Initializes the DurativePDDL model
            super(RSSDurativePDDL,self).__init__(domain_file,prob_file,pddl_pickle,
                                                 maximization,perform_scheduling,
                                                 duration_func,paris_params,
                                                 max_steps,verbose)
            #Adds the time window function
            self.time_window_func = time_window_func


        def state_transitions(self,state,action):
            """
            Adds time windows to the state transition function.
            """
            #If this node has been delayed, expands it.
            if action == '__expand__':

                #The only states that can be expanded are delayed states.
                if not state['delayed']:
                    import ipdb; ipdb.set_trace()
                    pass

                next_min_length,next_op_actions = self._shortest_length_and_actions(state['true_predicates'],
                                                                                    self.task.goals,
                                                                                    self.max_steps)
                #Returns a non-delayed copy of the current state with updated minimum
                #PDDL plan length and optimal actions.
                next_state = self.get_state(true_predicates = state['true_predicates'],
                                            last_event = state['last_event'],
                                            constraints = state['constraints'],
                                            optimal_next_actions = next_op_actions,
                                            min_pddl_length = next_min_length,
                                            delayed=False)
            else:

                #DEBUG: states with no optimal actions to be executed should be deemed
                #terminal and should never be dequed. Also, delayed states can only
                #be expanded
                if state['optimal_next_actions'] in [None,[]] or state['delayed']:
                    import ipdb; ipdb.set_trace()
                    pass

                #Applies PDDL operator to true predicates
                new_pred = action.pddl_operator.apply(state['true_predicates'])

                #Temporal constraint representing the duration of the activity
                constraints = [action.duration]

                #Temporal constraint representing the fact that this action comes after
                #the previous one
                constraints.append(TemporalConstraint(start=state['last_event'],end=action.start,
                                              ctype='controllable',lb=0.0,ub=float('inf')))

                #Temporal constraint representing the fact that this activity should end
                #before the end of the mission
                constraints.append(TemporalConstraint(start=action.end,end=self.global_end_event,
                                              ctype='controllable',lb=0.0,ub=float('inf')))

                #Verifies if there is a time window associated with this operator
                if self.time_window_func != None:
                    tw_ret = self.time_window_func(action.action)
                    constraints+=time_window_constraints(tw_ret[1],tw_ret[0],
                                                          self.global_start_event,
                                                          action)

                #If the current action is in the set of optimal next actions, do not
                #delay the expansion of the next state, for it is on a promising path
                if action.pddl_operator.name in state['optimal_next_actions']:
                    delayed=False
                    next_min_length = state['min_pddl_length']-1
                    next_op_actions = self._pa.first_actions(new_pred,
                                                             self.task.goals,
                                                             next_min_length)
                #However, if this is not an optimal action at the current state,
                #refrain from expanding the next state until strictly necessary.
                else:
                    delayed=True
                    next_min_length = state['min_pddl_length']
                    next_op_actions = None

                next_state = self.get_state(true_predicates = new_pred,
                                            last_event = action.end,
                                            constraints = state['constraints']+constraints,
                                            optimal_next_actions = next_op_actions,
                                            min_pddl_length = next_min_length,
                                            delayed=delayed)

                return [[next_state,1.0]]

def time_window_constraints(tw_type,tw_interval,start_of_time,action):
    """
    Returns the temporal constraints that ensure that an action is executed
    within a time window.
    """
    if tw_type == 'at-start' or tw_type == 'at-end':
        #Action start completely contained within the time window
        act_event = action.start if tw_type == 'at-start' else action.end
        return [TemporalConstraint(start=start_of_time,
                                   end=act_event,
                                   ctype='controllable',
                                   lb=tw_interval[0],
                                   ub=tw_interval[1])]
    elif tw_type == 'overall':
        #BothAction end completely contained within the time window
        at_start = time_window_constraints('at-start',tw_interval,start_of_time,action)
        at_end = time_window_constraints('at-end',tw_interval,start_of_time,action)
        return at_start+at_end
    else:
        raise ValueError('Invalid type of time window: %s'%(tw_type))


# from .pddlmodel import AbstractCCPDDL,PDDLEpisode
# from rmpyl.constraints import TemporalConstraint
# from pytemporal.paris import PARIS
# from rao.pddl.heuristics import PySATPlanAnalyzer
#
# class RSSDurativePDDL(AbstractCCPDDL):
#     """
#     Class encapsulating the planning model used for the RSS demo presented on
#     March 18th, 2016. It consists of PDDL model with durative actions that must
#     be used to generate an execution policy for a Mars rover operating under
#     uncertainty.
#     """
#     def __init__(self,domain_file,prob_file,pddl_pickle='',maximization=False,
#                  perform_scheduling=True,duration_func=None,time_window_func=None,
#                  paris_params={},max_steps=50,verbose=0):
#         #Initializes the generic PDDL model
#         super(RSSDurativePDDL,self).__init__(domain_file,prob_file,pddl_pickle,maximization,verbose)
#
#         self.perform_scheduling = perform_scheduling
#         self.duration_func = duration_func
#         self.time_window_func = time_window_func
#
#         #Maximum number of actions allowed in any plan
#         self.max_steps = max_steps
#
#         #PDDL plan analyzer
#         self._pa = PySATPlanAnalyzer(domain_file,prob_file,precompute_steps=max_steps,
#                                      sequential=True,remove_static=True,
#                                      verbose=verbose>0)
#
#         if self.perform_scheduling:
#             self.scheduler = PARIS(**paris_params)
#
#     def get_state(self,true_predicates,last_event,constraints,
#                   optimal_next_actions,min_pddl_length,delayed):
#         """
#         Returns a proper state representation.
#         """
#         state_dict = {'true_predicates':frozenset(true_predicates),
#                       'last_event':last_event, #Temporal event associated with state
#                       'constraints':constraints,     #Temporal constraints
#                       'optimal_next_actions':optimal_next_actions, #Best actions (length-wise) to execute
#                       'min_pddl_length':min_pddl_length, #Minimum length of the relaxed PDDL plan.
#                       'delayed':delayed} #Whether the state is on an optimal PDDL plan
#
#         return state_dict
#
#     def get_initial_belief(self,constraints=[]):
#         """
#         Proper initial representation of the initial belief state of the search.
#         """
#         belief = {}
#
#         #Determines the minimum length and the optimal first actions for the relaxed PDDL plan.
#         min_length,op_next_actions = self._shortest_length_and_actions(self.task.initial_state,
#                                                                        self.task.goals,
#                                                                        self.max_steps)
#
#         #Temporal constraint that makes sure that the start and end events of a
#         #mission are correctly aligned.
#         start_before_end = TemporalConstraint(start=self.global_start_event,
#                                               end=self.global_end_event,
#                                               ctype='controllable',lb=0.0,ub=float('inf'))
#
#         s0 = self.get_state(true_predicates = self.task.initial_state,
#                             last_event = self.global_start_event,
#                             constraints = constraints+[start_before_end],
#                             optimal_next_actions = op_next_actions,
#                             min_pddl_length = min_length,
#                             delayed = False)
#
#         belief[self.hash_state(s0)] = [s0,1.0]
#         return belief
#
#     def actions(self,state):
#         """
#         Actions available at a state.
#         """
#         if not self.is_terminal(state):
#             #If this state has has its expansion delayed, return the expansion action
#             if state['delayed']:
#                 return ['__expand__']
#             #If not, returns all applicable operators.
#             else:
#                 true_pred = state['true_predicates'] #True predicates
#                 if self.duration_func == None:
#                     return [PDDLEpisode(op) for op in self.task.operators if op.applicable(true_pred)]
#                 else:
#                     return [PDDLEpisode(op,duration=self.duration_func(op.name)) for op in self.task.operators if op.applicable(true_pred)]
#         else:
#             return []
#
#     def state_transitions(self,state,action):
#         """
#         Returns the next state, after executing an operator.
#         """
#         #If this node has been delayed, expands it.
#         if action == '__expand__':
#
#             #The only states that can be expanded are delayed states.
#             if not state['delayed']:
#                 import ipdb; ipdb.set_trace()
#                 pass
#
#             next_min_length,next_op_actions = self._shortest_length_and_actions(state['true_predicates'],
#                                                                                 self.task.goals,
#                                                                                 self.max_steps)
#             #Returns a non-delayed copy of the current state with updated minimum
#             #PDDL plan length and optimal actions.
#             next_state = self.get_state(true_predicates = state['true_predicates'],
#                                         last_event = state['last_event'],
#                                         constraints = state['constraints'],
#                                         optimal_next_actions = next_op_actions,
#                                         min_pddl_length = next_min_length,
#                                         delayed=False)
#         else:
#
#             #DEBUG: states with no optimal actions to be executed should be deemed
#             #terminal and should never be dequed. Also, delayed states can only
#             #be expanded
#             if state['optimal_next_actions'] in [None,[]] or state['delayed']:
#                 import ipdb; ipdb.set_trace()
#                 pass
#
#             #Applies PDDL operator to true predicates
#             new_pred = action.pddl_operator.apply(state['true_predicates'])
#
#             #Temporal constraint representing the duration of the activity
#             constraints = [action.duration]
#
#             #Temporal constraint representing the fact that this action comes after
#             #the previous one
#             constraints.append(TemporalConstraint(start=state['last_event'],end=action.start,
#                                           ctype='controllable',lb=0.0,ub=float('inf')))
#
#             #Temporal constraint representing the fact that this activity should end
#             #before the end of the mission
#             constraints.append(TemporalConstraint(start=action.end,end=self.global_end_event,
#                                           ctype='controllable',lb=0.0,ub=float('inf')))
#
#             #Verifies if there is a time window associated with this operator
#             if self.time_window_func != None:
#                 tw_ret = self.time_window_func(action.action)
#                 constraints+=time_window_constraints(tw_ret[1],tw_ret[0],
#                                                       self.global_start_event,
#                                                       action)
#
#             #If the current action is in the set of optimal next actions, do not
#             #delay the expansion of the next state, for it is on a promising path
#             if action.pddl_operator.name in state['optimal_next_actions']:
#                 delayed=False
#                 next_min_length = state['min_pddl_length']-1
#                 next_op_actions = self._pa.first_actions(new_pred,
#                                                          self.task.goals,
#                                                          next_min_length)
#             #However, if this is not an optimal action at the current state,
#             #refrain from expanding the next state until strictly necessary.
#             else:
#                 delayed=True
#                 next_min_length = state['min_pddl_length']
#                 next_op_actions = None
#
#             next_state = self.get_state(true_predicates = new_pred,
#                                         last_event = action.end,
#                                         constraints = state['constraints']+constraints,
#                                         optimal_next_actions = next_op_actions,
#                                         min_pddl_length = next_min_length,
#                                         delayed=delayed)
#
#             return [[next_state,1.0]]
#
#     def value(self,state,action):
#         """
#         Cost of performing an action at a state.
#         """
#         return 0.0 if action == '__expand__' else 1.0
#
#     def heuristic(self,state):
#         """
#         Number of unachieved goals, assuming that two goals cannot be achieved
#         simultaneously.
#         """
#         h_val = state['min_pddl_length']
#         #h_val = len(self.task.goals-state['true_predicates'])
#         #h_val = self.h_max.compute_heuristic(state['true_predicates'])
#
#         #print('Heuristic value: %f'%(h_val))
#         # h_val = self.heuristic_approach.compute_heuristic(state['true_predicates'])
#
#         #return len(self.task.goals-state['true_predicates'])
#         return h_val
#
#     def state_risk(self,state):
#         """
#         Returns the scheduling risk bound returned by PARIS.
#         """
#         #Checks if scheduling should be performed
#         if self.perform_scheduling:
#             if not 'scheduling_risk' in state:
#                 squeeze_dict,risk_bound,sc_schedule = self.scheduler.schedule(state['constraints'])
#                 state['scheduling_risk'] = 1.0 if squeeze_dict==None else min(1.0,risk_bound)
#
#             #If the scheduling risk is 1.0, it means that there is not scenario
#             #in which the environment can choose non-violating values for the
#             #durations.
#             if self.is_terminal(state):
#                 return state['scheduling_risk']
#             else:
#                 return 0.0
#         else:
#             return 0.0 #No scheduling risk
#
#     def execution_risk_heuristic(self,state):
#         """
#         The immediate risk is always an admissible estimate of the execution risk.
#         """
#         if 'scheduling_risk' in state:
#             return state['scheduling_risk']
#         else:
#             return 0.0
#
#     def is_terminal(self,state):
#         """A state is terminal if it has been expanded and reached the goal, if
#         no action is available at it or if the best decision is to stay put (empty set of actions)."""
#         return  (not state['delayed']) and ((self.task.goals<=state['true_predicates']) or \
#                 (state['optimal_next_actions'] in [None,[]]))
#
#
#     def _shortest_length_and_actions(self,initial_state,goals,max_steps,min_steps=-1):
#         """
#         Determines the minimum length and optimal first actions for the relaxed PDDL plan.
#         """
#         min_length = self._pa.shortest_plan_length(initial_state,goals,max_steps,min_steps)
#
#         if min_length < float('inf'):
#             #Determines the set of first actions for all optimal plan lengths.
#             optimal_next_actions = self._pa.first_actions(initial_state,goals,min_length)
#         else:
#             optimal_next_actions=None
#
#         return min_length,optimal_next_actions
#
#
