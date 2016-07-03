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

Class representing PDDL domains as (PO)MDP's.

@author: Pedro Santana (psantana@mit.edu).
"""
from . import models
from rmpyl.rmpyl import Episode
from rao.pddl.heuristics import PySATPlanAnalyzer

try:
    from rmpyl.constraints import TemporalConstraint
    from pytemporal.paris import PARIS
    _PYTEMPORAL_FOUND=True
except ImportError:
    _PYTEMPORAL_FOUND=False; print('PyTemporal or RMPyL not found. Cannot assess temporal risk.')

class PDDLEpisode(Episode):
    """
    Simple class encapsuling a PDDL operator and its representation as an RMPyL
    episode.
    """
    def __init__(self,pddl_op,start=None,end=None,**kwargs):
        if 'action' in kwargs:
            raise ValueError('You should not specify an action when creating a PDDL Episode.s')
        else:
            super(PDDLEpisode,self).__init__(start,end,action=pddl_op.name,**kwargs)
        self.pddl_operator = pddl_op

    @property
    def pddl_operator(self):
        return self._pddl_operator

    @pddl_operator.setter
    def pddl_operator(self,new_op):
        self._pddl_operator = new_op

    def __repr__(self):
        return '%s'%(self.action)


class AbstractCCPDDL(models.CCHyperGraphModel):
    """
    Abstract class representing a simple PDDL model as a (PO)MDP.
    """
    def __init__(self,domain_file,prob_file,pddl_pickle='',maximization=False,verbose=0):
        super(AbstractCCPDDL,self).__init__()

        self.verbose = verbose
        self.is_maximization = maximization #Trying to minimize plan cost
        self.immutable_actions = False #Available actions depend on state

        #Gets the PDDL domain and problem files, as well as the grounded task
        if len(pddl_pickle)>0:
            import pickle
            with open(pddl_pickle,'rb') as f:
                self.domain,self.problem,self.task = pickle.load(f)
        else:
            from rao.pddl.model_parser import model_parser
            self.domain,self.problem,self.task = model_parser(domain_file,prob_file,verbose=True)

    def get_state(self,true_predicates):
        """
        Returns a proper state representation.
        """
        state_dict = {'true_predicates':frozenset(true_predicates)}
        return state_dict

    def get_initial_belief(self):
        """
        Proper initial representation of the initial belief state of the search.
        """
        belief = {}
        s0 = self.get_state(self.task.initial_state)
        belief[self.hash_state(s0)] = [s0,1.0]
        return belief

    def actions(self,state):
        """
        Actions available at a state.
        """
        if not self.is_terminal(state):
            true_pred = state['true_predicates'] #True predicates
            return [op for op in self.task.operators if op.applicable(true_pred)]
        else:
            return []

    def value(self,state,action):
        """
        Cost of performing an action at a state.
        """
        return 1.0

    def terminal_value(self,state):
        """
        Final value of a terminal state.
        """
        if self.task.goals <= state['true_predicates']:
            return 0.0 #No cost if all goals were met
        else:
            return float('inf') #Infinite cost if goals were not met.

    def state_transitions(self,state,action,check_applicable=False):
        """
        Returns the next state, after executing an operator (if applicable).
        """
        true_pred = state['true_predicates']
        if check_applicable:
            new_pred = action.apply(true_pred) if action.applicable(true_pred) else true_pred
        else:
            new_pred = action.apply(true_pred)

        next_state = self.get_state(new_pred)
        return [[next_state,1.0]]

    def observations(self,state):
        """
        For a fully observable model, the state is the observation.
        """
        meas = id(state)
        return [[meas,1.0]]

    def is_terminal(self,state):
        """A state is terminal if it has reached the goal."""
        return self.task.goals <= state['true_predicates']


class FiniteHorizonAbstractCCPDDL(AbstractCCPDDL):
    """
    Extension of AbstractCCPDDL with a clock variable to make it finite horizon.
    """
    def __init__(self,domain_file,prob_file,pddl_pickle='',maximization=False,time_limit=-1,verbose=0):
        super(FiniteHorizonAbstractCCPDDL,self).__init__(domain_file,prob_file,pddl_pickle,maximization,verbose)
        if isinstance(time_limit,int):
            self.time_limit = time_limit
        else:
            raise TypeError('Time limit should be an integer value.')

    def get_state(self,true_predicates,t):
        """
        Returns a proper timed state representation.
        """
        state_no_time = super(FiniteHorizonAbstractCCPDDL,self).get_state(true_predicates)
        state_no_time['time']=t
        return state_no_time

    def get_initial_belief(self):
        """
        Proper initial representation of the initial belief state of the search.
        """
        belief = {}
        s0 = self.get_state(self.task.initial_state,t=0)
        belief[self.hash_state(s0)] = [s0,1.0]
        return belief

    def state_transitions(self,state,action,check_applicable=False):
        """
        Returns the next state, after executing an operator (if applicable).
        """
        true_pred = state['true_predicates']
        if check_applicable:
            new_pred = action.apply(true_pred) if action.applicable(true_pred) else true_pred
        else:
            new_pred = action.apply(true_pred)

        next_state = self.get_state(new_pred,t=state['time']+1)
        return [[next_state,1.0]]

    def is_terminal(self,state):
        """A state is terminal if it has reached the goal or if it has run
        out of time."""
        if self.time_limit==state['time']:
            return True
        else:
            true_pred = state['true_predicates']
            return self.task.goal_reached(true_pred)

if _PYTEMPORAL_FOUND:

    class DurativePDDL(AbstractCCPDDL):
        """
        Class encapsulating a PDDL model with durative actions and temporal
        constraints. Action durations can be controllable, set-bounded, or
        probabilistic.
        """
        def __init__(self,domain_file,prob_file,pddl_pickle='',maximization=False,
                     perform_scheduling=True,duration_func=None,paris_params={},
                     max_steps=50,verbose=0):
            #Initializes the generic PDDL model
            super(DurativePDDL,self).__init__(domain_file,prob_file,pddl_pickle,maximization,verbose)

            self.perform_scheduling = perform_scheduling
            self.duration_func = duration_func

            #Maximum number of actions allowed in any plan
            self.max_steps = max_steps

            #PDDL plan analyzer
            self._pa = PySATPlanAnalyzer(domain_file,prob_file,precompute_steps=max_steps,
                                         sequential=True,remove_static=True,
                                         verbose=verbose>0)

            if self.perform_scheduling:
                self.scheduler = PARIS(**paris_params)

        def get_state(self,true_predicates,last_event,constraints,
                      optimal_next_actions,min_pddl_length,delayed):
            """
            Returns a proper state representation.
            """
            state_dict = {'true_predicates':frozenset(true_predicates),
                          'last_event':last_event, #Temporal event associated with state
                          'constraints':constraints,     #Temporal constraints
                          'optimal_next_actions':optimal_next_actions, #Best actions (length-wise) to execute
                          'min_pddl_length':min_pddl_length, #Minimum length of the relaxed PDDL plan.
                          'delayed':delayed} #Whether the state is on an optimal PDDL plan

            return state_dict

        def get_initial_belief(self,constraints=[]):
            """
            Proper initial representation of the initial belief state of the search.
            """
            belief = {}

            #Determines the minimum length and the optimal first actions for the relaxed PDDL plan.
            min_length,op_next_actions = self._shortest_length_and_actions(self.task.initial_state,
                                                                           self.task.goals,
                                                                           self.max_steps)

            #Temporal constraint that makes sure that the start and end events of a
            #mission are correctly aligned.
            start_before_end = TemporalConstraint(start=self.global_start_event,end=self.global_end_event,
                                               ctype='controllable',lb=0.0,ub=float('inf'))

            s0 = self.get_state(true_predicates = self.task.initial_state,
                                last_event = self.global_start_event,
                                constraints = constraints+[start_before_end],
                                optimal_next_actions = op_next_actions,
                                min_pddl_length = min_length,
                                delayed = False)

            belief[self.hash_state(s0)] = [s0,1.0]
            return belief

        def actions(self,state):
            """
            Actions available at a state.
            """
            if not self.is_terminal(state):
                #If this state has has its expansion delayed, return the expansion action
                if state['delayed']:
                    return ['__expand__']
                #If not, returns all applicable operators.
                else:
                    true_pred = state['true_predicates'] #True predicates
                    if self.duration_func == None:
                        return [PDDLEpisode(op) for op in self.task.operators if op.applicable(true_pred)]
                    else:
                        return [PDDLEpisode(op,duration=self.duration_func(op.name)) for op in self.task.operators if op.applicable(true_pred)]
            else:
                return []

        def state_transitions(self,state,action):
            """
            Returns the next state, after executing an operator.
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

        def value(self,state,action):
            """
            Cost of performing an action at a state.
            """
            return 0.0 if action == '__expand__' else 1.0

        def heuristic(self,state):
            """
            Number of unachieved goals, assuming that two goals cannot be achieved
            simultaneously.
            """
            return state['min_pddl_length']

        def state_risk(self,state):
            """
            Returns the scheduling risk bound returned by PARIS.
            """
            #Checks if scheduling should be performed
            if self.perform_scheduling:
                if not 'scheduling_risk' in state:
                    squeeze_dict,risk_bound,sc_schedule = self.scheduler.schedule(state['constraints'])
                    state['scheduling_risk'] = 1.0 if squeeze_dict==None else min(1.0,risk_bound)

                #If the scheduling risk is 1.0, it means that there is not scenario
                #in which the environment can choose non-violating values for the
                #durations.
                if self.is_terminal(state):
                    return state['scheduling_risk']
                else:
                    return 0.0
            else:
                return 0.0 #No scheduling risk

        def execution_risk_heuristic(self,state):
            """
            The immediate risk is always an admissible estimate of the execution risk.
            """
            if 'scheduling_risk' in state:
                return state['scheduling_risk']
            else:
                return 0.0

        def is_terminal(self,state):
            """A state is terminal if it has been expanded and reached the goal, if
            no action is available at it or if the best decision is to stay put (empty set of actions)."""
            return  (not state['delayed']) and ((self.task.goals<=state['true_predicates']) or \
                    (state['optimal_next_actions'] in [None,[]]))


        def _shortest_length_and_actions(self,initial_state,goals,max_steps,min_steps=-1):
            """
            Determines the minimum length and optimal first actions for the relaxed PDDL plan.
            """
            min_length = self._pa.shortest_plan_length(initial_state,goals,max_steps,min_steps)

            if min_length < float('inf'):
                #Determines the set of first actions for all optimal plan lengths.
                optimal_next_actions = self._pa.first_actions(initial_state,goals,min_length)
            else:
                optimal_next_actions=None

            return min_length,optimal_next_actions
