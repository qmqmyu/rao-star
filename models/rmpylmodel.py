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

Class that frames optimal cRMPL program unraveling as a CC-POMDP.

@author: Pedro Santana (psantana@mit.edu).
"""
from collections import namedtuple
from .models import CCHyperGraphModel
from rmpyl.defs import ChoiceAssignment
from pytemporal.strong_consistency import PTPNStrongConsistency
import ipdb

#The reason why I'm having troubles modeling the ICAPS14 problem using RAO*
#is because ICAPS14 assumed an MPI objective over the controllable choices,
#which is somewhat unrealistic. In the new CC-POMDP framework, I can compute
#correct expected values over the choices, but the branching caused by the
#uncontrollable choices would case the strong consistency checking to be run
#multiple times with the same input. This is why it is important to create an
#object that caches previous solutions so as to avoid duplicate computations.

class RMPyLAction(namedtuple('RMPyLAction',['type','description'])):
    """
    Simple class representing an action in the unraveling of an RMPyL program.
    """
    __slots__= ()
    def __repr__(self):
        if self.type in ['assign','observe']: #Choice
            return '%s %s (%s)%s'%(self.type,self.description.var.name,
                                   self.description.var.id,
                                   '' if self.description.value==None else '='+str(self.description.value))
        else: #Composite or primitive
            return '%s "%s"'%(self.type, self.description)


class BaseRMPyLUnraveler(CCHyperGraphModel):
    """
    Base class for unraveling RMPyL programs in the CC-POMDP framework.
    """
    def __init__(self,verbose=0):
        super(BaseRMPyLUnraveler,self).__init__()
        self.verbose = verbose
        self.is_maximization = True #RMPyL programs seek to maximize utility
        self.immutable_actions = False #Available actions depend on state
        self.global_constraint_store=set()

    def get_state(self,episode_stack,decisions):
        """
        Returns a proper state representation.
        """
        state_dict = {'episode_stack':episode_stack, #Episode stack
                      'decisions':decisions} #Assignments to controllable choices
        return state_dict

    def get_initial_belief(self,prog):
        """
        The search starts with the program episode on the stack.
        """
        belief = {}
        s0 = self.get_state(episode_stack=[prog.plan],decisions=[])
        #Adds initial constraints to global constraint store
        self.global_constraint_store.update(prog.user_defined_temporal_constraints)
        belief[self.hash_state(s0)] = [s0,1.0]
        return belief

    def actions(self,state):
        """
        Actions correspond to either assigning controllable choices, or observing
        uncontrollable ones.
        """
        rmpyl_actions=[]
        if not self.is_terminal(state):
            if len(state['episode_stack'])>0:
                curr_epi = state['episode_stack'][-1] #Episode on top of the stack
                composition = curr_epi.composition

                #Choice episodes, depending on whether they are controllable or
                #uncontrollable, must be treated differently.
                if composition=='choose':
                    choice = curr_epi.start
                    if choice.type=='controllable':
                        #All possible assignments of the controllable choice
                        rmpyl_actions = [RMPyLAction(type='assign',
                                                     description=ChoiceAssignment(choice,v,False)) for v in choice.domain]
                    elif choice.type=='probabilistic':
                        rmpyl_actions = [RMPyLAction(type='observe',
                                                     description=ChoiceAssignment(choice,None,False))]
                    else:
                        raise ValueError('Set-bounded uncontrollable choices cannot be currently handled.')

                #Any other type of composition, including primitive, can only be
                #expanded.
                else:
                    if composition == None: #Primitive episode
                        rmpyl_actions = [RMPyLAction(type='primitive',description=curr_epi.action)]
                    else: #Sequential or parallel episode
                        rmpyl_actions = [RMPyLAction(type='expand',description=composition)]
            else:
                rmpyl_actions = [RMPyLAction(type='halt',description='unraveling')]

        return rmpyl_actions

    def state_transitions(self,state,action):
        """
        Returns the next state, after executing an unraveling operator.
        """
        #Halting action
        if action.type=='halt':
            prob_list = [1.0]
            #Extends the stack with the new active episodes
            stack_list = [None]
            #Decisions are unchanged
            decisions_list= [state['decisions']]
        else:
            curr_epi = state['episode_stack'][-1] #Current episode
            stack_popped = state['episode_stack'][:-1]

            #Updates the global constraint store
            self.global_constraint_store.update(curr_epi.temporal_constraints)

            #For controllable choices, we have a deterministic transition
            #corresponding to the assignment
            if action.type=='assign':
                choice = curr_epi.start
                #Deterministic transition
                prob_list = [1.0]
                #Selects internal episode corresponding to the decision assignment
                next_episode =  curr_epi.internal_episodes[choice.domain.index(action.description.value)]
                #Enstacks the next episode
                stack_list = [ stack_popped+[next_episode] ]
                #Adds the assignment to the controllable choice
                decisions_list= [ state['decisions']+[action.description] ]

            #For probabilistic uncontrollable choices, we have a distribution over
            #next states.
            elif action.type=='observe':
                choice = curr_epi.start
                #Probabilistic transition
                prob_list = choice.probability
                #List of next stack
                stack_list = [stack_popped+[ne] for ne in curr_epi.internal_episodes]
                #All branches share the same decisions
                decisions_list= [state['decisions']]*len(prob_list)

            #Just extends the stack with the new active episodes from a sequential
            #or parallel composition, or nothing for a primitive episode.
            else:
                prob_list = [1.0]
                #Extends the stack with the new active episodes
                stack_list = [ stack_popped+list(reversed(curr_epi.internal_episodes))]
                #Decisions are unchanged
                decisions_list=[state['decisions']]

        next_states=[]

        for p,s,d in zip(prob_list,stack_list,decisions_list):
            next_states.append([self.get_state(episode_stack=s,decisions=d),p])

        return next_states

    def is_terminal(self,state):
        """
        A state is terminal if all active episodes have been unraveled.
        """
        return state['episode_stack']==None

    def value(self,state,action):
        """
        Only controllable choices (decisions) have an associated utility.
        """
        if action.type=='assign':
            return float(action.description.utility)
        else:
            return 0.0

    def terminal_value(self,state):
        """
        No penalty if the unraveling was normally halted.
        """
        return 0.0 if self.is_terminal(state) else -float('inf')

    def heuristic(self,state):
        """
        Simple admissible heuristic that assumes that the highest reward of
        every possible future decision can be attained with probability 1.0.
        """
        return 0.0

    def state_risk(self,state):
        """
        No measure of risk for the base model.
        """
        return 0.0

    def execution_risk_heuristic(self,state):
        """
        The immediate risk is always an admissible execution risk heuristic.
        """
        return self.state_risk(state)

    def observations(self,state):
        """
        Fully observable model - generates a unique observation per state.
        """
        return [[id(state),1.0]]


class StrongStrongRMPyLUnraveler(BaseRMPyLUnraveler):
    """
    Extension of the base RMPyL model to consider scheduling risk from a Strong Consistency
    (with respect to observations), Strong Controllability (with respect to temporal
    durations) perspective.
    """
    def __init__(self,perform_scheduling=True,paris_params={},verbose=0):
        super(StrongStrongRMPyLUnraveler,self).__init__(verbose)
        self.perform_scheduling = perform_scheduling
        self.strong_cons = PTPNStrongConsistency(paris_params=paris_params,
                                                 verbose=verbose>0)

    def state_risk(self,state):
        """
        Returns the scheduling risk bound returned by the strong consistency checker
        that uses PARIS for unconditional scheduling.
        """
        #Checks if scheduling should be performed
        if self.perform_scheduling:
            if not 'scheduling_risk' in state:
                prob_success = self.strong_cons.strong_consistency(state['decisions'],self.global_constraint_store)
                # if prob_success<1.0:
                #     ipdb.set_trace()
                state['scheduling_risk'] = 1.0-prob_success

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
        There are no constraints to be violated, so no risk.
        """
        if 'scheduling_risk' in state:
            return state['scheduling_risk']
        else:
            return 0.0
