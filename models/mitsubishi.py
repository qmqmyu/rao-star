#!/usr/bin/env python
#
#  A Python model for the MERS Mitsubishi manufacturing demo
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

Class representing the Mitsubishi manufacturing demo@MERS

@author: Pedro Santana (psantana@mit.edu).
"""
import copy as cp
from .pddlmodel import FiniteHorizonAbstractCCPDDL

class SimpleFaultMitsubishi(FiniteHorizonAbstractCCPDDL):
    """
    Class representing a simple finite-horizon (PO)MDP representation of the
    Mitsubishi demo (no temporal constraints) with simple faults, in which an
    action does not succeed and leaves the state of the system unchanged.
    """
    def __init__(self,domain_file,prob_file,pddl_pickle='',time_limit=-1,drop_penalty=10.0,
                 p_fail=0.0,verbose=0):
        super(SimpleFaultMitsubishi,self).__init__(domain_file,prob_file,pddl_pickle,
                                                    maximization=False,
                                                    time_limit=time_limit,
                                                    verbose=verbose)
        if p_fail >=0.0 and p_fail <=1.0:
            self.p_fail = p_fail
        else:
            raise TypeError('Probability of failing must be in [0,1]')

        if drop_penalty>=0.0:
            self.drop_penalty = drop_penalty
        else:
            raise TypeError('The penalty for dropping goals should be positive!')

        self.action_cost=1.0 #Cost of performing an action

    def actions(self,state):
        """
        PDDL actions available at a state.
        """
        pddl_actions = super(SimpleFaultMitsubishi,self).actions(state)
        return pddl_actions+['(stop)']

    def value(self,state,action):
        """
        Cost of performing an action at a state.
        """
        return 0.0 if action == '(stop)' else self.action_cost

    def terminal_value(self,state):
        """
        Final value of a terminal state.
        """
        true_pred = state['true_predicates']
        if self.task.goal_reached(true_pred):
            return 0.0 #No cost if all goals were met
        else:
            return len(self.task.goals-state['true_predicates'])*self.drop_penalty

    def heuristic(self,state):
        """
        Heuristic estimate of the expected value associated with a state.
        """
        if self.time_limit<0: #unlimited steps
            return len(self.task.goals-state['true_predicates'])*self.action_cost
        else:
            num_goals_to_achieve = len(self.task.goals-state['true_predicates'])
            num_steps_remaining = self.time_limit-state['time']

            if num_steps_remaining>=num_goals_to_achieve:
                return num_goals_to_achieve*self.action_cost
            else:
                return num_goals_to_achieve*self.action_cost+(num_goals_to_achieve-num_steps_remaining)*self.drop_penalty

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

    def state_transitions(self,state,action,check_applicable=False):
        """
        Returns the next state, after executing an operator (if applicable).
        """
        if action == '(stop)':
            final_state = cp.deepcopy(state)
            final_state['time'] = self.time_limit
            return [[final_state,1.0]]
        else:
            state_tuples = super(SimpleFaultMitsubishi,self).state_transitions(state,action,check_applicable)
            next_state,prob = state_tuples[0]

            if (self.p_fail==0.0)or(next_state['true_predicates']==state['true_predicates']):
                return state_tuples
            else:
                failed_state = cp.deepcopy(state)
                failed_state['time']+=1
                return [[next_state,1.0-self.p_fail],[failed_state,self.p_fail]]





class DiagnosticMitsubishi(FiniteHorizonAbstractCCPDDL):
    """
    Class representing a partially-observable planning problem, in which the robot
    can be in one of two states: fit or unfit to complete a task. The robot
    can transition from fit to unfit and vice-versa dynamically, as it completes tasks
    and asks for help. A robot in an unfit state has a very high chance of not being
    able to complete its current action.

    Limit the probability of asking the human for unnecessary help.
    """
    def __init__(self,domain_file,prob_file,pddl_pickle='',time_limit=-1,verbose=0,param_dict={}):
        super(DiagnosticMitsubishi,self).__init__(domain_file,prob_file,pddl_pickle,
                                                    maximization=False,
                                                    time_limit=time_limit,
                                                    verbose=verbose)

        self.params = {'p_fail_fit':0.2,      #Prob. of failing a task, while being fit
                       'p_fail_unfit':0.999,  #Prob. of failing a task, while not being fit
                       'p_fit_fit':0.5,       #Prob. of remaining fit, if fit before
                       'p_fit_unfit':0.5,     #Prob. becoming fit, if unfit before
                       'goal_drop_penalty':10.0, #Penalty for not achieving a goal
                       'robot_action_cost':1.0, #Cost of robot performing an action
                       'human_action_cost':5.0} #Cost of human performing an action


        #Updates the values of the parameters
        for name,value in param_dict.items():
            if name in self.params:
                self.params[name]=value
            else:
                raise TypeError(name+' is not valid. Valid parameters are: '+str(list(self.params.keys())))

        if verbose==1:
            print('\n***** PARAMETERS *****\n')
            for name,value in self.params.items():
                print(name+': '+str(value))
            print('\n**********************\n')

    def get_state(self,true_predicates,t,fit,disturbed,last_action,failed):
        """
        Returns a proper state representation.
        """
        #Original state for finite-horizon PDDL
        state_dict = super(DiagnosticMitsubishi,self).get_state(true_predicates,t)

        state_dict['fit'] = fit #Whether the robot is fit for the task
        state_dict['disturbed'] = disturbed #Whether the robot disturbed a human
        state_dict['last_action'] = last_action #Last action executed
        state_dict['failed'] = failed #Whether the last action failed

        return state_dict

    def get_initial_belief(self):
        """
        The robot starts not knowing whether it is fit or not.
        """
        belief = {}
        s0_fit = self.get_state(self.task.initial_state,t=0,fit=True,
                                disturbed=False,last_action='_none_',
                                failed=False)
        s0_unfit = self.get_state(self.task.initial_state,t=0,fit=False,
                                  disturbed=False,last_action='_none_',
                                  failed=False)

        belief[self.hash_state(s0_fit)] = [s0_fit,0.5]
        belief[self.hash_state(s0_unfit)] = [s0_unfit,0.5]
        return belief

    def actions(self,state):
        """
        PDDL actions available at a state.
        """
        actions = super(DiagnosticMitsubishi,self).actions(state)+['(stop)']
        #All PDDL operators, plus (stop), that halts execution, and (help), that
        #asks a human for help to complete the last action.
        if not state['last_action'] in ['(help)','(stop)','_none_']:
            actions+=['(help)']

        return actions

    def value(self,state,action):
        """
        Cost of performing an action at a state.
        """
        if action == '(stop)':
            return 0.0
        elif action == '(help)':
            return self.params['human_action_cost']
        else:
            return self.params['robot_action_cost']

    def terminal_value(self,state):
        """
        Final value of a terminal state.
        """
        true_pred = state['true_predicates']
        if self.task.goal_reached(true_pred):
            return 0.0 #No cost if all goals were met
        else:
            return len(self.task.goals-state['true_predicates'])*self.params['goal_drop_penalty']

    def heuristic(self,state):
        """
        Heuristic estimate of the expected value associated with a state.
        """
        if self.time_limit<0: #unlimited steps
            return len(self.task.goals-state['true_predicates'])*self.params['robot_action_cost']
        else:
            num_goals_to_achieve = len(self.task.goals-state['true_predicates'])
            num_steps_remaining = self.time_limit-state['time']

            if num_steps_remaining>=num_goals_to_achieve:
                return num_goals_to_achieve*self.params['robot_action_cost']
            else:
                return num_goals_to_achieve*self.params['robot_action_cost']+(num_goals_to_achieve-num_steps_remaining)*self.params['goal_drop_penalty']

    def state_risk(self,state):
        """
        If the robot has disturbed a human, even though it was fit for executing
        a task, returns 1 (unwanted event). Otherwise, returns 0.
        """
        return 1.0 if state['disturbed'] else 0.0

    def execution_risk_heuristic(self,state):
        """
        Estimates the risk of disturbing a human.
        """
        return self.state_risk(state)

    def state_transitions(self,state,action,check_applicable=True):
        """
        Returns the next state, after executing an operator (if applicable).
        """
        if action == '(stop)': #Deterministically ends execution
            final_state = self.get_state(true_predicates=state['true_predicates'],
                                        t=self.time_limit,
                                        fit=state['fit'],
                                        disturbed=state['disturbed'],
                                        last_action='(stop)',
                                        failed=False)

            return [[final_state,1.0]]

        #Asks a human co-worker for help, with deterministic effects.
        elif action == '(help)':

            #A disturbance is flagged either if the human has been disturbed in
            #the past, or if the robot asked for help while being fit for a task
            disturbance = True if (state['disturbed'] or state['fit']) else False

            #Valid PDDL action
            if not state['last_action'] in ['(help)','(stop)','_none_']:
                true_pred = state['true_predicates']

                #Deterministic human action
                op = state['last_action']
                if check_applicable:
                    new_pred = op.apply(true_pred) if op.applicable(true_pred) else true_pred
                else:
                    new_pred = op.apply(true_pred)

                #We are not sure if the robot if fit for the next task anymore
                new_states = [self.get_state(true_predicates=new_pred,
                                            t=state['time']+1,
                                            fit=fitness,
                                            disturbed=disturbance,
                                            last_action='(help)',
                                            failed=False) for fitness in [True,False]]

                if state['fit']:
                    return [[new_states[0],self.params['p_fit_fit']],
                            [new_states[1],1.0-self.params['p_fit_fit']]]
                else:
                    return [[new_states[0],self.params['p_fit_unfit']],
                            [new_states[1],1.0-self.params['p_fit_unfit']]]


            #If previous action wasn't valid PDDL operators, does nothing
            #(potentially, just annoys the human)
            else:
                new_state = self.get_state( true_predicates=state['true_predicates'],
                                            t=state['time']+1,
                                            fit=state['fit'],
                                            disturbed=disturbance,
                                            last_action='(help)',
                                            failed=False)

                return [[new_state,1.0]]

        else: #standard PDDL action, which can fail with no effect, depending on
              #whether is fit for the task or not.
            true_pred = state['true_predicates']
            if check_applicable:
                new_pred = action.apply(true_pred) if action.applicable(true_pred) else true_pred
            else:
                new_pred = action.apply(true_pred)

            success_states = [self.get_state(true_predicates=new_pred, #New predicates
                                            t=state['time']+1,
                                            fit=fitness,
                                            disturbed=state['disturbed'],
                                            last_action=action,
                                            failed=False) for fitness in [True,False]]

            #Predicates didn't change, so it doesn't matter if action failed
            #or not.
            if new_pred == true_pred:
                if state['fit']:
                    return [[success_states[0],self.params['p_fit_fit']],
                            [success_states[1],1.0-self.params['p_fit_fit']]]
                else:
                    return [[success_states[0],self.params['p_fit_unfit']],
                            [success_states[1],1.0-self.params['p_fit_unfit']]]

            #Success and failure produce distinct predicates
            else:
                failed_state = self.get_state(true_predicates=true_pred, #Old predicates
                                              t=state['time']+1,
                                              fit=state['fit'],
                                              disturbed=state['disturbed'],
                                              last_action=action,
                                              failed=True)

                if state['fit']:
                    if self.params['p_fail_fit'] == 1.0:
                        return [[failed_state,self.params['p_fail_fit']]]
                    elif self.params['p_fail_fit'] == 0.0:
                        return [[success_states[0],self.params['p_fit_fit']],
                                [success_states[1],1.0-self.params['p_fit_fit']]]
                    else:
                        return [[success_states[0],(1.0-self.params['p_fail_fit'])*self.params['p_fit_fit']],
                                [success_states[1],(1.0-self.params['p_fail_fit'])*(1.0-self.params['p_fit_fit'])],
                                [failed_state,self.params['p_fail_fit']]]
                else:
                    if self.params['p_fail_unfit'] == 1.0:
                        return [[failed_state,self.params['p_fail_unfit']]]
                    elif self.params['p_fail_unfit'] == 0.0:
                        return [[success_states[0],self.params['p_fit_unfit']],
                                [success_states[1],1.0-self.params['p_fit_unfit']]]
                    else:
                        return [[success_states[0],(1.0-self.params['p_fail_unfit'])*self.params['p_fit_unfit']],
                                [success_states[1],(1.0-self.params['p_fail_unfit'])*(1.0-self.params['p_fit_unfit'])],
                                [failed_state,self.params['p_fail_unfit']]]


    def observations(self,state):
        """
        The observation corresponds it the previous action failed or not.
        """
        meas = 'Fail!' if state['failed'] else 'Success!'
        return [[meas,1.0]]


    def obs_repr(self,observation):
        """
        Observation are represented by their own strings.
        """
        return observation
