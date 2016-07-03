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

Class representing RMPyL programs as (PO)MDP's.

@author: Pedro Santana (psantana@mit.edu).
"""
from collections import namedtuple
from . import models
from rmpyl.defs import Choice

class RMPyLChooseAction(namedtuple('RMPyLChooseAction',['type','action','value'])):
    """
    Simple class representing an action in an RMPyL program.
    """
    __slots__= ()

    def __repr__(self):
        return self.action


class RMPyLModel(models.CCHyperGraphModel):
    """
    Class representing an RMPyL program as a (PO)MDP.
    """
    def __init__(self,prog,verbose=0):
        super(RMPyLModel,self).__init__()

        self.verbose = verbose
        self.is_maximization = True #RMPyL programs seek to maximize utility
        self.immutable_actions = False #Available actions depend on state
        self.prog = prog #RMPyL program

        self.episode_start_map = {e.start.id:e for e in prog.episodes}
        self.event_successors = {k.id:v for k,v in prog.event_successors.items()}
        self.terminal_events = set([e.end.id for e in self.episode_start_map.values() if e.terminal or (e.action in ['__stop__','(noop)'])])

        for ep in self.episode_start_map.values():
            if ep.composition=='parallel':
                raise TypeError('RAO* cannot handle parallel compositions!')

        self._decisions={}; self._observations={}
        for c in prog.choices: #Ensures choices are of the proper type
            if c.type=='controllable':
                if 'utility' in c.properties:
                    self._decisions[c.id]=c
                else:
                    raise RuntimeError('Controllable choices must have associated utilities.')
            elif c.type=='probabilistic':
                if 'probability' in c.properties:
                    self._observations[c.id]=c
                else:
                    raise RuntimeError('Uncontrollable choices must have associated probabilities.')
            else:
                raise RuntimeError('RAO* can currently only deal with controllable and probabilistic choices.')

        # Map from decision ID's to their best assignment
        self._max_util_map={d.id:max(d.properties['utility']) for d_id,d in self._decisions.items()}

        if verbose>0:
            print('\n***** Start event\n')
            print(prog.first_event)

            print('\n***** Last event\n')
            print(prog.last_event)

            print('\n***** Primitive episodes\n')
            for i,p in enumerate(prog.primitive_episodes):
                print('%d: %s\n'%(i+1,str(p)))

            print('\n***** Events\n')
            for i,e in enumerate(prog.events):
                print('%d: %s'%(i+1,str(e)))

            print('\n***** Event successors\n')
            for i,(ev,successors) in enumerate(prog.event_successors.items()):
                print('%d: %s -> %s'%(i+1,ev.name,str([e.name for e in successors])))

            print('\n***** Temporal constraints\n')
            for i,tc in enumerate(prog.temporal_constraints):
                print('%d: %s\n'%(i+1,str(tc)))

    def get_state(self,event,observation=None):
        """
        Returns a proper state representation.
        """
        if len(event.support)>1:
            raise ValueError('Did not expect to have more than one conjunction.')

        state_dict = {'event':event}
        if observation!=None:
            state_dict['observation']=observation

        return state_dict

    def get_initial_belief(self):
        """
        Proper initial representation of the initial belief state of the search.
        """
        belief = {}
        event = self._next_executable_start(self.prog.first_event)
        s0 = self.get_state(event=event)
        belief[self.hash_state(s0)] = [s0,1.0]
        return belief

    def state_transitions(self,state,action):
        """
        Returns the next state, after executing an operator (if applicable).
        """
        #Controllable choices have deterministic effects. Returns the successor
        #event corresponding to the value chosen for the decision.
        event = state['event']

        if action.type=='decide':
            val_idx = event.domain.index(action.value)
            successor = self.event_successors[event.id][val_idx]
            return [[self.get_state(event=self._next_executable_start(successor)),1.0]]

        #Returns the list of possible observations, with corresponding probabilities
        elif action.type=='observe':
            next_states=[]
            for idx,successor in enumerate(self.event_successors[event.id]):
                next_states.append([self.get_state(event=self._next_executable_start(successor),
                                                   observation=event.domain[idx]),
                                    event.properties['probability'][idx]])
            return next_states

        #Primitive method
        else:
            successor = self.event_successors[event.id][0]
            return [[self.get_state(event=self._next_executable_start(successor)),1.0]]

    def actions(self,state):
        """
        Actions available at a state.
        """
        if self.is_terminal(state): #No actions at terminal states
            return []
        else:
            event = state['event']
            #Controllable choices return a list of possible decisions, all
            #with deterministic effects.
            if event.id in self._decisions:
                return [RMPyLChooseAction(type='decide',
                                          action='(decide-%s=%s)'%(event.name,val),
                                          value=val) for val in event.domain]
            #Probabilistic uncontrollable choices return a single action (observe)
            #with a non-deterministic effect.
            elif event.id in self._observations:
                return [RMPyLChooseAction(type='observe',
                                          action='(observe-'+(event.name)+')',
                                          value='')]
            else:#If not a choice (decision or observation), must be primitive episode
                return [RMPyLChooseAction(type='primitive',
                                          action=self.episode_start_map[event.id].action,
                                          value='')]

    def is_terminal(self,state):
        """
        A state is terminal if it is the last event or the end event of a __stop__
        action.
        """
        return (state['event'].id==self.prog.last_event.id) or (state['event'].id in self.terminal_events)

    def value(self,state,action):
        """
        Only controllable choices (decisions) have an associated utility.
        """
        if action.type=='decide':
            return float(state['event'].properties['utility'][state['event'].domain.index(action.value)])
        else:
            return 0.0

    def terminal_value(self,state):
        """
        Final value of a terminal state.
        """
        return 0.0

    def heuristic(self,state):
        """
        Heuristic estimate of the expected value associated with a state.
        """
        #Maximal possible utility from the remaining decisions on the same path
        state_support = state['event'].support
        assigned_ids = set([assig.var.id for assig in state_support[0]])
        rem_consistent = [d for d in self._decisions.values() if (not d.id in assigned_ids) and len(d.static_support_AND(state_support,d.support))>0]
        return sum([self._max_util_map[d.id] for d in rem_consistent])

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
        meas = state['observation'] if 'observation' in state else id(state)
        return [[meas,1.0]]

    def obs_repr(self,observation):
        """
        Observation are represented by their own strings.
        """
        return observation

    def _next_executable_start(self,event):
        """
        Returns the first executable event (choice, observation, or primitive
        episode) coinciding or following the argument.
        """
        if event.id==self.prog.last_event.id or isinstance(event,Choice):
            return event
        else:
            #Current episode is the start event of something
            if event.id in self.episode_start_map:
                episode = self.episode_start_map[event.id]

                if episode.composition=='': #Primitive episode (executable)
                    return event
                else:
                    #Moves on to the start event of the first episode in the
                    #sequence
                    if episode.composition=='sequence':
                        return self._next_executable_start(*self.event_successors[event.id])
                    else:#parallel composition, which cannot be handled at the moment
                        pass

            #The current event is the end event of some episode, so moves on to
            #its successor.
            else:
                return self._next_executable_start(*self.event_successors[event.id])
