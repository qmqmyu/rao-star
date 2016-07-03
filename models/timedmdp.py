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

Defines a Timed Grid MDP class based on http://aima.cs.berkeley.edu/python/mdp.html

@author: Pedro Santana (psantana@mit.edu).
"""
import numpy as np
from . import models

class TimedGridMDP(models.CCHyperGraphModel):
    """Class representing a Timed Grid MDP model."""
    def __init__(self,grid,goal_list,goal_reward=10,time_limit=float('inf'),
                 deterministic=False):
        super(TimedGridMDP,self).__init__()
        self.is_maximization = True #Trying to find shortest path to goal
        self.immutable_actions=True #Actions are always the same
        self._movement_actions = [(-1,0),(0,1),(1,0),(0,-1)]
        self._action_list = self._movement_actions+[()] #STOP
        self.grid = grid
        self.goals = set(goal_list)
        self.time_limit=time_limit
        self.deterministic = deterministic
        self.goal_reward = goal_reward

    def get_state(self,pos,t,halted):
        """
        Returns a proper state representation.
        """
        state_dict = {'position':pos,
                      'time':t,
                      'halted':halted}
        return state_dict

    def get_initial_belief(self,init_pos):
        """
        Proper initial representation of the initial belief state of the search.
        """
        belief = {}
        s0 = self.get_state(init_pos,0,False)
        belief[self.hash_state(s0)] = [s0,1.0]
        return belief

    def actions(self,state):
        """We can move in any of the available directions, if the state is not
        terminal."""
        return [] if self.is_terminal(state) else self._action_list

    def value(self,state,action):
        """Value is the cost of moving."""
        return 0.0 if action==() else -1.0

    def terminal_value(self,state):
        """Value associated with a terminal state."""
        if self._is_goal(state['position']):
            return self.goal_reward #High reward for reaching the goal
        else:
            return 0.0 #No reward, otherwise.

    def heuristic(self,state):
        """The heuristic is the Manhattan distance to closest goal."""
        if self.is_terminal(state):
            return self.terminal_value(state)
        else:
            min_manhattan = min([self._manhattan(state['position'],g) for g in self.goals])
            return self.goal_reward-min_manhattan
            #return np.min([np.sum(np.abs(np.array(state['position'])-np.array(goal[:2]))) for goal in self.goals])

    def state_transitions(self,state,action):
        """The agent moves in the desired direction with high probability, or
        slides to the left or right of the desired path."""
        #Set of potential future states
        if action==():
            return [[self.get_state(state['position'],state['time'],True),1.0]]
        else:
            if self.deterministic:
                potential_transitions = [(self._go(state['position'],action),1.0)]
            else:
                potential_transitions = [(self._go(state['position'],action),0.7),
                                         (self._go(state['position'],self._turn_left(action)),0.2),
                                         (self._go(state['position'],self._turn_right(action)),0.1)]
            valid_transitions={}
            for pos,prob in potential_transitions:
                #If next state is invalid, stays put
                next_pos = pos if self._valid_state(pos) else state['position']
                if next_pos in valid_transitions:
                    valid_transitions[next_pos]+=prob
                else:
                    valid_transitions[next_pos]=prob

            return [[self.get_state(pos,state['time']+1,False),prob] for pos,prob in valid_transitions.items()]

    def _valid_state(self,pos):
        """Checks if the state is out-of-bounds, an obstacle or if the agent
        has run out of time. Returns True is valid."""
        if  pos[0]>=0 and pos[0]<self.grid.shape[0]:
            if pos[1]>=0 and pos[1]<self.grid.shape[1]:
                if self.grid[pos[0],pos[1]] != 1:
                    return True
        return False

    def _go(self,pos,action):
        return tuple([p+a for p,a in zip(pos,action)])

    def _turn_left(self,action):
        return self._movement_actions[self._movement_actions.index(tuple(action))-1]

    def _turn_right(self,action):
        return self._movement_actions[(self._movement_actions.index(tuple(action))+1)%4]

    def _is_goal(self,pos):
        return pos in self.goals

    def _manhattan(self,p1,p2):
        return sum([abs(x-y) for x,y in zip(p1,p2)])

    def observations(self,state):
        """
        For a fully observable model, the state is the observation.
        """
        meas = id(state)
        return [[meas,1.0]]

    def is_terminal(self,state):
        """A state is terminal if it is one of the goals or if the agent ran out
        of time."""
        return True if self._is_goal(state['position']) or state['time']>=self.time_limit or state['halted'] else False

    def state_risk(self,state):
        """No state risk at the moment."""
        return 0.0

    def execution_risk_heuristic(self,state):
        """Heuristic that estimates the risk of execution a plan from a given
        state."""
        return 0.0
