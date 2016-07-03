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

Abstract class representing the general structure of planning problems that can
be represented as hypergraphs.

@author: Pedro Santana (psantana@mit.edu).
"""
from rao.belief import hash_complex_type
from rmpyl.defs import Event

class HyperGraphModel(object):
    """
    Class representing a generic HyperGraph model.
    """
    def __init__(self):
        self.is_maximization=True     #Indicates a maximization problem
        self.immutable_actions=True  #Indicates that available actions are constant

        #Unique global start and end events.
        self.global_start_event = Event(name='global_start_event')
        self.global_end_event = Event(name='global_end_event')

    def actions(self,state):
        """Returns a list of actions at state."""
        raise NotImplementedError('Function actions must be implemented!')

    def value(self,state,action):
        """Returns the value of taking action at state."""
        raise NotImplementedError('Function value must be implemented!')

    def terminal_value(self,state):
        """Value associated with a terminal state."""
        raise NotImplementedError('Function terminal_value must be implemented!')

    def heuristic(self,state):
        """Returns the heuristic estimate of the objective at state."""
        raise NotImplementedError('Function heuristic must be implemented!')

    def state_transitions(self,state,action):
        """Returns a list of pairs (next_state,probability)."""
        raise NotImplementedError('Function state_transitions must be implemented!')

    def observations(self,state):
        """Returns a list of pairs (observation,probability)."""
        raise NotImplementedError('Function observations must be implemented!')

    def obs_repr(self,observation):
        """Returns a convenient representation for displaying the observation.
        By default, returns an empty string."""
        return ''

    def is_terminal(self,state):
        """Returns whether this state is terminal or not."""
        raise NotImplementedError('Function is_terminal must be implemented!')

    def hash_state(self,state):
        """Defines how states should be hashed."""
        return hash_complex_type(state)
        #return hash(hl.md5(str(state).encode('utf-8')).hexdigest())

class CCHyperGraphModel(HyperGraphModel):
    """
    Class representing a HyperGraph model with risk measures.
    """
    def __init__(self):
        super(CCHyperGraphModel,self).__init__()

    def state_risk(self,state):
        """Returns the probability of violating constraints in a given state."""
        raise NotImplementedError('Function state_risk must be implemented!')

    def execution_risk_heuristic(self,state):
        """Heuristic that estimates the risk of execution a plan from a given
        state."""
        raise NotImplementedError('Function execution_risk_heuristic must be implemented!')
