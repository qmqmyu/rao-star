#!/usr/bin/env python
#
#  Copyright (c) 2015 MIT. All rights reserved.
#
#   author: Pedro Santana
#   e-mail: psantana@mit.edu
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

Demo of RAO* on a Timed MDP example.

@author: Pedro Santana (psantana@mit.edu).
"""
from rao.models.timedmdp import TimedGridMDP
from rao.raostar import RAOStar
from rao.export import policy_to_dot,policy_to_rmpyl
import pickle
import numpy as np

grid = np.zeros([100,100])
goal = (99,99)
initial_pos = (0,0)

tmpd = TimedGridMDP(grid=grid,goal_list=[goal],goal_reward=1000,deterministic=True)

b0 = tmpd.get_initial_belief(initial_pos)

planner = RAOStar(tmpd,node_name='id',cc=1.0,cc_type='overall',
                  terminal_prob=1.0,randomization=0.0,propagate_risk=True,
                  expand_all_open=True,verbose=1)

policy,explicit,performance = planner.search(b0)

# dot_policy = policy_to_dot(explicit,policy)
# rmpyl_policy = policy_to_rmpyl(explicit,policy)
#
# #Writes control program to pickle file
# with open('rover_policy_rmpyl_fake.pickle','wb') as f:
#     pickle.dump(rmpyl_policy,f)
#
# dot_policy.write('timed_mdp_policy.svg',format='svg')
# rmpyl_policy.to_ptpn(filename='timed_mdp_rmpyl_ptpn.tpn')
