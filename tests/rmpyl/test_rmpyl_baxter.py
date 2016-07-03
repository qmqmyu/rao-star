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

Demo of RAO* being used to generate plans from an RMPyL program.

@author: Pedro Santana (psantana@mit.edu).
"""
from rao.raostar import RAOStar
from rao.export import policy_to_dot,policy_to_rmpyl
from rao.models.rmpylmodel import BaseRMPyLUnraveler
from baxter_collaborative import collaborative_pick_and_place

blocks=['Red','Green','Blue','Yellow']
#blocks=['Red','Green','Blue']
#blocks=['Red','Green']
#blocks=['Red']


#dur_dict = None #All activities have controllable, positive durations
dur_dict = {'ctype':'uncontrollable_probabilistic',
            'distribution':{'type':'uniform','lb':1.0,'ub':5.0}}

#time_window=-1 #No temporal window
time_window=10.0

#cc = 1.0 #Chance constriant is effectively innactive
cc = 0.37

prog = collaborative_pick_and_place(blocks,time_window,dur_dict,write_tpn=False,write_pickle=False)

prog.to_ptpn(filename='rmpyl_baxter_input_ptpn.tpn')

rmpyl_model = BaseRMPyLUnraveler()
# rmpyl_model = StrongStrongRMPyLModel(prog,verbose=0)

b0 = rmpyl_model.get_initial_belief(prog)

planner = RAOStar(rmpyl_model,node_name='id',cc=cc,cc_type='overall',
                  terminal_prob=1.0,randomization=0.0,propagate_risk=True,
                  verbose=1)
#ipdb.set_trace()
policy,explicit,performance = planner.search(b0)

dot_policy = policy_to_dot(explicit,policy)
dot_policy.write('rmpyl_baxter_collaborative_policy.svg',format='svg')

rmpyl_policy = policy_to_rmpyl(explicit,policy)
rmpyl_policy.to_ptpn(filename='rmpyl_baxter_collaborative_policy_ptpn.tpn')
