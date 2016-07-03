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

Demo showing the integration between pSulu and RAO* through RMPyL.

@author: Pedro Santana (psantana@mit.edu).
"""
from rao.models.fake_planner import FakePlannerRockSampleModel
from rao.raostar import RAOStar
from rao.export import policy_to_dot,policy_to_rmpyl
import pickle
import ipdb

# sites={'minerals':{'coords':(0.0,-10.0),'value':10.0},
#        'funny_rock':{'coords':(-10.0,-10.0),'value':5.0},
#        'boulder_shadow':{'coords':(-13.0,0.0),'value':3.0},
#        'geiser':{'coords':(0.0,10.0),'value':4.0},
#        'alien_lair':{'coords':(13.0,13.0),'value':100.0}}
#
# prior={'minerals':0.5,
#        'funny_rock':0.5,
#        'boulder_shadow':0.5,
#        'geiser':0.5,
#        'alien_lair':0.2}

# sites={'minerals':{'coords':(0.0,-10.0),'value':10.0}}
# prior={'minerals':0.3}

# sites={'minerals':{'coords':(0.0,-10.0),'value':10.0},
#        'alien_lair':{'coords':(13.0,13.0),'value':100.0}}
#
# prior={'minerals':0.5,
#        'alien_lair':0.2}

# sites={'minerals':{'coords':(0.0,-10.0),'value':10.0},
#        'funny_rock':{'coords':(-10.0,-10.0),'value':5.0},
#        'alien_lair':{'coords':(13.0,13.0),'value':100.0}}
#
# prior={'minerals':0.5,
#        'funny_rock':0.3,
#        'alien_lair':0.2}

sites={'minerals':{'coords':(0.0,-10.0),'value':10.0},
       'funny_rock':{'coords':(-10.0,-10.0),'value':5.0},
       'geiser':{'coords':(0.0,10.0),'value':4.0},
       'alien_lair':{'coords':(13.0,13.0),'value':100.0}}

prior={'minerals':0.5,
       'funny_rock':0.5,
       'geiser':0.5,
       'alien_lair':0.2}


initial_pos = (-12.5,13.5)
cc_fp = FakePlannerRockSampleModel(sites=sites,path_risks=[0.001],
                                   prob_discovery=0.95,velocities={'max':1.0,'avg':0.5},
                                   verbose=0)

b0 = cc_fp.get_initial_belief(initial_pos=initial_pos,prior=prior)

planner = RAOStar(cc_fp,node_name='id',cc=0.003,cc_type='overall',
                  terminal_prob=1.0,randomization=0.0,propagate_risk=True,
                  expand_all_open=True,verbose=1)

policy,explicit,performance = planner.search(b0)

dot_policy = policy_to_dot(explicit,policy)
rmpyl_policy = policy_to_rmpyl(explicit,policy)

#Writes control program to pickle file
with open('rover_policy_rmpyl_fake.pickle','wb') as f:
    pickle.dump(rmpyl_policy,f)

dot_policy.write('rover_policy_fake.svg',format='svg')
rmpyl_policy.to_ptpn(filename='rover_policy_rmpyl_fake_ptpn.tpn')
