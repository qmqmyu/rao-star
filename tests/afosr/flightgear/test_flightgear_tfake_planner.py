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

Uses tFakePlannerRockSampleModel to solve a partially-observable version
of the FlightGear demo.

@author: Pedro Santana (psantana@mit.edu).
"""
from rmpyl.rmpyl import RMPyL,Episode,TemporalConstraint
from rao.models.fake_planner import tFakePlannerRockSampleModel
from rao.raostar import RAOStar
from rao.export import policy_to_dot,policy_to_rmpyl
import pickle
import yaml
import numpy as np

def load_yaml(filename):
    with open(filename,'r') as f:
        env = yaml.safe_load(f)
    return env

#Generates environment description from YAML file
env = load_yaml('flightgear-environment.yaml')
sites_of_interest={'plantation1':1.0,'plantation2':2.0,'plantation3':3.0,'plantation4':4.0}
sites={}
for loc,loc_dict in env['environment']['features'].items():
    if (loc in ['start_point','end_point']) or (loc in sites_of_interest):
        centroid = np.average(loc_dict['corners'],axis=0)
        sites[loc]={'coords':centroid,'value':(0.0 if loc in ['start_point','end_point'] else sites_of_interest[loc])}

prior={loc:(0.0 if loc in ['start_point','end_point'] else 0.5) for loc in sites.keys()}

cc_fp = tFakePlannerRockSampleModel(sites=sites,duration_type='uniform',
                                    perform_scheduling=False,prob_discovery=1.0,
                                    velocities={'max':100.0,'avg':80.0},name='plane',verbose=0)
#Temporal constraints
init_tcs=[]
init_tcs.append(TemporalConstraint(start=cc_fp.global_start_event,
                                   end=cc_fp.global_end_event,
                                   ctype='controllable',lb=0.0, ub=1000.0))

b0 = cc_fp.get_initial_belief(prior=prior,initial_site='start_point',goal_site='end_point',
                              init_tcs=init_tcs)

planner = RAOStar(cc_fp,node_name='id',cc=0.01,cc_type='overall',
                  terminal_prob=1.0,randomization=0.0,propagate_risk=True,
                  halt_on_violation=False,verbose=1)

#Searches for the optimal policy
policy,explicit,performance = planner.search(b0)

#Converts policy to graphical SVG format
dot_policy = policy_to_dot(explicit,policy)
dot_policy.write('flightgear_policy.svg',format='svg')

#Converts optimal exploration policy into an RMPyL program
exploration_policy = policy_to_rmpyl(explicit,policy)

#The flight policy has the additional actions of taking off and landing.
flight_policy = RMPyL(name='run()')
flight_policy *= flight_policy.sequence(Episode(action='(takeoff plane)'),
                                        exploration_policy,
                                        Episode(action='(land plane)'))

#Eliminates probabilistic choices from the policy, since Pike (in fact, the
#Lisp tpn package) cannot properly handle them.
for obs in flight_policy.observations:
    if obs.type=='probabilistic':
        obs.type = 'uncontrollable'
        del obs.properties['probability']

#Converts the program to a TPN
flight_policy.to_ptpn(filename='flightgear_rmpyl.tpn')

# Writes control program to pickle file
with open('flightgear_rmpyl.pickle','wb') as f:
   pickle.dump(flight_policy,f,protocol=2)
