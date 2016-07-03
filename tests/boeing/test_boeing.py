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

Demo of RAO* being used to generate plans for the Boeing demo.

@author: Pedro Santana (psantana@mit.edu).
""" 
from rmpyl.examples.generic_manipulation import Object,Location,Manipulator,valid_assignment
from rao.models.boeing import GenericManipulation,BoeingRMPyLModel
from rao.raostar import RAOStar

#Parameters for the Boeing demo
components = ['redcomp']#['redcomp','bluecomp','greencomp','yellowcomp']
solders = ['solderobj']
cleaners = ['cleanerobj']
object_names = components+solders+cleaners

targets = ['redt']#['redt','bluet','greent','yellowt']
bins = ['redb','solderb','cleanb']#['redb']#['redb','blueb','greenb','yellowb','solderb','cleanb']
location_names = targets+bins

robot_manip = ['baxter_left']#['baxter_left','baxter_right']
human_manip = ['human_hand']
manipulator_names = robot_manip+human_manip

#Objects
component_objs = [Object(name=n,ob_type='componenttype',location_names=location_names,
                         manipulator_names=manipulator_names) for n in components]
solder_objs = [Object(name=n,ob_type='soldertype',location_names=location_names,
                         manipulator_names=manipulator_names) for n in solders]
cleaner_objs = [Object(name=n,ob_type='cleanertype',location_names=location_names,
                         manipulator_names=manipulator_names) for n in cleaners]
all_objects = component_objs+solder_objs+cleaner_objs

#Locations
target_objs = [Location(name=n,object_names=object_names) for n in targets]
bin_objs = [Location(name=n,object_names=object_names) for n in bins]
all_locations = target_objs+bin_objs

#Manipulators
robot_manip_objs = [Manipulator(name=n,ag_type='robot_agent',object_names=object_names) for n in robot_manip]
human_manip_objs = [Manipulator(name=n,ag_type='human_agent',object_names=object_names) for n in human_manip]
all_manipulators = robot_manip_objs+human_manip_objs

#Initial state
initial_state={}
for bin_ob,object_name in zip(bin_objs,object_names):
    initial_state[bin_ob.clean]='clean'
    initial_state[bin_ob.occupying_object]=object_name

for target_ob in target_objs: 
    initial_state[target_ob.clean]='not-clean'   
    initial_state[target_ob.occupying_object]='nothing'

for manip_ob in all_manipulators:
    initial_state[manip_ob.moving]='not-moving'
    initial_state[manip_ob.available]='available'
    initial_state[manip_ob.holding]='nothing'

for object_ob,bin_name in zip(all_objects,bins):
    initial_state[object_ob.location] = bin_name
    initial_state[object_ob.soldered] = 'not-soldered'
    initial_state[object_ob.holder] = 'nobody' 

#Goal state  
goal_state={}
for comp_ob,comp_target in zip(component_objs,targets):
    goal_state[comp_ob.location]=comp_target
    goal_state[comp_ob.soldered]='soldered'

################################################

if valid_assignment(initial_state) and valid_assignment(goal_state):
                      
    gen_manip_model = GenericManipulation(all_objects,all_manipulators,
                                          all_locations,goal_state)

    boeing_rmpyl_model = BoeingRMPyLModel(gen_manip_model,verbose=0)   
    

    b0 = boeing_rmpyl_model.get_initial_belief(initial_state)
                           
    planner = RAOStar(boeing_rmpyl_model,node_name='id',cc=1.0,terminal_prob=1.0,
                      propagate_risk=True,verbose=1)

    policy,explicit,performance = planner.search(b0)
    dot_policy = explicit.policy_to_dot(policy)
    rmpyl_policy = explicit.policy_to_rmpyl(policy) 

    dot_policy.write('boeing_rmpyl_policy.svg',format='svg')
    rmpyl_policy.to_ptpn(filename='boeing_rmpyl_ptpn.tpn')

else:
    print('Invalid initial or goal states.')








