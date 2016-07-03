#!/usr/bin/env python
#
#  Copyright (c) 2016 MIT. All rights reserved.
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

Tests PySAT, our basic SAT-based STRIPS planner.

@author: Pedro Santana (psantana@mit.edu).
"""
from rao.external_planners.pysat import PySAT
from rao.pddl.model_parser import model_parser
import time

#PDDL files for the RSS demo
dom_file = 'rss-domain-strips.pddl'
prob_file = 'rss-current-problem-strips.pddl'
max_sols=10

py_sat = PySAT(dom_file,prob_file,precompute_steps=40,remove_static=True,verbose=True)

domain,problem,task = model_parser(dom_file,prob_file,remove_static=True)

print('\n##### Determining optimal plan length!\n')
start = time.time()
min_steps = len(task.goals-task.initial_state)
plans = py_sat.plan(task.initial_state,task.goals,time_steps=35,
                    find_shortest=True,min_steps=min_steps)
elapsed = time.time()-start
print('\n##### All solving took %.4f s'%(elapsed))

if len(plans)>0:

    shortest_plan = plans[0]
    print('\n##### A plan with %d steps exists!\n'%(len(shortest_plan)))

    start = time.time()
    best_first_actions = py_sat.first_actions(task.initial_state,task.goals,time_steps=len(shortest_plan))
    elapsed = time.time()-start
    print('\n##### Number of different first actions for optimal plans: %d.'%(len(best_first_actions)))
    for a in best_first_actions:
        print(a)
    print('\n##### All solving took %.4f s'%(elapsed))

    print('\n##### Retrieving %s shortest solutions now.\n'%('all' if max_sols<0 else 'up to %d'%(max_sols)))
    start = time.time()
    all_plans = py_sat.plan(task.initial_state,task.goals,time_steps=len(shortest_plan),max_sols=max_sols)
    elapsed = time.time()-start
    print('\n##### All solving took %.4f s'%(elapsed))

    print('Number of computed plans with optimal length: %d'%(len(all_plans)))

    for plan_count,plan in enumerate(all_plans):
        print('\n##### Shortest plan %d '%(plan_count+1))
        for t,action in enumerate(plan):
            print('%d: %s'%(t,action))
else:
    print('\n##### No plan found.')
