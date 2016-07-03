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

Demo of RAO* being used to generate plans for the RSS midyear review.

@author: Pedro Santana (psantana@mit.edu), Tiago Vaquero (tvaquero@mit.edu).
"""
import os

from rao.raostar import RAOStar
from rao.export import policy_to_dot,policy_to_rmpyl
from rao.models.rss_durative_pddl import RSSDurativePDDL
from rmpyl.constraints import TemporalConstraint
from rss_model_utils import rss_duration_func,rss_time_window_func
from pytemporal.paris import PARIS

# from rao.pddl.model_parser import model_parser
# from rao.pddl.heuristics import PySATPlanAnalyzer

#PDDL files for the RSS demo
path = os.path.dirname(os.path.abspath(__file__))

dom_file = os.path.join(path,'rss-domain-strips.pddl')
prob_file = os.path.join(path,'rss-current-problem-strips.pddl')

#dom_file = os.path.join(path,'domain_with_no_plan_rao.pddl')
#prob_file = os.path.join(path,'problem_with_no_plan_rao.pddl')

# domain,problem,task = model_parser(dom_file,prob_file,remove_static=True)
# plan_analysis = PySATPlanAnalyzer(dom_file,prob_file,precompute_steps=40,
#                                   sequential=True,remove_static=True,verbose=True)
#
# min_length = plan_analysis.shortest_plan_length(task.initial_state,task.goals,30)
# plan_analysis.first_actions(task.initial_state,task.goals,min_length)

cc = 0.001

rss_model = RSSDurativePDDL(domain_file=dom_file,prob_file=prob_file,
                            perform_scheduling=True,
                            duration_func=rss_duration_func,
                            time_window_func=rss_time_window_func,
                            max_steps=20,
                            verbose=1)

time_window = TemporalConstraint(start=rss_model.global_start_event,
                                 end=rss_model.global_end_event,
                                 ctype='controllable',lb=0.0,ub=2000.0)

b0 = rss_model.get_initial_belief(constraints=[time_window])

planner = RAOStar(rss_model,node_name='id',cc=cc,cc_type='overall',
                  terminal_prob=1.0,randomization=0.0,propagate_risk=True,
                  expand_all_open=True,verbose=1,log=False,animation=False)

policy,explicit,performance = planner.search(b0)

dot_policy = policy_to_dot(explicit,policy)
dot_policy.write('rss_policy.svg',format='svg')

rmpyl_policy = policy_to_rmpyl(explicit,policy,
                               constraint_fields=['constraints'],
                               global_end=rss_model.global_end_event)

#Combines controllable temporal constraints.
rmpyl_policy.simplify_temporal_constraints()

rmpyl_policy.to_ptpn(filename='rss_policy_rmpyl_before_stnu_reform.tpn',
                    exclude_op=['__stop__','__expand__'])

paris = PARIS()
risk_bound,sc_schedule = paris.stnu_reformulation(rmpyl_policy,makespan=True,cc=cc)

if risk_bound != None:
    risk_bound = min(risk_bound,1.0)
    print('\nSuccessfully performed STNU reformulation with scheduling risk %f %%!'%(risk_bound*100.0))
    rmpyl_policy.to_ptpn(filename='rss_policy_rmpyl_after_stnu_reform.tpn')

    print('\nThis is the schedule:')
    for e,t in sorted([(e,t) for e,t in sc_schedule.items()],key=lambda x: x[1]):
        print('\t%s: %.2f s'%(e,t))
else:
    print('\nFailed to perform STNU reformulation...')
