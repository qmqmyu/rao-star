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

Demo of RAO* being used to generate plans for the Mitsubishi demo with durative
actions.

@author: Pedro Santana (psantana@mit.edu).
"""
from rao.raostar import RAOStar
from rao.export import policy_to_dot,policy_to_rmpyl
from rao.models.pddlmodel import DurativePDDL
from rmpyl.constraints import TemporalConstraint
from pytemporal.paris import PARIS

#PDDL files for the Mitsubishi demo
dom_file = 'mitsubishi-domain-strips.pddl'
prob_file = 'mitsubishi-problem.pddl'

cc = 0.1

def mitsubishi_duration_func(durative_pddl_action):
    """
    Function that generates the temporal durations for different PDDL actions.
    """
    act_tokens = durative_pddl_action.strip('() ').split(' ')
    manip = act_tokens[2].lower()

    if manip.find('baxter'):
        return {'ctype':'uncontrollable_probabilistic',
                'distribution':{'type':'gaussian','mean':8.0,'variance':4.0}}
    else:
        return {'ctype':'uncontrollable_probabilistic',
                'distribution':{'type':'uniform','lb':2.0,'ub':5.0}}

mitsubishi_model = DurativePDDL(domain_file=dom_file,prob_file=prob_file,
                                perform_scheduling=True,
                                duration_func=mitsubishi_duration_func,
                                max_steps=20,
                                verbose=1)

time_window = TemporalConstraint(start=mitsubishi_model.global_start_event,
                                 end=mitsubishi_model.global_end_event,
                                 ctype='controllable',lb=0.0,ub=2000.0)

b0 = mitsubishi_model.get_initial_belief(constraints=[time_window])

planner = RAOStar(mitsubishi_model,node_name='id',cc=cc,cc_type='overall',
                  terminal_prob=1.0,randomization=0.0,propagate_risk=True,
                  expand_all_open=True,verbose=1,log=False,animation=False)

policy,explicit,performance = planner.search(b0)

dot_policy = policy_to_dot(explicit,policy)
dot_policy.write('mitsubishi_durative_policy.svg',format='svg')

rmpyl_policy = policy_to_rmpyl(explicit,policy,
                               constraint_fields=['constraints'],
                               global_end=mitsubishi_model.global_end_event)

#Combines controllable temporal constraints.
rmpyl_policy.simplify_temporal_constraints()

rmpyl_policy.to_ptpn(filename='mitsubishi_durative_policy_rmpyl_before_stnu_reform.tpn',
                    exclude_op=['__stop__','__expand__'])

paris = PARIS()
risk_bound,sc_schedule = paris.stnu_reformulation(rmpyl_policy,makespan=True,cc=cc)

if risk_bound != None:
    risk_bound = min(risk_bound,1.0)
    print('\nSuccessfully performed STNU reformulation with scheduling risk %f %%!'%(risk_bound*100.0))
    rmpyl_policy.to_ptpn(filename='mitsubishi_durative_policy_rmpyl_after_stnu_reform.tpn')

    print('\nThis is the schedule:')
    for e,t in sorted([(e,t) for e,t in sc_schedule.items()],key=lambda x: x[1]):
        print('\t%s: %.2f s'%(e,t))
else:
    print('\nFailed to perform STNU reformulation...')
