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
from pytemporal.paris import PARIS
from rss_model_utils import rss_duration_func
from rmpyl.rmpyl import RMPyL
from rmpyl.episodes import Episode
from rmpyl.defs import Event
import time

#PDDL files for the RSS demo
dom_file = 'rss-domain-strips.pddl'
prob_file = 'rss-current-problem-strips.pddl'

py_sat = PySAT(dom_file,prob_file,precompute_steps=20,remove_static=True,
               write_dimacs=False,verbose=True)

domain,problem,task = model_parser(dom_file,prob_file,remove_static=True)

start = time.time()
sat_plans = py_sat.plan(task.initial_state,task.goals,time_steps=18)
elapsed = time.time()-start
print('\n##### All solving took %.4f s'%(elapsed))

if len(sat_plans)>0:
    plan = sat_plans[0]
    print('\n##### Plan found!\n')
    for t,action in enumerate(plan):
        print('%d: %s'%(t,action))

    prog = RMPyL(name='run()')
    prog.plan = prog.sequence(*[Episode(start=Event(name='start-of-'+op),
                                        end=Event(name='end-of-'+op),
                                        action=op,
                                        duration=rss_duration_func(op)) for op in plan])
    prog.add_overall_temporal_constraint(ctype='controllable',lb=0.0,ub=2000.0)
    prog.to_ptpn(filename='rss_pysat_before_stnu_reform.tpn')

    paris = PARIS()
    risk_bound,sc_sched = paris.stnu_reformulation(prog,makespan=True,cc=0.001)
    if risk_bound != None:
        risk_bound = min(risk_bound,1.0)
        print('\nSuccessfully performed STNU reformulation with scheduling risk %f %%!'%(risk_bound*100.0))
        prog.to_ptpn(filename='rss_pysat_after_stnu_reform.tpn')

        print('\nThis is the schedule:')
        for e,t in sorted([(e,t) for e,t in sc_sched.items()],key=lambda x: x[1]):
            print('\t%s: %.2f s'%(e,t))
    else:
        print('\nFailed to perform STNU reformulation...')
else:
    print('\n##### No plan found.')
