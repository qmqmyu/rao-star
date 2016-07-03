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
from rmpyl.rmpyl import RMPyL, Episode

class UAV:
    """Example Python UAV class that outputs Episodes as actions."""
    def __init__(self,name):
        self._name=name

    def fly(self,lb=3,ub=10):
        return Episode(duration={'ctype':'controllable','lb':lb,'ub':ub},
                       action='('+self._name+'-fly)')
    def scan(self,lb=1,ub=10):
        return Episode(duration={'ctype':'controllable','lb':lb,'ub':ub},
                       action='('+self._name+'-scan)')
    def crash(self):
        return Episode(action='('+self._name+'-crash)',terminal=True)

    def stop(self):
        return Episode(action='(stop)')

def rmpyl_uav():
    hello = UAV('hello')
    uav = UAV('uav')

    prog = RMPyL()
    # prog *= hello.fly()
    prog.plan = prog.sequence(
                    hello.scan(),
                    hello.fly(),
                    uav.fly(),
                    uav.scan(),
                    prog.decide({'name':'UAV-choice','domain':['Hello','UAV'],
                                 'utility':[5,7]},
                                 hello.fly(),
                                 uav.fly()))

    prog.add_overall_temporal_constraint(ctype='controllable',lb=0.0,ub=18.0)
    return prog

def rmpyl_nested_uav():
    hello = UAV('hello')
    uav = UAV('uav')

    prog = RMPyL()
    prog.plan = prog.sequence(
                    hello.scan(),
                    uav.scan(),
                    prog.decide(
                        {'name':'UAV-choice','domain':['Hello','UAV'],
                        'utility':[7,5]},
                        prog.sequence(
                            hello.fly(),
                            prog.observe(
                                {'name':'hello-success','domain':['Success','Failure'],
                                 'ctype':'probabilistic','probability':[0.8,0.2]},
                                prog.decide(
                                    {'name':'hello-assert-success',
                                     'domain':['Success'],
                                     'utility':[10]},
                                    hello.stop()),
                                prog.decide(
                                    {'name':'hello-assert-failure',
                                     'domain':['Failure'],
                                     'utility':[0]},
                                    hello.stop()))),
                         prog.sequence(
                            uav.fly(),
                            prog.observe(
                                {'name':'uav-success','domain':['Success','Failure'],
                                 'ctype':'probabilistic','probability':[0.95,0.05]},
                                prog.decide(
                                    {'name':'uav-assert-success',
                                     'domain':['Success'],
                                     'utility':[10]},
                                    uav.stop()),
                                prog.decide(
                                    {'name':'uav-assert-failure',
                                     'domain':['Failure'],
                                     'utility':[0]},
                                    uav.stop())))))
    return prog


def rmpyl_parallel_uav():
    hello = UAV('hello')
    uav = UAV('uav')

    prog = RMPyL()
    prog.plan = prog.parallel(
                    prog.sequence(
                        prog.decide({'name':'hello-action','domain':['Fly','Scan'],
                                     'utility':[0,1]},
                                     hello.fly(),
                                     hello.scan()),
                        prog.decide({'name':'hello-action','domain':['Fly','Scan'],
                                     'utility':[0,1]},
                                     hello.fly(),
                                     hello.scan()),
                        prog.decide({'name':'hello-action','domain':['Fly','Scan'],
                                     'utility':[0,1]},
                                     hello.fly(),
                                     hello.scan())
                    ),
                    prog.sequence(
                        prog.decide({'name':'uav-action','domain':['Fly','Scan'],
                                     'utility':[0,1]},
                                     uav.fly(),
                                     uav.scan()),
                        prog.decide({'name':'uav-action','domain':['Fly','Scan'],
                                     'utility':[0,1]},
                                     uav.fly(),
                                     uav.scan()),
                        prog.decide({'name':'uav-action','domain':['Fly','Scan'],
                                     'utility':[0,1]},
                                     uav.fly(),
                                     uav.scan())

                    ))
    return prog

# prog = rmpyl_uav()
# prog = rmpyl_nested_uav()
prog = rmpyl_parallel_uav()

prog.to_ptpn(filename='rmpyl_input_ptpn.tpn')

rmpyl_model = BaseRMPyLUnraveler()
b0 = rmpyl_model.get_initial_belief(prog)

planner = RAOStar(rmpyl_model,node_name='id',cc=0.0,cc_type='overall',
                  terminal_prob=1.0,randomization=0.0,propagate_risk=True,
                  verbose=1)

policy,explicit,performance = planner.search(b0)

dot_policy = policy_to_dot(explicit,policy)
dot_policy.write('rmpyl_unravel_policy.svg',format='svg')

rmpyl_policy = policy_to_rmpyl(explicit,policy)
rmpyl_policy.to_ptpn(filename='rmpyl_unravel_policy_ptpn.tpn')
