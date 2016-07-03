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

Demo of RAO* being used to generate plans from an RMPyL program.

@author: Pedro Santana (psantana@mit.edu).
"""
from rao.raostar import RAOStar
from rao.export import policy_to_dot,policy_to_rmpyl
from rao.models.rmpylmodel import BaseRMPyLUnraveler,StrongStrongRMPyLUnraveler
from rmpyl.rmpyl import RMPyL, Episode

def rmpyl_icaps14():
    """
    Example from (Santana & Williams, ICAPS14).
    """
    prog = RMPyL()
    prog *= prog.decide(
                {'name':'transport-choice','domain':['Bike','Car','Stay'],
                 'utility':[100,70,0]},
                 prog.observe(
                    {'name':'slip','domain':[True,False],
                     'ctype':'probabilistic','probability':[0.051,1.0-0.051]},
                     prog.sequence(Episode(action='(ride-bike)',
                                           duration={'ctype':'controllable','lb':15,'ub':25}),
                                   Episode(action='(change)',
                                           duration={'ctype':'controllable','lb':20,'ub':30})),
                      Episode(action='(ride-bike)',duration={'ctype':'controllable','lb':15,'ub':25})),
                 prog.observe(
                    {'name':'accident','domain':[True,False],
                     'ctype':'probabilistic','probability':[0.013,1.0-0.013]},
                     prog.sequence(Episode(action='(tow-vehicle)',
                                          duration={'ctype':'controllable','lb':30,'ub':90}),
                                  Episode(action='(cab-ride)',
                                          duration={'ctype':'controllable','lb':10,'ub':20})),
                     Episode(action='(drive)',duration={'ctype':'controllable','lb':10,'ub':20})),
                 Episode(action='(stay)'))

    prog.add_overall_temporal_constraint(ctype='controllable',lb=0.0,ub=30.0)
    return prog


def rmpyl_probabilistic_icaps14():
    """
    Example from (Santana & Williams, ICAPS14).
    """
    prog = RMPyL()
    prog *= prog.decide(
                {'name':'transport-choice','domain':['Bike','Car','Stay'],
                 'utility':[100,70,0]},
                 prog.observe(
                    {'name':'slip','domain':[True,False],
                     'ctype':'probabilistic','probability':[0.051,1.0-0.051]},
                     prog.sequence(Episode(action='(ride-bike)',
                                           duration={'ctype':'uncontrollable_probabilistic',
                                                     'distribution':{'type':'gaussian',
                                                                     'mean':20.0,
                                                                     'variance':5.0}}),
                                   Episode(action='(change)',
                                           duration={'ctype':'uncontrollable_probabilistic',
                                                     'distribution':{'type':'gaussian',
                                                                     'mean':25.0,
                                                                     'variance':3.0}})),
                      Episode(action='(ride-bike)',duration={'ctype':'uncontrollable_probabilistic',
                                                             'distribution':{'type':'gaussian',
                                                                             'mean':20.0,
                                                                             'variance':5.0}})),
                 prog.observe(
                    {'name':'accident','domain':[True,False],
                     'ctype':'probabilistic','probability':[0.013,1.0-0.013]},
                     prog.sequence(Episode(action='(tow-vehicle)',
                                          duration={'ctype':'uncontrollable_probabilistic',
                                                    'distribution':{'type':'gaussian',
                                                                    'mean':60.0,
                                                                    'variance':20.0}}),
                                  Episode(action='(cab-ride)',
                                          duration={'ctype':'uncontrollable_probabilistic',
                                                    'distribution':{'type':'gaussian',
                                                                     'mean':15.0,
                                                                     'variance':5.0}})),
                     Episode(action='(drive)',duration={'ctype':'uncontrollable_probabilistic',
                                                        'distribution':{'type':'gaussian',
                                                                        'mean':15.0,
                                                                        'variance':8.0}})),
                 Episode(action='(stay)'))

    prog.add_overall_temporal_constraint(ctype='controllable',lb=0.0,ub=30.0)
    return prog

#prog = rmpyl_icaps14()
prog = rmpyl_probabilistic_icaps14()

prog.to_ptpn(filename='rmpyl_icaps14_ptpn.tpn')

# rmpyl_model = BaseRMPyLUnraveler()
rmpyl_model = StrongStrongRMPyLUnraveler(verbose=1)

b0 = rmpyl_model.get_initial_belief(prog)

planner = RAOStar(rmpyl_model,node_name='id',cc=0.05,cc_type='overall',
                  terminal_prob=1.0,randomization=0.0,propagate_risk=True,
                  verbose=1)

policy,explicit,performance = planner.search(b0)

dot_policy = policy_to_dot(explicit,policy)
dot_policy.write('rmpyl_icaps14_policy.svg',format='svg')

rmpyl_policy = policy_to_rmpyl(explicit,policy)
rmpyl_policy.to_ptpn(filename='rmpyl_icaps14_policy_ptpn.tpn')
