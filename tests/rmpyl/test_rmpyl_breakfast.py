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
from rmpyl.constraints import TemporalConstraint

def rmpyl_breakfast():
    """
    Example from (Levine & Williams, ICAPS14).
    """
    #Actions that Alice performs
    get_mug_ep = Episode(action='(get alice mug)',duration={'ctype':'controllable','lb':0.5,'ub':1.0})
    get_glass_ep = Episode(action='(get alice glass)',duration={'ctype':'controllable','lb':0.5,'ub':1.0})

    make_cofee_ep = Episode(action='(make-coffee alice)',duration={'ctype':'controllable','lb':3.0,'ub':5.0})
    pour_cofee_ep = Episode(action='(pour-coffee alice mug)',duration={'ctype':'controllable','lb':0.5,'ub':1.0})
    pour_juice_glass = Episode(action='(pour-juice alice glass)',duration={'ctype':'controllable','lb':0.5,'ub':1.0})

    get_bagel_ep = Episode(action='(get alice bagel)',duration={'ctype':'controllable','lb':0.5,'ub':1.0})
    get_cereal_ep = Episode(action='(get alice cereal)',duration={'ctype':'controllable','lb':0.5,'ub':1.0})

    toast_bagel_ep = Episode(action='(toast alice bagel)',duration={'ctype':'controllable','lb':3.0,'ub':5.0})
    add_cheese_bagel_ep = Episode(action='(add-cheese alice bagel)',duration={'ctype':'controllable','lb':1.0,'ub':2.0})
    mix_cereal_ep = Episode(action='(mix-cereal alice milk)',duration={'ctype':'controllable','lb':1.0,'ub':2.0})

    #Actions that the robot performs
    get_grounds_ep = Episode(action='(get grounds robot)',duration={'ctype':'controllable','lb':0.5,'ub':1.0})
    get_juice_ep = Episode(action='(get juice robot)',duration={'ctype':'controllable','lb':0.5,'ub':1.0})
    get_milk_ep = Episode(action='(get milk robot)',duration={'ctype':'controllable','lb':0.5,'ub':1.0})
    get_cheese_ep = Episode(action='(get cheese robot)',duration={'ctype':'controllable','lb':0.5,'ub':1.0})

    prog = RMPyL()
    prog *= prog.sequence(
                prog.parallel(
                    prog.observe(
                        {'name':'observe-utensil','domain':['Mug','Glass'],'ctype':'uncontrollable'},
                        get_mug_ep,
                        get_glass_ep,
                        id='observe-utensil-ep'),
                    prog.decide(
                        {'name':'choose-beverage-ingredient','domain':['Grounds','Juice'],'utility':[0,0]},
                        get_grounds_ep,
                        get_juice_ep,
                        id='choose-beverage-ingredient-ep')),
                prog.observe(
                    {'name':'observe-alice-drink','domain':['Coffee','Juice'],'ctype':'uncontrollable'},
                    prog.sequence(make_cofee_ep,pour_cofee_ep),
                    pour_juice_glass,
                    id='observe-alice-drink-ep'),
                prog.parallel(
                    prog.observe(
                        {'name':'observe-food','domain':['Bagel','Cereal'],'ctype':'uncontrollable'},
                        get_bagel_ep,
                        get_cereal_ep,
                        id='observe-food-ep'),
                    prog.decide(
                        {'name':'choose-food-ingredient','domain':['Milk','Cheese'],'utility':[0,0]},
                        get_milk_ep,
                        get_cheese_ep,
                        id='choose-food-ingredient-ep'),
                    id='parallel-food-ep'),
                prog.observe(
                    {'name':'observe-alice-food','domain':['Bagel','Cereal'],'ctype':'uncontrollable'},
                    prog.sequence(toast_bagel_ep,add_cheese_bagel_ep),
                    mix_cereal_ep),
                id='breakfast-sequence')

    extra_tcs = [TemporalConstraint(start=prog.episode_by_id('breakfast-sequence').start,
                                  end=prog.episode_by_id('observe-utensil-ep').start,
                                  ctype='controllable',lb=0.0,ub=0.0),
                 TemporalConstraint(start=prog.episode_by_id('breakfast-sequence').start,
                                    end=prog.episode_by_id('choose-beverage-ingredient-ep').start,
                                    ctype='controllable',lb=0.2,ub=0.3),
                 TemporalConstraint(start=prog.episode_by_id('parallel-food-ep').start,
                                    end=prog.episode_by_id('observe-food-ep').start,
                                    ctype='controllable',lb=0.0,ub=0.0),
                 TemporalConstraint(start=prog.episode_by_id('parallel-food-ep').start,
                                    end=prog.episode_by_id('choose-food-ingredient-ep').start,
                                    ctype='controllable',lb=0.2,ub=0.3)]

    for tc in extra_tcs:
        prog.add_temporal_constraint(tc)

    prog.add_overall_temporal_constraint(ctype='controllable',lb=0.0,ub=7.0)
    prog.simplify_temporal_constraints()

    return prog

prog = rmpyl_breakfast()
prog.to_ptpn(filename='rmpyl_breakfast_input_ptpn.tpn')

# rmpyl_model = BaseRMPyLUnraveler()
#
# b0 = rmpyl_model.get_initial_belief(prog)
#
# planner = RAOStar(rmpyl_model,node_name='id',cc=1.0,cc_type='overall',
#                   terminal_prob=1.0,randomization=0.0,propagate_risk=True,
#                   verbose=2)
#
# policy,explicit,performance = planner.search(b0)
#
# dot_policy = policy_to_dot(explicit,policy)
# dot_policy.write('rmpyl_breakfast_policy.svg',format='svg')
#
# rmpyl_policy = policy_to_rmpyl(explicit,policy)
# rmpyl_policy.to_ptpn(filename='rmpyl_breakfast_collaborative_policy_ptpn.tpn')
