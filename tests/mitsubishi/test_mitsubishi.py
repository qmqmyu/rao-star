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

Demo of RAO* being used to generate plans for the Mitsubishi demo.

@author: Pedro Santana (psantana@mit.edu).
"""
from rao.raostar import RAOStar
from rao.export import policy_to_dot,policy_to_rmpyl
from rao.models.mitsubishi import SimpleFaultMitsubishi,DiagnosticMitsubishi

#Creates the MDP model for the Mitsubishi demo
dom_file = 'mitsubishi-domain-strips.pddl'
prob_file = 'mitsubishi-problem.pddl'
#prob_file = 'mitsubishi-problem-pick-and-place.pddl'

##No action failure (deterministic PDDL)
# mitsu_model = SimpleFaultMitsubishi(domain_file=dom_file,prob_file=prob_file,
#                                           time_limit=-1,p_fail=0.0,verbose=0)

##Action can fail by being no-ops.
# mitsu_model = SimpleFaultMitsubishi(domain_file=dom_file,prob_file=prob_file,
#                                     time_limit=5,drop_penalty=10.0,p_fail=0.1,verbose=0)


# Diagnostic demo, where the robot has to figure out if it is fit for the task

# With this setting of parameters, the robot tries an action a couple of times
# before bugging the human for help.
param_dict = {'p_fail_fit':0.2,      #Prob. of failing a task, while being fit
               'p_fail_unfit':0.999,  #Prob. of failing a task, while not being fit
               'p_fit_fit':1.0,       #Prob. of remaining fit, if fit before
               'p_fit_unfit':0.0,     #Prob. becoming fit, if unfit before
               'goal_drop_penalty':100.0, #Penalty for not achieving a goal
               'robot_action_cost':1.0, #Cost of robot performing an action
               'human_action_cost':5.0} #Cost of human performing an action

mitsu_model = DiagnosticMitsubishi(domain_file=dom_file,prob_file=prob_file,
                                    time_limit=5,verbose=1,param_dict=param_dict)

b0 = mitsu_model.get_initial_belief()

planner = RAOStar(mitsu_model,node_name='id',cc=0.3,cc_type='overall',
                  terminal_prob=1.0,randomization=0.0,propagate_risk=True,
                  verbose=1,log=False)

policy,explicit,performance = planner.search(b0)
dot_policy = policy_to_dot(explicit,policy)
rmpyl_policy = policy_to_rmpyl(explicit,policy)

dot_policy.write('mitsubishi_policy.svg',format='svg')
rmpyl_policy.to_ptpn(filename='mitsubishi_policy_rmpyl_ptpn.tpn')
