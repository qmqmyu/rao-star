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
Demo showing the integration between pySulu (pSulu+RMPyL), Pike, and the RSS stuff.

@author: Pedro Santana (psantana@mit.edu).
"""
from pysulu.psulu_rmpyl import PySuluRMPyL
from rmpyl.rmpyl import RMPyL,Episode,ChanceConstraint,sequence_composition
import pickle

class Rover(object):
    """
    Simple RMPyL model for a Mars rover.
    """
    def __init__(self,name):
        self.name=name
        self.path_planner = PySuluRMPyL()
        self.rover_param = {}
        self.rover_param['executor']='ProOFCSA'    #Solving algorithm
        self.rover_param['max_velocity']=0.4       #Agent's maximum velocity (in m/s)
        self.rover_param['save png']=0             #Save map to file

    def go_to(self,start,goal,risk,waypoints=10,time_horizon=200.0):
        """
        Returns the episode corresponding to the vehicle traveling.
        """
        self.rover_param['chance_constraint']=risk
        self.rover_param['waypoints']=waypoints
        self.rover_param['time_horizon']=time_horizon

        init_pos_var=0.0
        process_pos_var=0.1
        stoch_model = self.path_planner.simple_stochastic_model(self.rover_param,
                                                                init_pos_var,
                                                                process_pos_var,dim=2)

        return self.path_planner.plan_episode(start_state=start+(0.0,0.0),
                                              goal_state=goal+(0.0,0.0),
                                              parameters=self.rover_param,
                                              stoch_model=stoch_model,
                                              duration_type='gaussian',
                                              agent=self.name)


    def perform_science(self):
        """
        Returns the episode corresponding to the vehicle performing science experiments.
        """
        return sequence_composition(
                Episode(duration={'ctype':'uncontrollable_bounded','lb':9,'ub':11},
                        action='(drill %s)'%(self.name)),
                Episode(duration={'ctype':'uncontrollable_bounded','lb':10,'ub':15},
                        action='(collect %s)'%(self.name)),
                Episode(duration={'ctype':'controllable','lb':5,'ub':30},
                        action='(process %s)'%(self.name)))

    def relay(self):
        """
        Returns the episode representing the rover sending data back to a satellite.
        """
        return Episode(duration={'ctype':'controllable','lb':5,'ub':30},
                       action='(relay %s)'%(self.name))

loc={'start':(8.751,-8.625),
     'minerals':(0.0,-10.0),
     'funny_rock':(-5.0,-2.0),
     'relay':(0.0,0.0),
     'alien_lair':(0.0,10.0)}


rov1 = Rover(name='spirit')

prog = RMPyL(name='run()')#name=run() is a requirement for Enterprise at the moment
prog *= prog.sequence(
            rov1.go_to(start=loc['start'],goal=loc['minerals'],risk=0.01),
            rov1.go_to(start=loc['minerals'],goal=loc['funny_rock'],risk=0.01),
            rov1.go_to(start=loc['funny_rock'],goal=loc['alien_lair'],risk=0.01),
            rov1.go_to(start=loc['alien_lair'],goal=loc['relay'],risk=0.01))
tc=prog.add_overall_temporal_constraint(ctype='controllable',lb=0.0,ub=2000.0)
cc_time = ChanceConstraint(constraint_scope=[tc],risk=0.1)
prog.add_chance_constraint(cc_time)

#Option to export the RMPyL program to an Enterprise-compliant TPN.
prog.to_ptpn(filename='picard_rovers_rmpyl.tpn')

#Writes RMPyL program to pickle file.
with open('picard_rovers_rmpyl.pickle','wb') as f:
    print('Writing RMPyL program to pickle file.')
    pickle.dump(prog,f)
