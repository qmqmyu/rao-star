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
from rmpyl.rmpyl import RMPyL,Episode,sequence_composition
from rmpyl.rmpyl import ChanceConstraint,TemporalConstraint
import pickle
import math

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

    def go_to(self,start,goal,risk,ep_id=None,waypoints=10,time_horizon=200.0):
        """
        Returns the episode corresponding to the vehicle traveling.
        """
        return self._go_to_with_waypoints(start,goal,risk,ep_id,waypoints,time_horizon)
        #return self._simple_go_to(start,goal,ep_id)

    def _simple_go_to(self,start,goal,ep_id=None,duration_type='uncontrollable_probabilistic'):
        """
        Returns the episode corresponding to the vehicle traveling.
        """
        line_dist = math.sqrt(sum([(xi-xj)**2 for xi,xj in zip(goal,start)]))
        dist = line_dist*1.2
        lb_duration = dist/2.0
        ub_duration = dist/1.1

        if duration_type=='uncontrollable_bounded':
            duration_dict={'ctype':'uncontrollable_bounded',
                           'lb':lb_duration,'ub':ub_duration}
        elif duration_type=='uncontrollable_probabilistic':
            duration_dict={'ctype':'uncontrollable_probabilistic',
                           'distribution':{'type':'uniform','lb':lb_duration,'ub':ub_duration}}
        elif duration_type=='no_constraint':
            duration_dict={'ctype':'controllable','lb':0.0,'ub':float('inf')}
        else:
            raise ValueError('Duration type %s currently not supported.'%duration_type)

        goto_ep = Episode(duration=duration_dict,
                               action='(go-from-to %s %s %s)'%(self.name,str(start),str(goal)),
                               distance=dist)

        if ep_id!=None:
            goto_ep.id=ep_id
        return goto_ep

    def _go_to_with_waypoints(self,start,goal,risk,ep_id=None,waypoints=10,time_horizon=200.0):
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
        goto_ep = self.path_planner.plan_episode(start_state=start+(0.0,0.0),
                                              goal_state=goal+(0.0,0.0),
                                              parameters=self.rover_param,
                                              stoch_model=stoch_model,
                                              duration_type='gaussian',
                                              agent=self.name)
        if ep_id!=None:
            goto_ep.id=ep_id
        return goto_ep

    def perform_science(self,ep_id=None):
        """
        Returns the episode corresponding to the vehicle performing science experiments.
        """
        ps_ep = sequence_composition(
                    Episode(duration={'ctype':'uncontrollable_bounded','lb':9,'ub':11},
                            action='(drill %s)'%(self.name)),
                    Episode(duration={'ctype':'uncontrollable_bounded','lb':10,'ub':15},
                            action='(collect %s)'%(self.name)),
                    Episode(duration={'ctype':'controllable','lb':5,'ub':30},
                            action='(process %s)'%(self.name)))
        if ep_id!=None:
            ps_ep.id=ep_id
        return ps_ep

    def relay(self,ep_id=None):
        """
        Returns the episode representing the rover sending data back to a satellite.
        """
        rel_ep= Episode(duration={'ctype':'controllable','lb':5,'ub':30},
                       action='(relay %s)'%(self.name))
        if ep_id!=None:
            rel_ep.id=ep_id
        return rel_ep

loc={'start':(8.751,-8.625),
     'minerals':(0.0,-10.0),
     'funny_rock':(-5.0,-2.0),
     'relay':(0.0,0.0),
     'alien_lair':(0.0,10.0)}


rov1 = Rover(name='spirit')
rov2 = Rover(name='opportunity')

prog = RMPyL(name='run()')
prog *= prog.parallel(
            prog.sequence(
                rov1.go_to(start=loc['start'],goal=loc['minerals'],risk=0.01),
                rov1.perform_science(),
                rov1.go_to(start=loc['minerals'],goal=loc['funny_rock'],risk=0.01),
                rov1.perform_science(),
                rov1.go_to(start=loc['funny_rock'],goal=loc['relay'],risk=0.01),
                rov1.relay(ep_id=rov1.name+'_relay')),
            prog.sequence(
                rov2.go_to(start=loc['start'],goal=loc['alien_lair'],risk=0.01),
                rov2.perform_science(),
                rov2.go_to(start=loc['alien_lair'],goal=loc['relay'],risk=0.01),
                rov2.relay(ep_id=rov2.name+'_relay')))

r1_rel = prog.episode_by_id(rov1.name+'_relay')
r2_rel = prog.episode_by_id(rov2.name+'_relay')

tc_relay = TemporalConstraint(start=r1_rel.end,end=r2_rel.start,
                              ctype='controllable',lb=0.0,ub=10.0)
prog.add_temporal_constraint(tc_relay)

tc=prog.add_overall_temporal_constraint(ctype='controllable',lb=1800.0,ub=2000.0)
cc_time = ChanceConstraint(constraint_scope=[tc,tc_relay],risk=0.1)
prog.add_chance_constraint(cc_time)

#Option to export the RMPyL program to an Enterprise-compliant TPN.
prog.to_ptpn(filename='picard_two_rovers_rmpyl.tpn')

#Writes RMPyL program to pickle file.
with open('picard_two_rovers_rmpyl.pickle','wb') as f:
    print('Writing RMPyL program to pickle file.')
    pickle.dump(prog,f)
