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

Implementation of pSulu as a RAO* module.

@author: Pedro Santana (psantana@mit.edu).
"""
from .rock_sample import CCRockSample,tCCRockSample,FollowPathAction
from numpy import linalg as la
from rmpyl.rmpyl import RMPyL
from rmpyl.episodes import Episode
from rmpyl.constraints import TemporalConstraint

def _fake_plan_path(sites,start_site,goal_site,risk,velocities,duration_type,agent,**kwargs):
    """
    Fakes a chance-constrained path from a start location to a goal location.
    Specific parameters are given as keyword arguments.
    """
    start = sites[start_site]['coords']
    goal = sites[goal_site]['coords']

    line_dist = la.norm(goal-start)
    dist = line_dist*1.2 if risk<0.005 else line_dist*1.1
    lb_duration = dist/velocities['max']
    ub_duration = dist/velocities['avg']

    if duration_type=='uncontrollable_bounded':
        duration_dict={'ctype':'uncontrollable_bounded',
                       'lb':lb_duration,'ub':ub_duration}
    elif duration_type=='uniform':
        duration_dict={'ctype':'uncontrollable_probabilistic',
                       'distribution':{'type':'uniform','lb':lb_duration,'ub':ub_duration}}
    elif duration_type=='gaussian':
        duration_dict={'ctype':'uncontrollable_probabilistic',
                       'distribution':{'type':'gaussian',
                                       'mean':(lb_duration+ub_duration)/2.0,
                                       'variance':((ub_duration-lb_duration)**2)/36.0}}
    elif duration_type=='no_constraint':
        duration_dict={'ctype':'controllable','lb':0.0,'ub':float('inf')}
    else:
        raise ValueError('Duration type %s currently not supported in Fake Planner.'%duration_type)

    path_episode = Episode(duration=duration_dict,
                           action='(go-from-to %s %s %s)'%(agent,start_site,goal_site),
                           distance=dist,**kwargs)
    path_episode.properties['distance']=dist
    path_episode.properties['start_coords']=start
    path_episode.properties['goal_coords']=goal
    prog_path = RMPyL(); prog_path.plan = path_episode

    return prog_path


class FakePlannerRockSampleModel(CCRockSample):
    """
    Implements a fake chance-constrained path planner, that does not take into
    account the obstacles in the environment. It was created to remove the overhead
    of having to call pSulu all the time and cache the solutions, and also due to
    the fact that pSulu returns constant horizon solutions.
    """
    def __init__(self,sites,duration_type='uncontrollable_bounded',path_risks=[0.001,0.01],
                 prob_discovery=0.95,velocities={'max':1.0,'avg':0.5},name='robot',verbose=0):
        super(FakePlannerRockSampleModel,self).__init__(None,sites,prob_discovery,name,verbose)

        #Different risks the agent is allowed to take in its path
        self.path_risks = path_risks
        #Type of temporal duration used to represented traversal times
        self.duration_type=duration_type
        #Velocity dictionary for estimating temporal durations.
        self.velocities=velocities

    def actions(self,state,**kwargs):
        """
        The actions available at a state correspond to driving to different
        discovery sites.
        """
        if not self.is_terminal(state):
            curr_site=state['site']
            not_visited = [s for s in self.science_sites if (not s in state['visited']) and s!=curr_site]
            actions=[]
            for goal_site in not_visited:
                for r in self.path_risks:
                    path_rmpyl=self._plan_cc_path(curr_site,goal_site,r,**kwargs)
                    actions.append(FollowPathAction(rmpyl=path_rmpyl,
                                                    episode=path_rmpyl.plan,
                                                    start_site=curr_site,
                                                    goal_site=goal_site,
                                                    risk=r))
            return actions
        else:
            return []

    def _plan_cc_path(self,start_site,goal_site,risk,**kwargs):
        """
        Fakes a chance-constrained path from a start location to a goal location.
        Specific parameters are given as keyword arguments.
        """
        return _fake_plan_path(self.sites,start_site,goal_site,risk,
                               self.velocities,self.duration_type,self.name,**kwargs)


class tFakePlannerRockSampleModel(tCCRockSample):
    """
    Implements tCCRockSample using a fake path planner.
    """
    def __init__(self,sites,perform_scheduling=True,duration_type='uncontrollable_bounded',
                 path_risks=[0.001,0.01],prob_discovery=0.95,
                 velocities={'max':1.0,'avg':0.5},name='robot',paris_params={},verbose=0):

        super(tFakePlannerRockSampleModel,self).__init__(path_planner=None,sites=sites,
                                                         perform_scheduling=perform_scheduling,
                                                         prob_discovery=prob_discovery,
                                                         name=name,
                                                         paris_params=paris_params,
                                                         verbose=verbose)

        #Different risks the agent is allowed to take in its path
        self.path_risks = path_risks
        #Type of temporal duration used to represented traversal times
        self.duration_type=duration_type
        #Velocity dictionary for estimating temporal durations.
        self.velocities=velocities

    def _plan_cc_path(self,start_site,goal_site,risk,**kwargs):
        """
        Fakes a chance-constrained path from a start location to a goal location.
        Specific parameters are given as keyword arguments.
        """
        return _fake_plan_path(self.sites,start_site,goal_site,risk,
                               self.velocities,self.duration_type,self.name,**kwargs)

    def actions(self,state,**kwargs):
        """
        The actions available at a state correspond to driving to different
        discovery sites.
        """
        if not self.is_terminal(state):
            curr_site=state['site']
            not_visited = [s for s in self.science_sites if (not s in state['visited']) and s!=curr_site]
            actions=[]
            for goal_site in not_visited:
                for r in self.path_risks:
                    #Trajectory episode
                    path_rmpyl=self._plan_cc_path(curr_site,goal_site,r)

                    if curr_site == self.start_site:
                        #Add [0,0] temporal constraints to force consistence
                        tc = TemporalConstraint(start=self.global_start_event,
                                                end=path_rmpyl.first_event,
                                                ctype='controllable',lb=0.0, ub=0.0)
                        path_rmpyl.add_temporal_constraint(tc)
                    elif goal_site == self.goal:
                        #Add [0,0] temporal constraints to force consistence
                        tc = TemporalConstraint(start=path_rmpyl.last_event,
                                                end=self.global_end_event,
                                                ctype='controllable',lb=0.0, ub=0.0)
                        path_rmpyl.add_temporal_constraint(tc)
                    else:
                        pass

                    actions.append(FollowPathAction(rmpyl=path_rmpyl,
                                                    episode=path_rmpyl.plan,
                                                    start_site=curr_site,
                                                    goal_site=goal_site,
                                                    risk=r))
            return actions
        else:
            return []




#DEPRECATED

# class FakePlannerRockSampleModel(CCRockSample):
#     """
#     Implements a fake chance-constrained path planner, that does not take into
#     account the obstacles in the environment. It was created to remove the overhead
#     of having to call pSulu all the time and cache the solutions, and also due to
#     the fact that pSulu returns constant horizon solutions.
#     """
#     def __init__(self,sites,duration_type='uniform',path_risks=[0.001,0.01],
#                  prob_discovery=0.95,velocities={'max':1.0,'avg':0.5},verbose=0):
#         super(FakePlannerRockSampleModel,self).__init__(None,sites,prob_discovery,verbose)
#
#         #Different risks the agent is allowed to take in its path
#         self.path_risks = path_risks
#
#         #Type of temporal duration used to represented traversal times
#         self.duration_type=duration_type
#
#         #Velocity dictionary for estimating temporal durations.
#         self.velocities=velocities
#
#     def actions(self,state,**kwargs):
#         """
#         The actions available at a state correspond to driving to different
#         discovery sites.
#         """
#         if not self.is_terminal(state):
#             curr_site=state['site']
#             not_visited = [s for s in self.science_sites if (not s in state['visited']) and s!=curr_site]
#             actions=[]
#             for goal_site in not_visited:
#                 for r in self.path_risks:
#                     path_rmpyl=self._plan_cc_path(curr_site,goal_site,r,**kwargs)
#                     actions.append(FollowPathAction(rmpyl=path_rmpyl,
#                                                     episode=path_rmpyl.plan,
#                                                     start_site=curr_site,
#                                                     goal_site=goal_site,
#                                                     risk=r))
#             return actions
#         else:
#             return []
#
#     def _plan_cc_path(self,start_site,goal_site,risk,**kwargs):
#         """
#         Fakes a chance-constrained path from a start location to a goal location.
#         Specific parameters are given as keyword arguments.
#         """
#         start = self.sites[start_site]['coords']
#         goal = self.sites[goal_site]['coords']
#
#         line_dist = la.norm(goal-start)
#         dist = line_dist*1.2 if risk<0.005 else line_dist*1.1
#         lb_duration = dist/self.velocities['max']
#         ub_duration = dist/self.velocities['avg']
#
#         duration_dict={'ctype':'uncontrollable_bounded',
#                        'lb':lb_duration,'ub':ub_duration}
#
#         # duration_dict={'ctype':'uncontrollable_probabilistic',
#         #                'distribution':{'type':'uniform','lb':lb_duration,'ub':ub_duration}}
#
#         path_episode = Episode(duration=duration_dict,
#                                action='(go-from-to %s %s)'%(start_site,goal_site),
#                                distance=dist,**kwargs)
#         path_episode.properties['distance']=dist
#         path_episode.properties['start_coords']=start
#         path_episode.properties['goal_coords']=goal
#         prog_path = RMPyL(); prog_path.plan = path_episode
#
#         return prog_path

# class tFakePlannerRockSampleModel(FakePlannerRockSampleModel):
#     """
#     Extension of the FakePlannerRockSampleModel model that also requires the policies
#     to be strongly controllable in a probabilistic sense. Uses PyTemporal
#     for checking.
#     """
#     def __init__(self,sites,perform_scheduling=True,duration_type='uniform',
#                  path_risks=[0.001,0.01],prob_discovery=0.95,
#                  velocities={'max':1.0,'avg':0.5},verbose=0):
#         super(tFakePlannerRockSampleModel,self).__init__(sites,duration_type,path_risks,prob_discovery,velocities,verbose)
#
#         #Events used to enforce global durations
#         self.global_start_event = Event(name='depart-from-start')
#         self.global_end_event = Event(name='arrive-at-goal')
#         self.perform_scheduling = perform_scheduling
#         self.pt = PyTemporal() #PyTemporal temporal consistency checker
#
#     def get_state(self,position,site,crashed,visited,new_discovery,tcs):
#         """
#         Returns a proper state representation.
#         """
#         state_dict = {'position':position,      #Position on map
#                       'site':site,              #Map site
#                       'crashed':crashed,        #Crashed against obstacle flag
#                       'visited':visited,        #Sets of visited locations
#                       'tcs':tcs,                #Temporal constraints
#                       'new_discovery':new_discovery} #Dictionary containing if a site
#                                                      #contains a new discovery
#
#         #Checks if the temporal network is still strongly controllable
#         if len(tcs)>0 and self.perform_scheduling:
#             squeeze_dict,objective,sc_schedule = self.pt.strongly_controllable_stp(tcs)
#             if squeeze_dict==None: #No ccSC schedule found.
#                 state_dict['sched-risk']=1.0
#                 #print('Strongly controllable reformulation failed...')
#             else: #ccSC schedule found, so it records the scheduling risk
#                 prob_success=1.0
#                 for tc_dict in squeeze_dict.values():
#                     prob_success*=(1.0-tc_dict['risk'])
#                 state_dict['sched-risk']=1.0-prob_success
#                 #print('Scheduling risk (assuming independent stochastic durations): %.4f%%'%(state_dict['sched-risk']*100.0))
#         else:
#             #print('No constraints. Trivially schedulable.')
#             state_dict['sched-risk']=0.0
#
#         return state_dict
#
#     def get_initial_belief(self,prior,initial_site='_start_',initial_pos=None,
#                            init_tcs=[],goal_site='_goal_',goal_pos=None):
#         """
#         Generates an initial belief distribution over the presence of new discoveries
#         in the map, assuming independence of the prior probabilities. Morevoer,
#         initializes the list of temporal constraints.
#         """
#         scenarios=self._generate_scenarios(prior)
#         self._handle_site(initial_site,initial_pos); self.start_site = initial_site
#         self._handle_site(goal_site,goal_pos); self.goal = goal_site
#         return self._scenarios_to_states(scenarios=scenarios,
#                                          initial_pos=self.sites[self.start_site]['coords'],
#                                          initial_site=self.start_site,
#                                          init_tcs=init_tcs)
#
#     def _scenarios_to_states(self,scenarios,initial_pos,initial_site,init_tcs):
#         """"
#         Generates the initial belief particles from a list of scenarios.
#         """
#         belief = {}
#         for s in scenarios:
#             state = self.get_state(position=initial_pos,site=initial_site,
#                                    crashed=False,visited=set(),new_discovery=s[0],
#                                    tcs=init_tcs)
#             belief[self.hash_state(state)] = [state,s[1]]
#         return belief
#
#     def actions(self,state,**kwargs):
#         """
#         The actions available at a state correspond to driving to different
#         discovery sites.
#         """
#         if not self.is_terminal(state):
#             curr_site=state['site']
#             not_visited = [s for s in self.science_sites if (not s in state['visited']) and s!=curr_site]
#             actions=[]
#             for goal_site in not_visited:
#                 for r in self.path_risks:
#                     #Trajectory episode
#                     path_rmpyl=self._plan_cc_path(curr_site,goal_site,r)
#
#                     if curr_site == self.start_site:
#                         #Add [0,0] temporal constraints to force consistence
#                         tc = TemporalConstraint(start=self.global_start_event,
#                                                 end=path_rmpyl.first_event,
#                                                 ctype='controllable',lb=0.0, ub=0.0)
#                         path_rmpyl.add_temporal_constraint(tc)
#                     elif goal_site == self.goal:
#                         #Add [0,0] temporal constraints to force consistence
#                         tc = TemporalConstraint(start=path_rmpyl.last_event,
#                                                 end=self.global_end_event,
#                                                 ctype='controllable',lb=0.0, ub=0.0)
#                         path_rmpyl.add_temporal_constraint(tc)
#                     else:
#                         pass
#
#                     actions.append(FollowPathAction(rmpyl=path_rmpyl,
#                                                     episode=path_rmpyl.plan,
#                                                     start_site=curr_site,
#                                                     goal_site=goal_site,
#                                                     risk=r))
#             return actions
#         else:
#             return []
#
#     def state_transitions(self,state,action):
#         """
#         Returns the next state, after executing an operator.
#         """
#         if self._at_site(state['position'],state['site']):
#             if state['crashed']:#If you're crashed, you stay crashed
#                 return [[state,1.0]]
#             else:
#                 crash_state = self.get_state(position=(),site='_crash_',crashed=True,
#                                              visited=set(),new_discovery={},tcs=[])
#
#                 #The goal state has an additional temporal constraint related to the
#                 #duration of the traversal.
#                 goal_state = self.get_state(position=action.episode.properties['goal_coords'][0:2],
#                                             site=action.goal_site,crashed=False,
#                                             visited=state['visited'].union(set([state['site']])),
#                                             new_discovery=state['new_discovery'],
#                                             tcs=state['tcs']+list(action.rmpyl.temporal_constraints))
#                 #ipdb.set_trace()
#                 return [[goal_state,1.0-action.risk],[crash_state,action.risk]]
#         else:
#             raise ValueError('Current position %s and current site %s diverged'%(str(state['position']),state['site']))
#
#     def is_terminal(self,state):
#         """
#         A state is terminal if all sites have been visited or it has crashed.
#         """
#         return state['crashed'] or (state['site'] == self.goal)
#
#     def terminal_value(self,state):
#         """
#         No extra reward at a terminal state.
#         """
#         if state['crashed']:
#             return 0.0
#         else:
#             if state['site'] == self.goal:
#                 if state['site'] in state['new_discovery'] and state['new_discovery'][state['site']]:
#                     return self.sites[state['site']]['value']
#                 else:
#                     return 0.0
#             else:
#                 return -np.inf
#
#     def state_risk(self,state):
#         """
#         Crash states have risk 1.0, while all others have risk related to
#         scheduling.
#         """
#         return 1.0 if state['crashed'] else (state['sched-risk'] if 'sched-risk' in state else 0.0)
