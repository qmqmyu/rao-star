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
import pickle
import numpy as np
from numpy import linalg as la

try:
    from rmpyl.rmpyl import Event,TemporalConstraint
    from pytemporal.interface import PyTemporal
    _PYTEMPORAL_FOUND=True
except ImportError:
    _PYTEMPORAL_FOUND=False; print('PyTemporal or RMPyL not found. Cannot assess temporal risk.')


def _load_cached_paths(sites,filename,tol=1e-2):
    """
    Loads a dictionary of cached pSulu solutions from a pickle file.
    """
    try:
        with open(filename,'rb') as f:
            print('Reading site dictionary and cached pSulu paths from '+filename)
            loaded_sites,loaded_paths = pickle.load(f)

            reuse=set() #Computes sets of sites that can be reused
            for s_name,s_dict in sites.items():
                if s_name in loaded_sites and la.norm(loaded_sites[s_name]['coords']-sites[s_name]['coords'])<tol:
                    reuse.add(s_name)

            #Restores the cached paths for sites that can be reused.
            cached={}
            if len(reuse)>=2:
                for path_tup,risk_dict in loaded_paths.items():
                    if (path_tup[0] in reuse) and (path_tup[1] in reuse):
                        cached[path_tup]=risk_dict

        return cached,loaded_sites,loaded_paths
    except IOError:
        print('Could not find '+filename)
        return {},{},{}


def _write_cached_paths(sites,loaded_sites,loaded_paths,cached,filename):
    """
    Writes the sites and cached paths to a pickle file.
    """
    with open(filename,'wb') as f:
        print('Writing site dictionary and cached pSulu paths to '+filename)
        for s_name, s_dict in sites.items():
            if s_name in loaded_sites:
                loaded_sites[s_name].update(s_dict)
            else:
                loaded_sites[s_name]=s_dict

        for path_tup, risk_dict in cached.items():
            if path_tup in loaded_paths:
                loaded_paths[path_tup].update(risk_dict)
            else:
                loaded_paths[path_tup]=risk_dict

        pickle.dump([loaded_sites,loaded_paths],f)


def _plan_psulu_path(path_planner,sites,cached,start_site,goal_site,risk,
                     path_parameters,duration_type,agent,**kwargs):
    """
    Plans a chance-constrained path from a start location to a goal location.
    Specific parameters are given as keyword arguments.
    """
    path_tup = (start_site,goal_site); generate_path=False
    if path_tup in cached:
        if not risk in cached[path_tup]:
            generate_path=True
    else:
        cached[path_tup]={}
        generate_path=True

    if generate_path:
        start = sites[start_site]['coords']
        goal = sites[goal_site]['coords']

        #Updates the chance constraint
        path_parameters['chance_constraint']=risk

        sol_properties,waypoints = path_planner.plan(start_state=tuple(start)+(0.0,0.0),
                                                     goal_state=tuple(goal)+(0.0,0.0),
                                                     parameters=path_parameters,
                                                     parse_output=True)
        #Inputs to pSulu
        psulu_input=path_planner.last_input

        if sol_properties!=None:
            final_pos=waypoints[-1][0:2]
            if la.norm(final_pos-goal)>1e-2:
                raise RuntimeError('pSulu failed to arrive at %s from %s. Final position was %s'%(goal,start,final_pos))
            #TODO: ideally, the collision risk should be extracted from the output of pSulu.
            #However, since this is not available at the moment, the chance constraint is
            #an overestimate of it.
            cached[path_tup][risk]=[sol_properties,waypoints,psulu_input]
        else:
            raise RuntimeError('pSulu could not find path from %s to %s.'%(str(start),str(goal)))
    else:
        sol_properties,waypoints,psulu_input = cached[path_tup][risk]

    return path_planner.as_rmpyl(sol_properties,waypoints,duration_type=duration_type,
                                 psulu_input=psulu_input,agent=agent,**kwargs)



class pSuluRockSampleModel(CCRockSample):
    """
    Implementation of a chance-constrained path planner using pSulu.
    """
    def __init__(self,path_planner,sites,duration_type='uniform',path_risks=[0.001],prob_discovery=0.95,verbose=0):
        super(pSuluRockSampleModel,self).__init__(path_planner,sites,prob_discovery,verbose)
        self._cached={} #Cache of computed paths

        self.path_parameters={}
        self.path_parameters['executor']='ProOFCSA'
        self.path_parameters['waypoints']=10
        self.path_parameters['time_horizon']=200.0
        self.path_risks = path_risks

        init_pos_var=0.0
        process_pos_var=0.1
        self.stoch_model = self.path_planner.simple_stochastic_model(self.path_parameters,
                                                                     init_pos_var,
                                                                     process_pos_var,dim=2)
        #Writes the model file to avoid having to do so over and over again
        self.path_planner.write_model_file(self.stoch_model)

        #Type of temporal duration used to represented traversal times
        self.duration_type=duration_type

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

    def load_cached_paths(self,filename,tol=1e-2):
        """
        Loads a dictionary of cached pSulu solutions from a pickle file.
        """
        self._cached,self._loaded_sites,self._loaded_paths = _load_cached_paths(self.sites,filename,tol)

    def write_cached_paths(self,filename):
        """
        Writes the sites and cached paths to a pickle file.
        """
        _write_cached_paths(self.sites,self._loaded_sites,self._loaded_paths,self._cached,filename)

    def _plan_cc_path(self,start_site,goal_site,risk,**kwargs):
        """
        Plans a chance-constrained path from a start location to a goal location.
        Specific parameters are given as keyword arguments.
        """
        return _plan_psulu_path(self.path_planner,self.sites,self._cached,start_site,goal_site,risk,
                                self.path_parameters,self.duration_type,self.name,**kwargs)


if _PYTEMPORAL_FOUND:

    class ptSuluRockSampleModel(tCCRockSample):
        """
        Extension of the pSuluRockSample model that also requires the policies
        to be strongly controllable in a probabilistic sense. Uses PyTemporal
        for checking.
        """
        def __init__(self,path_planner,sites,perform_scheduling=True,duration_type='uniform',
                     path_risks=[0.001],prob_discovery=0.95,verbose=0):
            super(ptSuluRockSampleModel,self).__init__(path_planner,sites,perform_scheduling,
                                                       prob_discovery,verbose)
            self._cached={} #Cache of computed paths

            self.path_parameters={}
            self.path_parameters['executor']='ProOFCSA'
            self.path_parameters['waypoints']=10
            self.path_parameters['time_horizon']=200.0
            self.path_risks = path_risks

            init_pos_var=0.0
            process_pos_var=0.1
            self.stoch_model = self.path_planner.simple_stochastic_model(self.path_parameters,
                                                                        init_pos_var,
                                                                        process_pos_var,dim=2)
            #Writes the model file to avoid having to do so over and over again
            self.path_planner.write_model_file(self.stoch_model)

            #Type of temporal duration used to represented traversal times
            self.duration_type=duration_type

        def load_cached_paths(self,filename,tol=1e-2):
            """
            Loads a dictionary of cached pSulu solutions from a pickle file.
            """
            self._cached,self._loaded_sites,self._loaded_paths = _load_cached_paths(self.sites,filename,tol)

        def write_cached_paths(self,filename):
            """
            Writes the sites and cached paths to a pickle file.
            """
            _write_cached_paths(self.sites,self._loaded_sites,self._loaded_paths,self._cached,filename)

        def _plan_cc_path(self,start_site,goal_site,risk,**kwargs):
            """
            Plans a chance-constrained path from a start location to a goal location.
            Specific parameters are given as keyword arguments.
            """
            return _plan_psulu_path(self.path_planner,self.sites,self._cached,start_site,goal_site,risk,
                                    self.path_parameters,self.duration_type,self.name,**kwargs)

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

    # class ptSuluRockSampleModel(pSuluRockSampleModel):
    #     """
    #     Extension of the pSuluRockSample model that also requires the policies
    #     to be strongly controllable in a probabilistic sense. Uses PyTemporal
    #     for checking.
    #     """
    #     def __init__(self,path_planner,sites,perform_scheduling=True,duration_type='uniform',
    #                  path_risks=[0.001],prob_discovery=0.95,verbose=0):
    #
    #         super(ptSuluRockSampleModel,self).__init__(path_planner,sites,duration_type,path_risks,prob_discovery,verbose)
    #
    #         #Events used to enforce global durations
    #         self.global_start_event = Event(name='depart-from-start')
    #         self.global_end_event = Event(name='arrive-at-goal')
    #         self.perform_scheduling = perform_scheduling
    #         self.pt = PyTemporal() #PyTemporal temporal consistency checker
    #
    #     def get_state(self,position,site,crashed,visited,new_discovery,tcs=[]):
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
    #                                          initial_pos=np.array(initial_pos),
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
    #         #return super(ptSuluRockSampleModel,self).state_transitions(state,action)
    #
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


# class pSuluRockSampleModel(CCRockSample):
#     """
#     Implementation of a chance-constrained path planner using pSulu.
#     """
#     def __init__(self,path_planner,sites,duration_type='uniform',path_risks=[0.001],prob_discovery=0.95,verbose=0):
#         super(pSuluRockSampleModel,self).__init__(path_planner,sites,prob_discovery,verbose)
#         self._cached={} #Cache of computed paths
#
#         self.path_parameters={}
#         self.path_parameters['executor']='ProOFCSA'
#         self.path_parameters['waypoints']=10
#         self.path_parameters['time_horizon']=200.0
#         self.path_risks = path_risks
#
#         init_pos_var=0.0
#         process_pos_var=0.1
#         self.stoch_model = self.path_planner.simple_stochastic_model(self.path_parameters,
#                                                                      init_pos_var,
#                                                                      process_pos_var,dim=2)
#         #Writes the model file to avoid having to do so over and over again
#         self.path_planner.write_model_file(self.stoch_model)
#
#         #Type of temporal duration used to represented traversal times
#         self.duration_type=duration_type
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
#     def load_cached_paths(self,filename,tol=1e-2):
#         """
#         Loads a dictionary of cached pSulu solutions from a pickle file.
#         """
#         try:
#             with open(filename,'rb') as f:
#                 print('Reading site dictionary and cached pSulu paths from '+filename)
#                 self._loaded_sites,self._loaded_paths = pickle.load(f)
#
#                 reuse=set() #Computes sets of sites that can be reused
#                 for s_name,s_dict in self.sites.items():
#                     if s_name in self._loaded_sites and la.norm(self._loaded_sites[s_name]['coords']-self.sites[s_name]['coords'])<tol:
#                         reuse.add(s_name)
#
#                 #Restores the cached paths for sites that can be reused.
#                 if len(reuse)>=2:
#                     for path_tup,risk_dict in self._loaded_paths.items():
#                         if (path_tup[0] in reuse) and (path_tup[1] in reuse):
#                             self._cached[path_tup]=risk_dict
#
#         except FileNotFoundError:
#             self._loaded_sites={}; self._loaded_paths={}
#             print('Could not find '+filename)
#
#     def write_cached_paths(self,filename):
#         """
#         Writes the sites and cached paths to a pickle file.
#         """
#         with open(filename,'wb') as f:
#             print('Writing site dictionary and cached pSulu paths to '+filename)
#             for s_name, s_dict in self.sites.items():
#                 if s_name in self._loaded_sites:
#                     self._loaded_sites[s_name].update(s_dict)
#                 else:
#                     self._loaded_sites[s_name]=s_dict
#
#             for path_tup, risk_dict in self._cached.items():
#                 if path_tup in self._loaded_paths:
#                     self._loaded_paths[path_tup].update(risk_dict)
#                 else:
#                     self._loaded_paths[path_tup]=risk_dict
#
#             pickle.dump([self._loaded_sites,self._loaded_paths],f)
#
#     def _plan_cc_path(self,start_site,goal_site,risk,**kwargs):
#         """
#         Plans a chance-constrained path from a start location to a goal location.
#         Specific parameters are given as keyword arguments.
#         """
#         path_tup = (start_site,goal_site); generate_path=False
#         if path_tup in self._cached:
#             if not risk in self._cached[path_tup]:
#                 generate_path=True
#         else:
#             self._cached[path_tup]={}
#             generate_path=True
#
#         if generate_path:
#             start = self.sites[start_site]['coords']
#             goal = self.sites[goal_site]['coords']
#
#             #Updates the chance constraint
#             self.path_parameters['chance_constraint']=risk
#
#             sol_properties,waypoints = self.path_planner.plan(start_state=tuple(start)+(0.0,0.0),
#                                                               goal_state=tuple(goal)+(0.0,0.0),
#                                                               parameters=self.path_parameters,
#                                                               parse_output=True)
#             #Inputs to pSulu
#             psulu_input=self.path_planner.last_input
#
#             if sol_properties!=None:
#                 final_pos=waypoints[-1][0:2]
#                 if la.norm(final_pos-goal)>1e-2:
#                     raise RuntimeError('pSulu failed to arrive at %s from %s. Final position was %s'%(goal,start,final_pos))
#                 #TODO: ideally, the collision risk should be extracted from the output of pSulu.
#                 #However, since this is not available at the moment, the chance constraint is
#                 #an overestimate of it.
#                 self._cached[path_tup][risk]=[sol_properties,waypoints,psulu_input]
#             else:
#                 raise RuntimeError('pSulu could not find path from %s to %s.'%(str(start),str(goal)))
#         else:
#             sol_properties,waypoints,psulu_input = self._cached[path_tup][risk]
#
#         return self.path_planner.as_rmpyl(sol_properties,waypoints,
#                                             duration_type=self.duration_type,
#                                             psulu_input=psulu_input,**kwargs)
