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

Classes useful for representing chance-constrained path planning problems, in
which RAO* uses a chance-constrained path planner for each individual trajectory
in order to make sure that the overall risk of the driving mission does not
exceed the acceptable chance constraint.

@author: Pedro Santana (psantana@mit.edu).
"""
from collections import namedtuple
import numpy as np
from numpy import linalg as la
from .models import CCHyperGraphModel
try:
    from pytemporal.paris import PARIS
    _PYTEMPORAL_FOUND=True
except ImportError:
    _PYTEMPORAL_FOUND=False; print('PyTemporal or RMPyL not found. Cannot assess temporal risk.')


class FollowPathAction(namedtuple('FollowPathAction',['rmpyl','episode','start_site','goal_site','risk'])):
    """
    Simple class representing an action where an agent follows a path.
    """
    __slots__= ()

    def __repr__(self):
        return '(go-from-to %s %s),risk<=%.4f'%(self.start_site,self.goal_site,self.risk)


class CCRockSample(CCHyperGraphModel):
    """
    Class defining common methods for chance-constrained path planning applications.
    """
    def __init__(self,path_planner,sites,prob_discovery=0.95,name='robot',
                 max_visits=-1,verbose=0):
        super(CCRockSample,self).__init__()
        self.verbose = verbose                 #Output verbosity
        self.is_maximization = True            #Goal is to maximize the value of visiting sites
        self.immutable_actions = False         #Available actions depend on state
        self.path_planner=path_planner         #Path planner
        self.name=name                         #Agent's name

        self.sites=sites  #Dictionary listing sites to be visited.
        for s_name,s_dict in self.sites.items():
            if not isinstance(s_dict['coords'],np.ndarray):
                s_dict['coords'] = np.array(s_dict['coords'])

        self.science_sites = tuple(self.sites.keys())
        self._num_sites = len(self.science_sites) #All science sites
        self.prob_discovery = prob_discovery #Probability of makign a discovery, if there is one

    def _at_site(self,position,site,tol=1e-2):
        """
        Returns whether an agent is close to the expected site.
        """
        if site =='_crash_': #Anywhere can be a crash
            return True
        else:
            return la.norm(self.sites[site]['coords']-position)<tol

    def get_state(self,position,site,crashed,visited,new_discovery):
        """
        Returns a proper state representation.
        """
        state_dict = {'position':position,      #Position on map
                      'site':site,              #Map site
                      'crashed':crashed,        #Crashed against obstacle flag
                      'visited':visited,        #Sets of visited locations
                      'new_discovery':new_discovery} #Dictionary containing if a site
                                                     #contains a new discovery
        return state_dict

    def get_initial_belief(self,prior,initial_site='_start_',initial_pos=None):
        """
        Generates an initial belief distribution over the presence of new discoveries
        in the map, assuming independence of the prior probabilities.
        """
        scenarios=self._generate_scenarios(prior)
        self._add_new_site(initial_site,initial_pos); self.start_site = initial_site
        return self._scenarios_to_states(scenarios=scenarios,
                                         initial_pos=self.sites[self.start_site]['coords'],
                                         initial_site=self.start_site)

    def _generate_scenarios(self,prior):
        """
        Generates the possible execution scenarios, given the prior distribution
        of science sites.
        """
        scenarios=[[{},1.0]]
        for site_name, prior_prob in prior.items():
            new_scenarios=[]
            for s in scenarios:
                if np.isclose(prior_prob,0.0):
                    s[0][site_name]=False #There is certainly nothing at the site
                elif np.isclose(prior_prob,1.0):
                    s[0][site_name]=True #There is certainly something at the site
                else: #Forks on the two possible scenarios
                    s_copy=[dict(s[0]),s[1]]
                    s[0][site_name]=True; s[1]*=prior_prob
                    s_copy[0][site_name]=False; s_copy[1]*=(1.0-prior_prob)
                    new_scenarios.append(s_copy)
            if len(new_scenarios)>0:
                scenarios=scenarios+new_scenarios

        return scenarios

    def _add_new_site(self,site,site_pos):
        """
        Handles a site specified at the initial belief.
        """
        if not (site in self.sites):
            self.sites[site]={}
            self.sites[site]['coords'] = np.array(site_pos)
            self.sites[site]['value'] = 0.0
            self._num_sites+=1 #Adds another site
        elif site_pos!=None:
            raise ValueError('Tried to redefine position of site '+site)
        else:
            pass

    def _scenarios_to_states(self,scenarios,initial_pos,initial_site):
        """"
        Generates the initial belief particles from a list of scenarios.
        """
        belief = {}
        for s in scenarios:
            state = self.get_state(position=initial_pos,
                                   site=initial_site,crashed=False,
                                   visited=set(),new_discovery=s[0])
            belief[self.hash_state(state)] = [state,s[1]]
        return belief

    def is_terminal(self,state):
        """
        A state is terminal if all sites have been visited or it has crashed.
        """
        return state['crashed'] or (len(state['visited'])==self._num_sites)

    def state_transitions(self,state,action):
        """
        Returns the next state, after executing an operator.
        """
        if self._at_site(state['position'],state['site']):
            if state['crashed']:#If you're crashed, you stay crashed
                return [[state,1.0]]
            else:
                crash_state = self.get_state(position=(),site='_crash_',crashed=True,
                                        visited=set(),new_discovery={})

                goal_state = self.get_state(position=action.episode.properties['goal_coords'][0:2],
                                            site=action.goal_site,crashed=False,
                                            visited=state['visited'].union(set([state['site']])),
                                            new_discovery=state['new_discovery'])
                return [[goal_state,1.0-action.risk],[crash_state,action.risk]]
        else:
            raise ValueError('Current position %s and current site %s diverged'%(str(state['position']),state['site']))

    def value(self,state,action):
        """
        Returns the value of making a discovery at a site when it is visited the
        first time.
        """
        curr_site = state['site'] #Current site
        if not (curr_site in [self.start_site,'_crash_']):
            #First time we visited the site
            if not (curr_site in state['visited']):
                if state['new_discovery'][curr_site]: #New discovery at this site
                    return self.sites[curr_site]['value'] #Value of making discovery

        #If the site has already been visited or if there is no new discovery
        #there, returns no value of visiting it.
        return 0.0

    def terminal_value(self,state):
        """
        No extra reward at a terminal state.
        """
        if state['site'] in state['new_discovery'] and state['new_discovery'][state['site']]:
            return self.sites[state['site']]['value']
        else:
            return 0.0

    def heuristic(self,state):
        """
        Admissible heuristic of the value of being at a state.
        """
        #Very simple admissible heuristic, in which we just assume that we will
        #find new discoveries at each undiscovered site.
        if state['crashed']:
            return 0.0
        else:
            unvisited_discoveries = [s for s in self.science_sites if not (s in state['visited']) and state['new_discovery'][s]]
            return np.sum([self.sites[site]['value'] for site in unvisited_discoveries])

    def state_risk(self,state):
        """
        Crash states have risk 1.0, while all others have risk 0. I should take
        into account risks from other sources (running out of time, for instance)
        in future iterations.
        """
        return 1.0 if state['crashed'] else 0.0

    def execution_risk_heuristic(self,state):
        """
        Admissible estimate of the remaining mission risk.
        """
        return self.state_risk(state)

    def observations(self,state):
        """
        Assumes that a new discovery will be made with high probability, if one
        exists, or returns that the agent has crashed.
        """
        curr_site=state['site']
        if state['crashed']:
            return [[('obstacle',curr_site),1.0]]
        else:
            #If there is something new to be discovered there, it will be discovered
            #with high probability.
            if curr_site in state['new_discovery'] and state['new_discovery'][curr_site]:
                return [[('new_discovery',curr_site),self.prob_discovery],
                        [('nothing',curr_site),1.0-self.prob_discovery]]
            #If there is nothing to discover, no new knowledge can be acquired.
            else:
                return [[('nothing',curr_site),1.0]]

    def obs_repr(self,observation):
        """
        Nice representation of a discovery at a site.
        """
        return '%s_%s'%(observation[0],observation[1])



if _PYTEMPORAL_FOUND:

    class tCCRockSample(CCRockSample):
        """
        Extension of CCRockSample that requires the policies to be strongly
        controllable in a probabilistic sense. Uses PyTemporal for checking.
        """
        def __init__(self,path_planner,sites,perform_scheduling=True,
                     prob_discovery=0.95,name='robot',paris_params={},verbose=0):

            super(tCCRockSample,self).__init__(path_planner,sites,prob_discovery,name,verbose)

            #Events used to enforce global durations
            #self.global_start_event = Event(name='depart-from-start')
            #self.global_end_event = Event(name='arrive-at-goal')
            self.perform_scheduling = perform_scheduling

            #PARIS strong controllability checker
            self.scheduler = PARIS(**paris_params)


        def get_state(self,position,site,crashed,visited,new_discovery,tcs=[]):
            """
            Returns a proper state representation.
            """
            state_dict = {'position':position,      #Position on map
                          'site':site,              #Map site
                          'crashed':crashed,        #Crashed against obstacle flag
                          'visited':visited,        #Sets of visited locations
                          'tcs':tcs,                #Temporal constraints
                          'new_discovery':new_discovery} #Dictionary containing if a site
                                                         #contains a new discovery

            #Checks if the temporal network is still strongly controllable
            if len(tcs)>0 and self.perform_scheduling:
                squeeze_dict,objective,sc_schedule = self.scheduler.schedule(tcs)
                if squeeze_dict==None: #No ccSC schedule found.
                    state_dict['sched-risk']=1.0
                    #print('Strongly controllable reformulation failed...')
                else: #ccSC schedule found, so it records the scheduling risk
                    prob_success=1.0
                    for tc_dict in squeeze_dict.values():
                        prob_success*=(1.0-tc_dict['risk'])
                    state_dict['sched-risk']=1.0-prob_success
                    #print('Scheduling risk (assuming independent stochastic durations): %.4f%%'%(state_dict['sched-risk']*100.0))
            else:
                #print('No constraints. Trivially schedulable.')
                state_dict['sched-risk']=0.0

            return state_dict

        def get_initial_belief(self,prior,initial_site='_start_',initial_pos=None,
                               init_tcs=[],goal_site='_goal_',goal_pos=None):
            """
            Generates an initial belief distribution over the presence of new discoveries
            in the map, assuming independence of the prior probabilities. Morevoer,
            initializes the list of temporal constraints.
            """
            scenarios=self._generate_scenarios(prior)
            self._add_new_site(initial_site,initial_pos); self.start_site = initial_site
            self._add_new_site(goal_site,goal_pos); self.goal = goal_site
            return self._scenarios_to_states(scenarios=scenarios,
                                             initial_pos=self.sites[self.start_site]['coords'],
                                             initial_site=self.start_site,
                                             init_tcs=init_tcs)

        def _scenarios_to_states(self,scenarios,initial_pos,initial_site,init_tcs):
            """"
            Generates the initial belief particles from a list of scenarios.
            """
            belief = {}
            for s in scenarios:
                state = self.get_state(position=initial_pos,site=initial_site,
                                       crashed=False,visited=set(),new_discovery=s[0],
                                       tcs=init_tcs)
                belief[self.hash_state(state)] = [state,s[1]]
            return belief

        def state_transitions(self,state,action):
            """
            Returns the next state, after executing an operator.
            """
            if self._at_site(state['position'],state['site']):
                if state['crashed']:#If you're crashed, you stay crashed
                    return [[state,1.0]]
                else:
                    crash_state = self.get_state(position=(),site='_crash_',crashed=True,
                                                 visited=set(),new_discovery={},tcs=[])

                    #The goal state has an additional temporal constraint related to the
                    #duration of the traversal.
                    goal_state = self.get_state(position=action.episode.properties['goal_coords'][0:2],
                                                site=action.goal_site,crashed=False,
                                                visited=state['visited'].union(set([state['site']])),
                                                new_discovery=state['new_discovery'],
                                                tcs=state['tcs']+list(action.rmpyl.temporal_constraints))

                    return [[goal_state,1.0-action.risk],[crash_state,action.risk]]
            else:
                raise ValueError('Current position %s and current site %s diverged'%(str(state['position']),state['site']))

        def is_terminal(self,state):
            """
            A state is terminal if all sites have been visited or it has crashed.
            """
            return state['crashed'] or (state['site'] == self.goal)

        def terminal_value(self,state):
            """
            No extra reward at a terminal state.
            """
            if state['crashed']:
                return 0.0
            else:
                if state['site'] == self.goal:
                    if state['site'] in state['new_discovery'] and state['new_discovery'][state['site']]:
                        return self.sites[state['site']]['value']
                    else:
                        return 0.0
                else:
                    return -np.inf #You MUST end at the goal location, or crash

        def state_risk(self,state):
            """
            Crash states have risk 1.0, while all others have risk related to
            scheduling.
            """
            return 1.0 if state['crashed'] else (state['sched-risk'] if 'sched-risk' in state else 0.0)
