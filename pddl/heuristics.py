#!/usr/bin/env python
#
#  Copyright (c) 2014 MIT. All rights reserved.
#
#   author: Tiago Vaquero, Pedro Santana
#   e-mail: tvaquero@mit.edu, psantana@mit.edu
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

Class representing PDDL domains as (PO)MDP's.

@author: Tiago Vauero (tvaquero@mit.edu), Pedro Santana (psantana@mit.edu).
"""
import os
import copy as cp
import time
from rao.pddl.model_parser import model_parser
from rao.external_planners.external_pddl_planner import External_PDDL_Planner
from rao.external_planners.pysat import PySAT

def get_heuristic_approach(name, domain_file='', problem_file='',
                           domain=None, problem=None, task=None):
    """
    Creates the target Heuristic object (based on the name of the approach)
    to handle heurticic value computation
    """
    if name == 'h_max':
        return HMaxHeuristic(task)
    elif name == 'external_planner':
        return SubPlannerHeuristic(domain_file,problem_file, domain, problem, task)

    return None

class PDDLHeuristic(object):
    """
    Generic class defining the interface of PDDL heuristics.
    """
    def __init__(self,**kwargs):
        pass

    def compute_heuristic(self,**kwargs):
        raise NotImplementedError('You should implement the heuristic computation function.')


class HMaxHeuristic(PDDLHeuristic):
    """
    Computes the Hmax (delete effect) heuristic for PDDL problem.
    """
    def __init__(self,pddl_task):
        #Makes a copy of all PDDL operators and removes their delete effects
        self.task = pddl_task
        self.no_del_operators = cp.deepcopy(pddl_task.operators)
        for op in self.no_del_operators:
            op.del_effects = frozenset()

    def compute_heuristic(self,true_predicates,goals=None):
        """
        Computes the Hmax heuristic.
        """
        combined_predicates = set(true_predicates)
        goal_predicates = goals if goals != None else self.task.goals
        last_size = -1
        layers = 0

        while(len(combined_predicates)!=last_size):
            if goal_predicates <= combined_predicates:
                return layers
            else:
                last_size = len(combined_predicates)
                layers+=1

                for op in [o for o in self.no_del_operators if o.applicable(combined_predicates)]:
                    combined_predicates.update(op.add_effects)

        return float('inf')


class PySATPlanAnalyzer(object):
    """
    Provides planning analytics that can be useful in heuristic forward
    search for PDDL planning. This implementation uses PySAT to determine
    the smallest number of actions that achieve a goal, in addition to the
    set of optimal initial actions.
    """
    def __init__(self,domain_file,prob_file,precompute_steps=0,sequential=True,
                 remove_static=True,cache_solutions=True,verbose=True):

        self._py_sat = PySAT(domain_file,prob_file,
                             precompute_steps=precompute_steps,
                             sequential=sequential,
                             remove_static=remove_static,
                             write_dimacs=False,
                             verbose=verbose)
        #HMAX heuristic
        self._h_max = HMaxHeuristic(self._py_sat.task)

        #Whether to cache previous solutions for future use
        self.cache_solutions = cache_solutions
        if self.cache_solutions:
            self._plan_length_cache = {}
            self._optimal_actions_cache = {}

        #Verbosity flag
        self.verbose = verbose

    def shortest_plan_length(self,initial_state,goals,max_steps,min_steps=-1):
        """
        Determines the minimum number of steps to drive the system from
        an initial to a goal state, while optionally caching solutions.
        """
        if self.cache_solutions:
            cache_tuple = (initial_state,goals)
            if cache_tuple in self._plan_length_cache:
                #If the shortest plan length has already been determined
                #and it is consistent with the maximum number of steps,
                #returns the cached solution
                if self._plan_length_cache[cache_tuple]<=max_steps:
                    spl = self._plan_length_cache[cache_tuple]
                #Not enough steps to solve the plan
                else:
                    spl = float('inf')

            #Computes the shortest plan length and caches it.
            else:
                spl = self._shortest_plan_length(initial_state,goals,max_steps,min_steps)
                self._plan_length_cache[cache_tuple] = spl
        else:
            spl = self._shortest_plan_length(initial_state,goals,max_steps,min_steps)

        return spl


    def _shortest_plan_length(self,initial_state,goals,max_steps,min_steps):
        """
        Determines the minimum number of steps to drive the system from
        an initial to a goal state.
        """
        if self.verbose:
            print('\n##### Determining optimal plan length!\n')

        #If a minimum number of steps was not provided, estimates it
        #using the Hmax heuristic
        if min_steps <0:
            min_steps = self._h_max.compute_heuristic(initial_state)

        #If Hmax detects that the plan is infeasible, the minimum
        #length is infinite
        if min_steps == float('inf'):
            if self.verbose:
                print('\n##### Plan infeasibility detected by Hmax.')
            shortest_plan_length = float('inf')
        #Otherwise, uses PySAT's binary search procedure to determine
        #the optimal plan length.
        else:
            if self.verbose:
                print('\n##### Determining shortest plan length (Max=%d,Min=%d).'%(max_steps,min_steps))
            start = time.time()
            plans = self._py_sat.plan(initial_state,goals,time_steps=max_steps,
                                      find_shortest=True,min_steps=min_steps)
            elapsed = time.time()-start
            if self.verbose:
                print('\n##### Determining shortest plan length took %.4f s'%(elapsed))

            if plans != None:
                shortest_plan_length = len(plans[0])
                if self.verbose:
                    print('##### Shortest plan has %d steps!\n'%(shortest_plan_length))
            else:
                shortest_plan_length = float('inf')
                if self.verbose:
                    print('##### The plan is infeasible!\n')

        return shortest_plan_length

    def first_actions(self,initial_state,goals,time_steps):
        """
        Determines the set of possible first actions for plans of a
        given length, while optionally caching solutions.
        """
        if self.cache_solutions:
            cache_tuple = (initial_state,goals,time_steps)
            #If the set of first actions has already been determined,
            #returns the cached value
            if cache_tuple in self._optimal_actions_cache:
                first_actions = self._optimal_actions_cache[cache_tuple]

            #Computes the first actions and caches them.
            else:
                first_actions = self._first_actions(initial_state,goals,time_steps)
                self._optimal_actions_cache[cache_tuple] = first_actions
        else:
            first_actions = self._first_actions(initial_state,goals,time_steps)

        return first_actions


    def _first_actions(self,initial_state,goals,time_steps):
        """
        Determines the set of possible first actions for plans of a
        given length
        """
        if self.verbose:
            print('\n##### Determining set of optimal first actions for %d steps.\n'%(time_steps))
        start = time.time()
        first_actions = self._py_sat.first_actions(initial_state,goals,time_steps=time_steps)
        if self.verbose:
            print('\n##### Determining optimal actions took %.4f s'%(time.time()-start))
            if first_actions != None:
                print('\n##### Number of optimal first actions for plans with %d steps: %d.'%(time_steps,len(first_actions)))
                for i,action in enumerate(first_actions):
                    print('%d: %s'%(i,action))
            else:
                print('\n##### No feasible plans for plans with %d steps.'%(time_steps))

        return first_actions



class PySATHmaxHeuristic(PDDLHeuristic):
    """
    Uses the simple PySAT solver, combined with Hmax, to guide the search in a PDDL domain.
    """
    def __init__(self,domain_file,prob_file,precompute_steps=40,remove_static=True,
                verbose=True):
        #PySAT solver with precomputation
        self._pysat = PySAT(domain_file,prob_file,precompute_steps=precompute_steps,
                           sequential=True,add_noop=True,remove_static=remove_static,
                           write_dimacs=False,verbose=verbose)
        #Parses the PDDL domain
        self._domain,self._problem,self._task = model_parser(domain_file,prob_file,
                                                         remove_static=remove_static)

        #HMAX heuristic
        self._h_max = HMaxHeuristic(self._task)

    def compute_heuristic(self,true_predicates,time_steps,find_shortest=True):
        """
        Uses Hmax as a 'sanity checker' before calling PySAT.
        """
        #If Hmax returns a finite value, computes a refinement with PySAT
        if self._h_max.compute_heuristic(true_predicates) != float('inf'):
            plans = self._pysat.plan(self._task.initial_state,self._task.goals,
                                time_steps=time_steps,find_shortest=find_shortest)
            #Length of shortest plan
            if len(plans)>0:
                return len(plans)
            #No plan exists
            else:
                float('inf')
        #If Hmax returns no solution, it's hopeless to search
        else:
            return float('inf')




class SubPlannerHeuristic(PDDLHeuristic):
    """
    Object that can compute the heuristic for PDDL
    problems using sub-planner.
    """
    def __init__(self, domain_file, problem_file, pddl_domain, pddl_problem, pddl_task):

        self.domain = pddl_domain
        self.problem = pddl_problem
        self.task = pddl_task
        self.subplanner = External_PDDL_Planner(domain_file=domain_file,
                                                problem_file=problem_file,
                                                domain=pddl_domain,
                                                problem=pddl_problem,
                                                task=pddl_task,
                                                #arguments=['domain','problem','output'],
                                                #arguments=['--domain','domain','--problem','problem','--output','output'],
                                                #folder='lapkt',
                                                #executable='siw-then-bfsf'
                                                )

        #change the problem file to a temporary copy of the problem witch will
        # be use when calling the planner multiple times. IN that way it wont
        # change the original problem
        self.subplanner.problem_file = os.path.join(self.subplanner.executable_dir, os.path.basename(problem_file)+'_temp')



    def compute_heuristic(self,true_predicates):
        #combined_predicates = set(true_predicates)
        h_value = 0.0

        relaxed_plan = self.subplanner.plan(initial_state=true_predicates)
        if len(relaxed_plan) == 0:
            h_value = float('inf')
        else:
            h_value = len(relaxed_plan)

        #remove
        #if os.path.isfile(self.subplanner.problem_file):
        #    os.remove(self.subplanner.problem_file)
        print(h_value)
        return h_value #float('inf')
