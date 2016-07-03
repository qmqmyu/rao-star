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

A simple SAT-based PDDL planner written in Python

@author: Pedro Santana (psantana@mit.edu), Tiago Vaquero (tvaquero@mit.edu).
"""
from .pddl_planner import PDDL_Planner
from rao.pddl.model_parser import model_parser
from rao.pddl.task import Operator
import time
import itertools
import pycosat

class PySAT(PDDL_Planner):
    """
    A simple SAT-based PDDL planner written in Python.
    """
    def __init__(self,domain_file,prob_file,precompute_steps=0,sequential=True,
                 remove_static=True,add_noops=False,write_dimacs=False,verbose=True):
        self.domain,self.problem,self.task = model_parser(domain_file,
                                                          prob_file,
                                                          remove_static=remove_static)
        #Whether to return sequential plans
        self.sequential = sequential

        #Verbosity level
        self.verbose = verbose

        #Whether to write CNF's to a DIMACS file
        self.write_dimacs=write_dimacs

        #Adds no ops to the layers
        if add_noops:
            noop = Operator('(noop)',frozenset(),frozenset(),frozenset())
            self.task.operators.append(noop)

        #Initializes the integer variable counter
        self._last_integer=0

        if self.verbose:
            print('\nPrecomputing clauses for %d time steps'%(precompute_steps))

        start = time.time()
        self._build_sat_map(precompute_steps)
        if self.verbose:
            print('Time spent precomputing clauses: %.4f s'%((time.time()-start)))

    def plan(self,init,goals,time_steps,find_shortest=False,min_steps=0,max_sols=1,conflict_func=None):
        """
        Solves the planning problem for a given number of time steps.
        """
        #If the shortest plan is required, performs binary search on the length
        if find_shortest:
            if goals<=init:
                sols = [[]] #Goals are already achieved, so do nothing
            else:
                sol = self._binary_plan_length_search(init,goals,
                                                      min_steps=min_steps,
                                                      max_steps=time_steps)
                sols = [sol] if sol != None else None
        #If the shortest plan is not sought, just tries to find one that works
        else:
            sols = self._find_multiple_plans(init,goals,time_steps,max_sols=max_sols,conflict_func=conflict_func)

        #If a solution was found, returns the operators. Otherwise, returns an
        #empty list.
        return [self._plan_list(s) for s in sols] if sols != None else None

    def first_actions(self,init,goals,time_steps,max_sols=-1):
        """
        Finds the set of first actions in all plans.
        """
        if goals<=init:
            return [] #The optimal action is to take no actions
        else:
            #Forces all solutions to have distinct first actions
            sols = self._find_multiple_plans(init,goals,time_steps,max_sols=max_sols,
                                             conflict_func=self._not_first_op)

        #If a solution was found, returns the operators. Otherwise, returns None
        return [self._plan_list(s)[0] for s in sols] if sols != None else None

    def _build_plan_cnf(self,init,goals,time_steps):
        """
        Builds the CNF expression representing the planning problem.
        """
        #Initial condition clauses
        cnf_init = self._assert_initial_conditions(init)

        #Goal condition clauses
        cnf_goals = self._assert_goals(goals,time_steps)

        #Operator clauses
        self._update_operator_clauses(time_steps)
        cnf_ops = [c for c in itertools.chain(*self._op_clause_cache[0:time_steps])]
        # for t in range(time_steps):
        #     cnf+= self._op_clause_cache[t]

        cnf = cnf_init+cnf_goals+cnf_ops

        if self.write_dimacs:
            if self.verbose:
                print('##### Writing CNF to DIMACS file.')
            to_DIMACS(cnf)

        return cnf

    def _solve_sat(self,cnf):
        """
        Calls the SAT solver.
        """
        start = time.time()
        sol = pycosat.solve(cnf)
        elapsed = time.time()-start
        if self.verbose:
            print('\t- Took %.3f ms to solve SAT problem.'%(1000.0*elapsed))
        return sol

    def _not_same_plan(self,sat_sol):
        """Prevents the same plan from being generated twice."""
        return [[-v for v in self._plan_active_op_vars(sat_sol)]]

    def _not_first_op(self,sat_sol):
        """
        Returns a clause preventing the first operator from being picked again.
        """
        #Negation of first operator's SAT var.
        return [[-min(self._plan_active_op_vars(sat_sol),key=lambda v: self._int_to_op[v][1])]]

    def _find_multiple_plans(self,init,goals,time_steps,max_sols=1,conflict_func=None):
        """
        Finds one or more feasible plans within the time horizon, with the
        possibility of adding conflicts between solutions to differentiate
        between them.
        """
        #If not particular conflict function is specified, prevents the same plan
        #from being generated twice.
        conflict_clause_func = conflict_func if conflict_func != None else self._not_same_plan

        #Constructs the planning CNF
        cnf = self._build_plan_cnf(init,goals,time_steps)

        sols=[]; sol_count=0
        while(sol_count != max_sols):
            sol = self._solve_sat(cnf)

            #Solution space has been exhausted
            if not self._is_valid_sol(sol):
                break
            #New solution found
            else:
                sol_count+=1
                sols.append(sol) #Append current valid solution

                #Adds clauses preventing the same solution from being generated
                #twice.
                conflict_clauses = conflict_clause_func(sol)
                cnf+= conflict_clauses

        #Returns the list of solutions
        return sols if len(sols)>0 else None

    def _binary_plan_length_search(self,init,goals,min_steps,max_steps):
        """
        Performs binary search to find shortest plan length.
        """
        t_min = min_steps; t_max = max_steps
        last_sol = []; last_t=-1
        while(True):
            #Stopping condition
            if t_max-t_min==1:
                #t_min was not tested and is a valid solution
                if t_min == min_steps:
                    sols = self._find_multiple_plans(init,goals,t_min)
                    if sols != None:
                        last_sol = sols[0]; last_t = t_min; break

                #t_max was not tested and is a valid solution
                if t_max == max_steps:
                    sols = self._find_multiple_plans(init,goals,t_max)
                    if sols != None:
                        last_sol = sols[0]; last_t = t_max

                break

            #Distance is even (there is a middle element)
            elif (t_max-t_min)%2==0:
                t = (t_max+t_min)//2 #Takes middle point
            #Distance is odd (no middle point)
            else:
                t = (t_max+t_min+1)//2 #Takes "upper" middle point

            if self.verbose:
                print('\t- Trying plan with %d steps!'%(t))

            sols = self._find_multiple_plans(init,goals,t)

            #If a solution was found for t, it means that we do not have to
            #consider horizons larger than t anymore.
            if sols != None:
                t_max = t
                last_sol = sols[0]
                last_t = t
                if self.verbose:
                    print('\t- SAT\n')
            #If a solution was not found for t, it means that we need to
            #consider longer horizons
            else:
                t_min = t
                if self.verbose:
                    print('\t- UNSAT\n')

        #Both extrema have been updated during search. If a solution
        #has been previously found, returns it.
        if last_t>0:
            if self.verbose:
                print('Shortest plan has %d steps!'%(last_t))
            return last_sol
        #If no solution has been found, the problem has no solution
        else:
            return None

    def _is_valid_sol(self,sol):
        """
        Whether a solution to the SAT solver can be converted to a plan.
        """
        return isinstance(sol,list)

    def _next_sat_integer(self):
        """
        Returns the next available integer to represent SAT variables.
        """
        self._last_integer+=1
        return self._last_integer

    def _plan_list(self,sat_sol):
        """
        Reconstructs a sequence of PDDL operators from a SAT solution.
        """
        op_sat_vars = self._plan_active_op_vars(sat_sol)
        #Tuples (op,t) of PDDL operator objects and time steps
        op_ints = [self._int_to_op[i] for i in op_sat_vars]

        #Sorts the operator integers by time step
        op_ints.sort(key=lambda x:x[1])

        #For sequential plans, just returns the list of operators
        if self.sequential:
            plan_list = [x[0].name for x in op_ints]
        #For parallel plans, groups them by time step
        else:
            max_step = max([x[1] for x in op_ints])
            plan_list = [[] for i in range(max_step+1)]
            for op,t in op_ints:
                plan_list[t].append(op.name)           

        return plan_list

    def _plan_active_op_vars(self,sat_sol):
        """
        Returns the SAT variables associated with active operators.
        """
        return [i for i in sat_sol if i>0 and (i in self._int_to_op)]

    def _build_sat_map(self,num_steps):
        """
        Creates a mapping between predicates and operators to SAT variables for
        each time step.
        """
        #Mapping from predicates to integers (the last layer should contain the
        #goals and no operator).
        self._pred_to_int= [None]*(num_steps+1)
        # self._pred_to_int= {}
        for i in range(num_steps+1):
            self._pred_to_int[i]={}
            for p in self.task.facts:
                self._pred_to_int[i][p] = self._next_sat_integer()

        #Mapping from operators to integers (there will be as many operators as
        #there are steps in the plan).
        self._op_to_int= [None]*(num_steps)
        # self._op_to_int= {}
        self._int_to_op={}
        for j in range(num_steps):
            self._op_to_int[j]={}
            for op in self.task.operators:
                sat_int = self._next_sat_integer()
                self._op_to_int[j][op] = sat_int
                #Inverse mapping from integers to operators and a time step
                self._int_to_op[sat_int]=[op,j]

        #Mapping from predicates to the operators that add and delete them
        self._pred_op_map={p:{'add':set(),'del':set()} for p in self.task.facts}
        for op in self.task.operators:
            for p in op.add_effects:
                self._pred_op_map[p]['add'].add(op)
            for p in op.del_effects:
                self._pred_op_map[p]['del'].add(op)

        #Precomputes operator clauses
        self._op_clause_cache=[]
        # self._op_clause_cache={}
        self._update_operator_clauses(num_steps)

    def _update_operator_clauses(self,num_steps):
        """
        Updates the cache of operator clauses.
        """
        if len(self._op_clause_cache)<num_steps:
            for t in range(len(self._op_clause_cache),num_steps):
                self._op_clause_cache.append(self._assert_operators(t)+self._assert_predicate_inertia(t))
                # self._op_clause_cache[t] = self._assert_operators(t)
                # self._op_clause_cache[t] += self._assert_predicate_inertia(t)

    def _assert_initial_conditions(self,init):
        """
        Disjunction of singletons representing the initial conditions
        """
        #Everything that is true
        cnf = self._assert_predicates(init,0)
        #Everything that is false
        neg_pred = [p for p in self.task.facts if not p in init]
        cnf += self._assert_predicates(neg_pred,0,negated=True)
        return cnf

    def _assert_goals(self,goals,last_t):
        """
        Disjunction of singletons representing the goals.
        """
        return self._assert_predicates(goals,last_t)

    def _assert_operators(self,t):
        """
        Implements
            op_t => op_preconditions_t
            op_t => effects_t+1
            at least one op_t must be true
            if sequential, no pair (op1_t,op2_t) can be true at the same time
            if parallel (not sequential), then only mutex operators cannot coexist
        """
        cnf=[]
        some_op_clause=[]
        for op,op_sat_int in self._op_to_int[t].items():

            #This operator could be executed
            some_op_clause.append(op_sat_int)

            #Operators imply their preconditions
            for p in op.preconditions:
                cnf.append( [-op_sat_int, self._pred_to_int[t][p]] )

            #Operators imply their add effects at the next step
            for p in op.add_effects:
                cnf.append( [-op_sat_int, self._pred_to_int[t+1][p]] )

            #Operators imply the abscense of delete effects at the next step
            for p in op.del_effects:
                cnf.append( [-op_sat_int, -self._pred_to_int[t+1][p]] )

        #At least one operator must be chosen
        cnf.append(some_op_clause)

        for op1,op1_sat_int in self._op_to_int[t].items():
            for op2,op2_sat_int in self._op_to_int[t].items():
                if op1_sat_int != op2_sat_int:
                    #For sequential plans,no two operators can coexist. For parallel
                    #plans, only if they are not mutex.
                    if self.sequential or self._are_mutex(op1,op2):
                        cnf+= [[-op1_sat_int,-op2_sat_int]]

        return cnf

    def _are_mutex(self,op1,op2):
        """
        Tests if two operators are mutually exclusive.
        """
        op_list = [op1,op2]
        for i in range(2):
            #One operator deletes the preconditions from the other or they have
            #inconsistent effects
            if  len(op_list[i].preconditions & op_list[1-i].del_effects)>0 or \
                len(op_list[i].add_effects & op_list[1-i].del_effects)>0:
                return True

        return False

    def _assert_predicate_inertia(self,t):
        """
        Implements
            p_t+1 => p_t OR (OR of operators that add p_t+1)
            not p_t+1 => not p_t OR (OR of operators that delete p_t+1)
        """
        cnf=[]
        for p in self.task.facts:
            #p holds at the next step
            positive_inertia_clause = [-self._pred_to_int[t+1][p], self._pred_to_int[t][p]]
            #Actions that add p
            positive_inertia_clause += [self._op_to_int[t][op] for op in self._pred_op_map[p]['add']]

            #p does not hold at the next step
            negative_inertia_clause = [self._pred_to_int[t+1][p], -self._pred_to_int[t][p]]
            #Actions that deletes p
            negative_inertia_clause += [self._op_to_int[t][op] for op in self._pred_op_map[p]['del']]

            cnf+=[positive_inertia_clause,negative_inertia_clause]

        return cnf

    def _assert_predicates(self,predicates,t,negated=False):
        """
        Disjunction of singletons representing the initial conditions
        """
        mul = -1 if negated else 1
        return [ [self._pred_to_int[t][p]*mul ] for p in predicates]



def to_DIMACS(cnf,filename='problem.dimacs'):
    """
    Exports a CNF list to DIMACS format.
    """
    num_vars = max([ max([abs(v) for v in clause]) for clause in cnf])
    with open(filename,'w') as f:
        f.write('p cnf %d %d\n'%(num_vars,len(cnf)))
        for clause in cnf:
            f.write(' '.join([str(v) for v in clause])+' 0\n')
