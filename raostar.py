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

Implements RAO*, a forward, heuristic planner for partially-observable,
chance-constrained domains.

@author: Pedro Santana (psantana@mit.edu).
"""
from rao.raostarhypergraph_new import RAOStarGraphNode, RAOStarGraphOperator, RAOStarHyperGraph
from rao.models.models import HyperGraphModel
from rao.belief import BeliefState,avg_func,bound_prob
from rao.belief import predict_belief,update_belief,compute_observation_distribution,is_terminal_belief
from rmpyl.rmpyl import Episode
from rao.export import policy_to_dot

#from rao.debugging import debug_all_hypegraph_values,debug_policy_values,debug_inconsistent_node_value

import numpy as np
import random
from collections import deque
import operator
import time
import os

class RAOStar(object):
    """RAO* algorithm, which finds optimal policies on an AND-OR tree representing
    partially-observable domains."""

    def __init__(self,model,node_name='id',cc=0.0,cc_type='overall',
                 terminal_prob=1.0,randomization=0.0,propagate_risk=True,
                 minimize_risk=False,expand_all_open=True,halt_on_violation=False,
                 enforce_action_ordering=False,enforce_DFS=True,verbose=0,log=False,
                 animation=False):
        if not isinstance(model,HyperGraphModel):
            raise TypeError("The model object isn't of type HyperGraphModel.")

        if (verbose in [0,1,2]):
            self._verbose = verbose
        else:
            raise TypeError('Verbosity must be 0 (min), 1, or 2 (max).') #Verbosity flag
        self._log = log           #Search logging flag
        self._log_f= None         #Logging file
        self._model = model       #Reference to hypergraph model
        self._animation = animation #Whether to create an animation of the search process (very costly!)
        if (randomization>=0.0 and randomization<=1.0):
            self._randomization = randomization
        else:
            raise TypeError('Randomization must be a valid probability value')

        self._cc = cc             #Chance constraint(currently, only a single value)
        #Type of chance constraint
        #
        #overall -> execution risk bound at the root (constraints overall execution)
        #everywhere -> bounds the execution risk at every policy node.
        self._cc_type = cc_type

        self._terminal_prob = terminal_prob #Probability threshold for deeming a belief state terminal
        self._explicit = None     #Reference to explicit graph constructed during the search process
        self._opennodes=set()        #Set of nodes yet to be explored.
        self._policy_ancestors={}        #Ancestor set for updating policies
        self._propagate_risk = propagate_risk
        self._expand_all_open = expand_all_open #Whether to expand all nodes in
                                                #the open list, instead of a
                                                #single one.
        self._halt_on_violation = halt_on_violation #Whether execution should cease at constraint violations

        #Whether or not actions should be sorted
        self._enforce_action_ordering = enforce_action_ordering

        #Whether to enforce Depth First Search (DFS)
        self._enforce_dfs = enforce_DFS

        #Execution risk cap
        if self._cc_type.lower() in ['overall','o']:
            self._er_cap = 1.0
        elif self._cc_type.lower() in['everywhere','e']:
            self._er_cap = self._cc
        else:
            raise TypeError("Choose either \'overall\' or \'everywhere\' for chance constraint type.")

        #Function pointers
        if node_name.lower() == 'id':
            self._create_node = self._create_node_id_name
        elif node_name.lower() == 'state':
            self._create_node = self._create_node_state_name
        else:
            raise TypeError("Choose either \'id\' or \'state\' for node_name.")

        #Choose the comparison function depending on the type of search
        if model.is_maximization:
            self._is_better = operator.gt
            self._is_worse = operator.lt
            self._initial_Q_value = -np.infty
            self._select_best = lambda n_list: max(n_list,
                                                   key=lambda node: node.value)
        else:
            self._is_better = operator.lt
            self._is_worse = operator.gt
            self._initial_Q_value = np.infty
            self._select_best = lambda n_list: min(n_list,
                                                   key=lambda node: node.value)

        #Model function aliases
        self.A,self.V,self.TV,self.T,self.O,self.h,self.r,self.e,self.term = [self._model.actions,
                                                                              self._model.value,
                                                                              self._model.terminal_value,
                                                                              self._model.state_transitions,
                                                                              self._model.observations,
                                                                              self._model.heuristic,
                                                                              self._model.state_risk,
                                                                              self._model.execution_risk_heuristic,
                                                                              self._model.is_terminal]
        #Performance measures
        self._start_time = 0.0

        self.performance={}
        self.performance['optimal_value'] = 0.0
        self.performance['exec_risk_for_optimal_value'] = 0.0
        self.performance['total_elapsed_time'] = 0.0
        self.performance['expanded_nodes'] = 0
        self.performance['evaluated_particles'] = 0
        self.performance['time_to_best_value'] = 0.0
        self.performance['iter_to_best_value'] = 0
        self.performance['root_value_series'] = []
        self.performance['root_exec_risk_series'] = []

    def search(self,b0,time_limit=np.infty,iter_limit=np.infty):
        """Searches for the optimal path from start to goal on the hypergraph."""

        if self._verbose>=1:
            print('\n##### Starting RAO* search!\n')

        self._start_time = time.time()
        self._init_search(b0)
        count=0
        root = self._explicit.root

        #Initial objective at the root, which is the best possible (it can
        #only degrade with an admissible heuristic).
        prev_root_value = np.infty if self._model.is_maximization else -np.infty

        if self._log: #Creates a log file, if necessary
            filename = time.strftime("%d_%m_%Y_%H_%M_%S")+'_log_rao.txt'

            if not os.path.exists('./log'):
                os.makedirs('./log')
            self._log_f=open('./log/'+filename,'w')
            print("\nLogging into "+filename)

        if self._animation:
            if not os.path.exists('./animation'):
                os.makedirs('./animation')
            print("\nCreated animaition directory")

        interrupted=False
        try:
            while len(self._opennodes)>0 and (count<=iter_limit) and (time.time()-self._start_time<=time_limit):
                count+=1

                ########### CORE FUNCTIONS
                #Expands the current best solution
                expanded_nodes = self._expand_best_partial_solution()

                #Updates the value estimates and policy
                self._update_values_and_best_actions(expanded_nodes)

                #NOTE: the value inconsistency here was coming from nodes that
                #were not contained in the best partial policy graph, but contained
                #children in it.
                #debug_all_hypegraph_values(self,'After value update')

                #Updates the mapping of ancestors on the best policy graph and also
                #the list of open nodes to be expanded.
                self._update_policy_open_nodes()
                #debug_policy_values(self,'Policy values after policy update')

                #debug_all_hypegraph_values(self,'After policy update')
                #######################################################

                #Performance measures and info
                root_value = root.value
                self.performance['root_value_series'].append(root_value)
                self.performance['root_exec_risk_series'].append(root.exec_risk)

                #Root node changed from its best value
                if not np.isclose(root_value,prev_root_value):
                    #If the heuristic is really admissible, the root value can
                    #only degrade (decrease for maximization, or increase for
                    #minimization).
                    if self._is_better(root_value,prev_root_value):
                        print('WARNING: root value improved, which might indicate inadmissibility.')
                    else:
                        self.performance['time_to_best_value'] = time.time()-self._start_time
                        self.performance['iter_to_best_value'] = count
                        prev_root_value = root_value

                if self._verbose>=2:
                    print("Expanded nodes [%s]"%(' '.join([str(n.name) for n in expanded_nodes])))

                if self._verbose>=1:
                    total_states=sum([len(node.state.belief) for node in expanded_nodes])
                    print("Iter: %d, Open nodes: %d, States evaluted: %d, Root value: %.4f, Root ER: %.4f"%(count,
                                                                                                            len(self._opennodes),
                                                                                                            total_states,
                                                                                                            root.value,
                                                                                                            root.exec_risk))
                if self._animation: #If an animation should be generated
                    partial_policy = self._extract_policy(partial=True)
                    dot_policy = policy_to_dot(self._explicit,partial_policy)
                    dot_policy.write('./animation/%d_rao.svg'%(count),format='svg')

        except KeyboardInterrupt:
            interrupted=True
            print("\n\n***** EXECUTION TERMINATED BY USER AT ITERATION %d. *****"%(count))

        self.performance['total_elapsed_time'] = time.time()-self._start_time
        self.performance['optimal_value'] = root.value
        self.performance['exec_risk_for_optimal_value'] = root.exec_risk

        if self._log: #Closes the log file, if one has been created
            self._log_f.close()
            print("\nClosed "+filename)

        print("\nTotal elapsed time: %f s"%(self.performance['total_elapsed_time']))
        print("Time to optimal value: %f s"%(self.performance['time_to_best_value']))
        print("Iterations till optimal value: %d"%(self.performance['iter_to_best_value']))
        print("Number of expanded nodes: %d"%(self.performance['expanded_nodes']))
        print("Number of evaluated particles: %d"%(self.performance['evaluated_particles']))
        print("Optimal value: %f"%(self.performance['optimal_value']))
        print("Execution risk for optimal value: %f"%(self.performance['exec_risk_for_optimal_value']))

        policy = self._extract_policy(partial=interrupted)

        if len(policy)==0:
            print('\n##### Failed to find policy (it is empty)...\n')
        elif self.performance['optimal_value']==-float('inf'):
            print('\n##### Failed to find policy (probably due to chance constraint violation)...\n')
        else:
            print('\n##### Policy found!\n')

        return policy, self._explicit, self.performance

    def _init_search(self,b0):
        """Initializes the search fields."""
        #Initializes the explicit graph with the start node
        self._explicit = RAOStarHyperGraph()
        start_node = self._create_node(b0)
        self._set_new_node(start_node,0,self._cc)
        self._explicit.add_node(start_node)
        self._explicit.root = start_node

        #Adds the start node to the list of open nodes.
        self._update_policy_open_nodes()

    def _create_node_state_name(self,belief_dict):
        """Creates a new node that uses its state string as name."""
        return RAOStarGraphNode(name=str(belief_dict),
                                value=None,state=BeliefState(belief_dict))

    def _create_node_id_name(self,belief_dict):
        """Creates a new node that uses a uuid as name."""
        return RAOStarGraphNode(value=None,state=BeliefState(belief_dict))

    def _set_new_node(self,node,depth,er_bound):
        """Sets the fields of a terminal node."""
        b = node.state.belief

        #Depth of a node is its distance to the root (the root being 0)
        node.depth = depth

        #Probability of violating constraints in a belief state. This number
        #should never change
        node.risk = bound_prob(avg_func(b,self.r))

        if is_terminal_belief(b,self.term,self._terminal_prob):
            self._set_terminal_node(node)
        else:
            #The value of a node is the average of the heuristic only when it's first
            #created. After that, the value is given as a function of its children
            node.value = avg_func(b,self.h)

            ##TODO: remove this if we come to the conclusion that enforcing action
            ## ordering or exploration depth prevents RAO* for spending time looking
            ## for solutions at useless places
            ##
            # #If the heuristic value shows that the node is hopeless, marks it as
            # #terminal.
            # if node.value == self._initial_Q_value:
            #     self._set_terminal_node(node)
            # else:

            node.terminal = False #Non-terminal new node

            #New nodes have no action associated to them
            node.best_action = None

            #Execution risk bound
            node.exec_risk_bound = bound_prob(er_bound)

            #Avg heuristic estimate of execution risk at that node
            #avg_h_exec_risk = self._avg_func(node.get_state(),heur_exec_risk_func)
            node.exec_risk = node.risk

    def _set_terminal_node(self,node):
        """Sets the fields of a terminal node."""
        node.terminal = True
        #For terminal nodes, value is the average of the terminal value function
        node.value = avg_func(node.state.belief,self.TV)
        node.exec_risk = node.risk
        node.best_action=None

        self._explicit.remove_all_hyperedges(node)

    def _get_all_actions(self,belief,A):
        """Gets all actions that can be applied at the states in the belief.
        We assume that non-applicable actions just keep the system in the same
        state."""
        if len(belief)>0:
            #If the set of actions is immutable, compute only for one. Otherwise, we
            #must compute union (intersection?)
            if self._model.immutable_actions:
                #Starts with the actions from the first particle.
                particle_state = belief[list(belief.keys())[0]][0]
                all_node_actions = list(A(particle_state))
            else:
                all_node_actions = []
                action_ids=set() #Uses str(a) as ID

                for particle_key,particle_tuple in belief.items():
                    new_actions = [a for a in A(particle_tuple[0]) if not str(a) in action_ids]
                    all_node_actions.extend(new_actions) #TODO: this is supposed to be intersection
                    action_ids.update([str(a) for a in new_actions])

            # #Only sorts actions for debugging purposes
            # if self._log:
            #     all_node_actions.sort(key=lambda x:str(x))

            return all_node_actions
        else:
            return []

    def _expand_best_partial_solution(self):
        """Expands a node in the hypergraph currently contained in the best
        partial solution. The result of the expansion is stored as new nodes
        and edges on the hypergraph."""
        #Chooses a random element from the list to be
        if self._expand_all_open:
            nodes_to_expand = self._opennodes
            self._opennodes = None
        else:
            nodes_to_expand = [self._choose_node_to_be_expanded()]

        for node in nodes_to_expand:

            belief = node.state.belief #Belief state associated to the node
            self.performance['evaluated_particles']+=len(belief)

            parent_risk = node.risk   #Execution risk for current node
            parent_bound = node.exec_risk_bound #ER bound for current node
            parent_depth = node.depth #Distance of the parent to the root

            #If the current node is guaranteed to violate constraints and we
            #want to halt in this case, makes the node terminal by returning
            #no applicable actions.
            if self._halt_on_violation and np.isclose(parent_risk,1.0):
                all_node_actions=[]
            else:
                #Actions applicable at the belief state
                all_node_actions = self._get_all_actions(belief,self.A)

                # #TODO: remove this debug
                # action_set = set([str(ac) for ac in all_node_actions])
                # if len(all_node_actions)!=len(action_set):
                #     print('WARNING: DUPLICATED actions')
                #     ipdb.set_trace()

            if self._log:
                log_str = '\n[%d][EXPAND] Node %s\n'%(time.time(),node.name[0:5])
                log_str +='\tActions considered: '+str(all_node_actions)+'\n'
                self._log_f.write(log_str)

            action_added = False #Flags if a new action has been added
            if len(all_node_actions)>0:

                self.performance['expanded_nodes']+=1 #Increases the expanded node counter

                #For every action available at the node being expanded
                added_count=0 #TODO: remove this

                for act in all_node_actions:
                    #Obtains the list of child nodes generated by his action.
                    child_obj_list,prob_list,prob_safe_list,new_child_idxs,pretty_obs_list=self._obtain_child_objs_and_probs(belief,self.T,self.O,self.r,act)
                    #Makes sure there is a non-empty list of children. In theory,
                    #this should not happen, right?
                    if len(child_obj_list)>0:

                        #Initializes the new child nodes
                        for c_idx in new_child_idxs:
                            self._set_new_node(child_obj_list[c_idx],parent_depth+1,0.0)

                        #If the parent bound Delta is 1.0 (or very close, given
                        #numerical errors), the child nodes are guaranteed to have
                        #their risk bound equal to 1 (just look at the risk
                        #propagation equation).
                        if (not np.isclose(parent_bound,1.0)) and self._propagate_risk:
                            #Computes execution risk bounds for the child nodes, given
                            #given the parent for the parent and its own risk.
                            er_bounds,er_bound_infeasible = self._compute_exec_risk_bounds(parent_bound,
                                                                                           parent_risk,
                                                                                           child_obj_list,
                                                                                           prob_safe_list)
                        else: #Risk bounds are not propagated
                            er_bounds = [1.0]*len(child_obj_list)
                            er_bound_infeasible = False

                        #Only creates new operator if all er bounds are non-negative.
                        if not er_bound_infeasible:

                            #Updates the values of the execution risk for all children
                            #that will be added to the graph.
                            for idx,child in enumerate(child_obj_list):
                                child.exec_risk_bound = er_bounds[idx]

                            #Average instantaneous value (cost or reward)
                            avg_op_value = avg_func(belief,self.V,act)

                            #Checks to see if the current action is an RMPyL episode
                            if isinstance(act,Episode):
                                act_episode = act
                            elif hasattr(act,'episode'):
                                act_episode = act.episode
                            else:
                                act_episode = None

                            act_obj = RAOStarGraphOperator(name=str(act),op_value=avg_op_value,
                                                           properties={'prob':prob_list,
                                                                       'prob_safe':prob_safe_list,
                                                                       'obs':pretty_obs_list,
                                                                       'episode':act_episode})
                            #Adds the hyperedge to the graph
                            self._explicit.add_hyperedge(parent_obj=node,
                                                         child_obj_list=child_obj_list,
                                                         op_obj=act_obj)

                            action_added = True #Flags that at least one action was added
                            added_count += 1

                            if self._log:
                                log_str = "\t*** Node %s: Added operator %s with value %.4f\n"%(node.name[0:5],str(act),avg_op_value)
                                for idx,child in enumerate(child_obj_list):
                                    log_str += "\t\tChild %s, Prob %.4f\n"%(child.name[0:5],
                                                                            prob_list[idx])
                                self._log_f.write(log_str)

                            # #TODO: remove this debug
                            # operator_name_list = [op.name for op in self._explicit.all_node_operators(node)]
                            # operator_name_set = set(operator_name_list)
                            # if len(operator_name_list)!=len(operator_name_set):
                            #     print('WARNING: DUPLICATED operators at expand.')
                            #     ipdb.set_trace()
                            #     pass

                            # #TODO: remove this debug
                            # if added_count != len(self._explicit.all_node_operators(node)):
                            #     print('WARNING: WRONG number of operators!')
                            #     ipdb.set_trace()

                        else:
                            if self._verbose>=2:
                                print("EXPAND SOLUTION: Action %s at node %s is infeasible due to risk."%(str(act),node.name))
                            if self._log:
                                log_str = "\tAction %s is infeasible due to risk.\n"%(str(act))
                                self._log_f.write(log_str)
                    else:
                        raise RuntimeError("WARNING: node expansion yielded an empty set of child nodes.")

            #If no action was added, the current node must be terminal
            if not action_added:
                self._set_terminal_node(node)
                if self._verbose>=2:
                    print("EXPAND SOLUTION: Node %s deemed terminal due to lack of feasible actions."%(node.name))
                if self._log:
                    log_str = "\tNode %s not expanded due to lack of feasible actions.\n"%(node.name[0:5])
                    self._log_f.write(log_str)

        return nodes_to_expand

    def _update_values_and_best_actions(self,expanded_nodes):
        """Updates the Q values on nodes on the hypergraph and the current
        best policy."""
        #For each expanded node at a time
        for exp_idx,exp_node in enumerate(expanded_nodes):
            #Updates all ancestors of the expanded node
            Z = self._build_ancestor_list(exp_node)

            if self._log:
                log_str = '\n[%d][UPDATE] Nodes to be updated: '%(time.time())
                log_str += str([n.name[0:5] for n in Z])+'\n'
                self._log_f.write(log_str)

            #Updates best action at the node
            for node in Z:

                #All actions available at that node
                all_action_operators = [] if node.terminal else self._explicit.all_node_operators(node)

                # #TODO: remove this debug
                # operator_name_list = [op.name for op in all_action_operators]
                # operator_name_set = set(operator_name_list)
                # if len(operator_name_list)!=len(operator_name_set):
                #     print('WARNING: DUPLICATED operators at update.')
                #     ipdb.set_trace()

                #Only sorts operators for debugging purposes
                if self._log or self._enforce_action_ordering:
                    all_action_operators.sort(key=lambda op: op.name)

                #Risk at the node's belief state (does not depend on the action taken)
                risk = node.risk
                #Current *admissible* (optimistic) estimate of the node's Q value
                current_Q = node.value

                #Execution risk bound. The execution risk cap depends on the type
                #of chance constraint being imposed
                er_bound = min([node.exec_risk_bound,self._er_cap])

                if self._log:
                    log_str = "\tUpdating node %s.\n"%(node.name[0:5])
                    self._log_f.write(log_str)

                best_action_idx=-1
                best_Q = self._initial_Q_value
                best_D = -1
                exec_risk_for_best = -1.0

                #Estimates value and risk of the current node for each
                #possible action
                for act_idx,act in enumerate(all_action_operators):

                    probs = act.properties['prob']
                    probs_safe = act.properties['prob_safe']
                    children = self._explicit.hyperedge_successors(node,act)

                    #Computes an estimate of the value (Q) of taking this action at the
                    #current node. It is composed of the current reward and the average
                    #reward of its children.
                    Q = act.op_value+np.sum([p*child.value for (p,child) in zip(probs,children)])

                    #Average child depth
                    D = 1+np.sum([p*child.depth for (p,child) in zip(probs,children)])

                    #Computes an estimate of the execution risk (er) of taking
                    #this action at the current node. It is composed of the current
                    #risk and the average execution risk of its children.
                    exec_risk = risk+(1.0-risk)*np.sum([p*child.exec_risk for (p,child) in zip(probs_safe,children)])

                    #If the execution risk bound has been violated or if the
                    #Q value for this action is worse (not equal or better) than
                    #the current best, we should definitely not select it.
                    if (exec_risk>er_bound) or self._is_worse(Q,best_Q):
                        select_action = False
                    #In here, the execution risk bound is respected, and the Q
                    #value is either better or equal.
                    else:
                        if np.isclose(Q,best_Q) and self._enforce_dfs:
                            #Select the action if the current node is deeper
                            select_action = D > best_D
                        else:
                            select_action = True

                    #Test if the risk bound for the current node has been violated.
                    #if (exec_risk<=er_bound) and self._is_better(Q,best_Q):
                    if select_action:
                        #Updates the execution risk bounds for the children
                        child_er_bounds,cc_infeasible = self._compute_exec_risk_bounds(er_bound,risk,children,probs_safe)
                        #If the chance constraint has not been violated
                        if not cc_infeasible:
                             #Updates the execution risk bounds for all children
                            for idx,child in enumerate(children):
                                child.exec_risk_bound = child_er_bounds[idx]

                            #Updates the best action at that node
                            best_Q = Q
                            best_action_idx = act_idx
                            best_D = D
                            exec_risk_for_best = exec_risk
                        else:
                            if self._log:
                                log_str = "\t\t Action %s violates risk bound.\n"%(act.name)
                                self._log_f.write(log_str)

                #Tests is some action has been selected
                if best_action_idx>=0:

                    if (not np.isclose(best_Q,current_Q)) and self._is_better(best_Q,current_Q):
                        print('WARNING: node Q value improved, which might indicate inadmissibility.')

                    #updates optimal value estimate
                    node.value = best_Q
                    #updates execution risk estimate
                    node.exec_risk = exec_risk_for_best
                    #Mark the best action
                    node.best_action = all_action_operators[best_action_idx]

                    # #TODO: remove this debug
                    # children = self._explicit.hyperedge_successors(node,node.best_action)
                    # probs = node.best_action.properties['prob']
                    # Q = node.best_action.op_value+np.sum([p*child.value for (p,child) in zip(probs,children)])
                    # if not np.isclose(Q,node.value):
                    #     print('WARNING(%s): inconsistent value at node!'%(call_str))
                    #     ipdb.set_trace()
                    #     pass

                    # #TODO:remove this debug
                    # if np.isclose(node.value,node.best_action.op_value):
                    #     ipdb.set_trace()
                    #     pass

                    if self._log:
                        log_str = "\t\t*** Best action: %s, Value: %.4f\n"%(node.best_action.name,best_Q)
                        self._log_f.write(log_str)

                #If no action was selected, this node is terminal and all of
                #its edges should be removed.
                else:
                    if not node.terminal:
                        self._set_terminal_node(node)

                    # #TODO:remove this debug
                    # if node.best_action!=None and np.isclose(node.value,node.best_action.op_value):
                    #     ipdb.set_trace()
                    #     pass

                    if self._verbose>=2:
                        print("Policy Update: Node %s deemed terminal due to lack of feasible actions."%(node.name))
                    if self._log:
                        log_str = "\tUpdated node %s deemed terminal due to lack of feasible actions."%(node.name[0:5])
                        self._log_f.write(log_str)


    def _compute_exec_risk_bounds(self,parent_bound,parent_risk,child_list,
                                    prob_safe_list,is_terminal_node=False):
        """Computes the execution risk bounds for each sibling in a list of
        children of a node."""
        exec_risk_bounds = [0.0]*len(child_list)

        #If the parent bound is almost one, the risk of the children are guaranteed
        #to be feasible, even if they are all equal to 1.
        if np.isclose(parent_bound,1.0):
            exec_risk_bounds = [1.0]*len(child_list)
            infeasible = False
        else:
            #If the parent bound isn't one, but the risk is, or if the parent
            #risk already violates the risk bound, there is no point
            #in trying to propagate the bound, since the children are guaranteed
            #to violate it.
            if np.isclose(parent_risk,1.0) or (parent_risk>parent_bound):
                infeasible = True
            #Only if the the parent bound and the parent risk are below 1, and
            #the parent risk is below the parent bound, that we should try to
            #propagate risks
            else:
                infeasible = False #Infeasibility flag

                #Risk "consumed" by the parent node
                parent_term = (parent_bound-parent_risk)/(1.0-parent_risk)

                for idx_child,child in enumerate(child_list):

                    #Risk consumed by the siblings of the current node
                    sibling_term = np.sum([p*c.exec_risk for (p,c) in zip(prob_safe_list,child_list) if (c!= child)])

                    if np.isclose(prob_safe_list[idx_child],0.0):
                        import ipdb; ipdb.set_trace()

                    #Execution risk bound, which caps at 1.0
                    exec_risk_bound = min([(parent_term-sibling_term)/prob_safe_list[idx_child],1.0])

                    #A negative bound means that the chance constraint is guaranteed
                    #to be violated. The same is true if the admissible estimate
                    #of the execution risk for a child node violates its upper bound.
                    if exec_risk_bound>=0.0:
                        if child.exec_risk<=exec_risk_bound or np.isclose(child.exec_risk,exec_risk_bound):
                            exec_risk_bounds[idx_child] = exec_risk_bound
                        else:
                            infeasible = True
                            #import ipdb; ipdb.set_trace()
                            break
                    else:
                        infeasible = True
                        #import ipdb; ipdb.set_trace()
                        break

        #Returns the execution risk bounds and the infeasibility flag.
        return exec_risk_bounds,infeasible

    def _update_policy_open_nodes(self):
        """
        Traverses the hypergraph starting at the root along marked actions,
        recording ancestors and open nodes along the way.
        """
        queue = deque([self._explicit.root]) #Starts traversal at root
        # self._policy_ancestors={};
        self._opennodes=set()
        while len(queue)>0:
            node = queue.popleft()
            if node.best_action != None: #Node has been assigned a best action
                children = self._explicit.hyperedge_successors(node,node.best_action)

                # for child in children:
                #     #If the child node can be reached by different paths and
                #     #has already been added, appends the current node
                #     if child in self._policy_ancestors:
                #         self._policy_ancestors[child].append(node)
                #     #Otherwise, initializes the list of ancestors.
                #     else:
                #         self._policy_ancestors[child] = [node]

                #Extends the search to the child nodes
                queue.extend(children)
            #No action has been assinged to the current node
            else:
                #If it is not terminal, then it must be part of the open nodes.
                if not node.terminal:
                    self._opennodes.add(node)

    def _build_ancestor_list(self,expanded_node):
        """Create a set Z that contains the expanded node and all of its
        ancestors in the explicit graph along marked action arcs. (i.e., only
        includes ancestor nodes from which the expanded node can be reached by
        following the current best policy.)"""
        Z = []
        queue = deque([expanded_node]) #Expanded node
        while len(queue)>0:
            node = queue.popleft()
            Z.append(node)
            for parent in self._explicit.all_node_ancestors(node):
                if not parent.terminal and parent.best_action!=None:
                    if node in self._explicit.hyperedge_successors(parent,parent.best_action):
                        queue.append(parent)
        return Z

    def _extract_policy(self,partial=False):
        """Extracts the mapping from nodes to actions constituting the policy."""
        queue = deque([self._explicit.root]) #Starts traversal at root
        policy={}
        while len(queue)>0:
            node = queue.popleft()
            if node.best_action != None: #Node has been assigned a best action
                policy[node] = node.best_action
                children = self._explicit.hyperedge_successors(node,node.best_action)

                #TODO: remove this debug
                #debug_inconsistent_node_value(self,node,call_str='Policy')

                queue.extend(children)
            else:
                if not (node.terminal or partial):
                    raise RuntimeError('Found non-terminal node in policy with no assigned action!')
        return policy

    def _choose_node_to_be_expanded(self):
        """Chooses an element from the open list to be expanded."""
        if len(self._opennodes)>1:
            #Deterministically selects the best node to expand if DFS is being
            #enfore, or there is no randomization, or the randomization did not
            #exceed the probability threshold.
            if (self._enforce_dfs) or (self._randomization==0.0) or np.random.rand()>self._randomization:
                node = self._select_best(self._opennodes) #Select most promising node
            else:
                node = random.sample(self._opennodes,1)[0] #Random open node
            self._opennodes.remove(node)
        else:
            node = self._opennodes.pop()
        return node

    def _obtain_child_objs_and_probs(self,belief,T,O,r,act):
        """Obtains the list of references to child node objects in the graph.
        For RAO*, the children correspond to different observations."""

        #Predicts new particles using the current belief and the state
        #transition model.
        pred_belief,pred_belief_safe = predict_belief(belief,T,r,act,self._model.hash_state)

        #Given the predicted belief, computes the probability distribution of
        #potential observations. Each observations corresponds to a new node
        #on the hypergraph, whose edge is annotated by the probability of
        #that particular observation.
        obs_distribution,obs_distribution_safe,state_to_obs = compute_observation_distribution(pred_belief,
                                                                                               pred_belief_safe,
                                                                                               O,
                                                                                               self._model.hash_state)
        if self._log:
            log_str = "\t\t Observation distribution: %s\n"%(str(obs_distribution))
            log_str += "\t\t Safe observation distribution: %s\n"%(str(obs_distribution_safe))
            log_str += "\t\t Likelihood: %s\n"%(str(state_to_obs))
            self._log_f.write(log_str)

        #For each possible observation, computes the corresponding updated
        #belief
        child_obj_list=[]; prob_list=[]; prob_safe_list=[]; new_child_idxs=[]
        pretty_obs_list=[]; count = 0

        for str_obs,(obs,obs_prob) in obs_distribution.items():

            #Performs belief state update
            child_blf_state = update_belief(pred_belief,state_to_obs,obs)

            # The node will be initialized in the expansion function.
            candidate_child_obj = self._create_node(child_blf_state)

            #Node already present in the graph
            if self._explicit.has_node(candidate_child_obj):
                child_obj = self._explicit.nodes[candidate_child_obj]
            #Uses the newly created one
            else:
                # The node will be initialized in the expansion function.
                child_obj = candidate_child_obj
                new_child_idxs.append(count)

            child_obj_list.append(child_obj)
            prob_list.append(obs_prob)

            if str_obs in obs_distribution_safe:
                obs_safe_tuple = obs_distribution_safe[str_obs]
                prob_safe_list.append(obs_safe_tuple[1])
            else:
                prob_safe_list.append(0.0)

            pretty_obs_list.append(self._model.obs_repr(obs))
            count+=1

        #Returns a list of child nodes and their respective probabilities.
        return child_obj_list,prob_list,prob_safe_list,new_child_idxs,pretty_obs_list
