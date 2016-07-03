#!/usr/bin/env python
#
#  A simple Python simulator for power networks.
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

Implements AO*, a forward heuristic planner for fully-observable domains.

@author: Pedro Santana (psantana@mit.edu).
""" 

from rao.hypergraphtools import HyperGraph, GraphNode, GraphOperator
from rao.models.models import HyperGraphModel

import numpy as np
import pydot
from collections import deque
import operator

class AOStarGraphNode(GraphNode):
    """Hypergraph node used in AO*, which stores an estimate of its Q value."""
    def __init__(self,name=None,node_id=None,value=None,state=None):
        super(AOStarGraphNode,self).__init__(name,node_id,dict_key=None)

        #Turns the state into a string to make sure is is hashable                
        self.dict_key = lambda: str(self.get_state())

        self._value = value        #Q value
        
        self._is_terminal=False
        self._state = state

    def get_value(self):
        return self._value    
    def get_state(self):
        return self._state

    def set_state(self,new_state):
        self._state = new_state
    def set_value(self,new_value):
        self._value = new_value    

    def is_terminal(self):
        return self._is_terminal
    def set_terminal(self,is_terminal):
        self._is_terminal = is_terminal


class AOStarGraphOperator(GraphOperator):
    """Hypergraph operator used in AO*, which has a value associated to it."""
    def __init__(self,name=None,op_id = None,op_value=0.0,properties=()):
        super(AOStarGraphOperator,self).__init__(name,op_id,properties)
        self._op_value = op_value

    def op_value(self):
        return self._op_value
    def set_op_value(self,new_value):
        self._op_value = new_value


class AOStarHyperGraph(HyperGraph):
    """Hypergraph containing the additional info necessary to perform AO*
    search. For AO*, the graph is, in fact, a tree."""
    def __init__(self):
        super(AOStarHyperGraph,self).__init__()
        self._root=None

    def get_root(self):
        return self._root
    def set_root(self,node_obj):
        self._root = node_obj

    def to_dot(self, policy=None):
        """Returns a Dot representation of the AO* policy graph."""
        #DIrected graph, Left to Right (LR)
        #dot_graph = pydot.Dot(graph_type="digraph",rank="same",rankdir="LR")
        dot_graph = pydot.Dot(graph_type="digraph")

        #Adds all edges to the Dot graph
        if policy != None:
            policy_nodes = [self.get_node_by_key(node_key) for node_key in list(policy.keys())]
        else:
            policy_nodes=[]

        graph_nodes = policy_nodes if policy!=None else self.get_all_nodes()

        for node in graph_nodes:
            node_label = "State:%s\nValue:%.4f"%(node.get_name(), node.get_value())
            if policy!= None and (node.dict_key() in list(policy.keys())):
                node_label += "\nOp:"+policy[node.dict_key()].get_name()
            dot_graph.add_node(pydot.Node(node.get_name(),
                                          style="filled",
                                          fillcolor="white",
                                          label=node_label))

            graph_ops = [policy[node.dict_key()]] if policy!=None else self.get_all_operators_at_node(node)
            for op in graph_ops:
                for child_idx,child in enumerate(self.get_successors(node,op)):
                    child_label = "State:%s\nValue:%.4f"%(child.get_name(),
                                                          child.get_value())
                    if policy!=None and (child.dict_key() in list(policy.keys())):
                        child_label += "\nOp:"+policy[child.dict_key()].get_name()

                    dot_graph.add_node(pydot.Node(child.get_name(),
                                                      style="filled",
                                                      fillcolor="white",
                                                      label=child_label))

                    dot_graph.add_edge(pydot.Edge(src=node.get_name(),
                                                  dst=child.get_name(),
                                                  label=str(op.get_properties()['prob'][child_idx])))
        return dot_graph



class AOStar(object):
    """AO* algorithm, which finds optimal policies on an AND-OR tree."""
    def __init__(self,model,node_name='id'):
        if not isinstance(model,HyperGraphModel):
            print("\nThe model object isn't of type HyperGraphModel.")
            raise TypeError

        if node_name == 'id':
            self._create_node = self._create_node_id_name
        elif node_name == 'state':
            self._create_node = self._create_node_state_name
        else:
            print("ERROR: choose either id or state for node_name.")
            return None
            
        self._model = model
        self._explicit = None
        self._opennodes=[]
        self._policy={}
        self._ancestors={}

        #Choose the comparison function depending on the type of search
        if model.is_maximization:
            self._is_better = operator.gt
            self._default_value = -np.infty
            self._select_best = lambda n_list: max(n_list,
                                                   key=lambda node: node.get_value())
        else:
            self._is_better = operator.lt
            self._default_value = np.infty
            self._select_best = lambda n_list: min(n_list,
                                                   key=lambda node: node.get_value())

    def search(self,start,verbose=False):
        """Searches for the optimal path from start to goal on the hypergraph."""
        self._init_search(start)
        count=0
        while len(self._opennodes)>0:
            if verbose:
                print("Iteration "+str(count)+", Open nodes: "+str(len(self._opennodes)))
                count+=1

            #Expands the current best solution
            node = self._expand_best_partial_solution()

            if verbose:
                print("Expanded node "+str(node.get_state()))

            #Updates the value estimates and policy
            self._update_values_and_policy(node)
            #Updates the mapping of ancestors on the best policy graph and also
            #the list of open nodes to be expanded.
            self.__update_policy_ancestors_and_open_nodes()

        return self._policy, self._explicit

    def _create_node_state_name(self,state):
        return AOStarGraphNode(name=str(state),value=self._default_value,
                               state=state)

    def _create_node_id_name(self,state):
        return AOStarGraphNode(name=None,value=self._default_value,
                               state=state)

    def _init_node_fields(self,node,heur_func,term_fun):
        """Initializes the fields of a recently created node."""                
        node.set_value(heur_func(node.get_state()))                            
        node.set_terminal(term_fun(node.get_state()))

    def _init_search(self,start):
        """Initializes the search fields."""
        #Initializes the explicit graph with the start node
        self._explicit =AOStarHyperGraph()
        start_node = self._create_node(start)
        self._init_node_fields(start_node,self._model.heuristic,
                               self._model.is_terminal)
        self._explicit.add_node(start_node)
        self._explicit.set_root(start_node)

        #Adds the start node to the list of open nodes.
        self.__update_policy_ancestors_and_open_nodes()

    def _expand_best_partial_solution(self):
        """Expands a node in the hypergraph currently contained in the best
        partial solution. The result of the expansion is stored as new nodes
        and edges on the hypergraph."""
        #Chooses a random element from the list to be
        node = self.__choose_node_to_be_expanded()

        #Aliases for model quantities
        A,V,T,h = [self._model.actions, 
                   self._model.value, 
                   self._model.state_transitions, 
                   self._model.heuristic]
        all_node_actions = A(node.get_state())

        if len(all_node_actions)>0:
            #For every action available at the node being expanded
            for act in all_node_actions:
                #Obtains the list of child nodes generated by his action.
                child_obj_list,prob_list = self.__obtain_child_objs_and_probs(T(node.get_state(),act))

                for child_obj in child_obj_list:
                    self._init_node_fields(child_obj,h,self._model.is_terminal)                    

                #Creates operator representing the action.
                act_obj = AOStarGraphOperator(name=str(act),op_value=V(node.get_state(),act),
                                              properties={'prob':prob_list})

                #Adds the hyperedge to the graph
                self._explicit.add_hyperedge(parent_obj=node,
                                             child_obj_list=child_obj_list,
                                             op_obj=act_obj)
        else:
            node.set_terminal(True)

        return node

    def _update_values_and_policy(self,expanded_node):
        """Updates the Q values on nodes on the hypergraph and the current
        best policy."""
        #Updates all ancestors of the expanded node
        Z =  self.__build_ancestor_set(expanded_node)
        while len(Z)>0:
            node = Z.pop()
            #All actions available at that node
            all_action_operators = self._explicit.get_all_operators_at_node(node)

            if len(all_action_operators)>0:
                #Estimates the value of the current node for each possible action
                best_Q = self._default_value; best_action_idx=-1
                for act_idx,act in enumerate(all_action_operators):

                    probs = act.get_properties()['prob']
                    children = self._explicit.get_successors(node,act)

                    #Computes an estimate of the value (Q) of taking this action at
                    #current node. It is composed of the current reward and the average
                    #reward of the potential children.
                    Q = act.op_value()+np.sum([p*child.get_value() for (p,child) in zip(probs,children)])

                    if self._is_better(Q,best_Q):
                        best_Q = Q; best_action_idx = act_idx

                #updates the node's value
                node.set_value(best_Q)
                #Mark the best action
                self._policy[node.dict_key()] = all_action_operators[best_action_idx]

    def __update_policy_ancestors_and_open_nodes(self):
        """Traverses the best solution graph recording the ancestors along marked
        edges."""
        queue = deque([self._explicit.get_root()]) #Starts traversal at root

        self._ancestors={}; self._opennodes=[]; traversed_nodes=set()
        while len(queue)>0:
            node = queue.popleft()
            if self.__policy_has_node(node):
                traversed_nodes.add(node.dict_key())
                children = self._explicit.get_successors(node,self._policy[node.dict_key()])
                for child in children:
                    #If the child node can be reached by different paths and
                    #has already been added, appends the current node
                    if child.dict_key() in self._ancestors:
                        self._ancestors[child.dict_key()].append(node)
                    #Otherwise, initializes the list of ancestors.
                    else:
                        self._ancestors[child.dict_key()] = [node]
                #Extends the search to the child nodes
                queue.extend(children)
            else:
                if not node.is_terminal():
                    self._opennodes.append(node)

        #TODO: test if this makes sense
        remove_keys = [k for k in list(self._policy.keys()) if not k in traversed_nodes]
        for key in remove_keys:
            del self._policy[key]


    def __choose_node_to_be_expanded(self):
        """Chooses an element from the open list to be expanded."""
        node = self._select_best(self._opennodes)
        self._opennodes.remove(node)
        return node

    def __obtain_child_objs_and_probs(self,child_tuples):
        """Obtains the list of references to child node objects in the graph."""
        child_obj_list=[]; prob_list=[]
        for child_state,prob in child_tuples:
            #Node already present in the graph
            child_state_str = str(child_state)
            if self._explicit.has_node_key(child_state_str):
                child_obj = self._explicit.get_node_by_key(child_state_str)
            #Creates a new node if it doesn't exist
            else:
                child_obj = self._create_node(child_state)
                self._init_node_fields(child_obj,self._model.heuristic,
                               self._model.is_terminal)

            child_obj_list.append(child_obj)
            prob_list.append(prob)

        return child_obj_list,prob_list

    def __policy_has_node(self,node):
        """Checks whether the node is part of the current best policy."""
        return node.dict_key() in self._policy

    def __build_ancestor_set(self,expanded_node):
        """Create a set Z that contains the expanded state and all of its
        ancestors in the explicit graph along marked action arcs. (I.e., only
        include ancestor states from which the expanded state can be reached by
        following the current best solution.)"""
        Z = [expanded_node]
        for node in Z:
            if node.dict_key() in self._ancestors:
                Z.extend(self._ancestors[node.dict_key()])
        Z.reverse()
        return Z















