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

Defines the structures used in the RAO* hypergraph.

@author: Pedro Santana (psantana@mit.edu).
"""
import uuid
#import hashlib as hl

class GraphElement(object):
    """
    Generic graph element with a name and a unique ID.
    """
    def __init__(self,name=None,el_id=None,properties={}):
        self.id = str(uuid.uuid4()) if el_id == None else el_id
        self.name = str(self.id) if name == None else name
        self.properties = properties
        self.hash_key = id(self) #Default hash key

    @property
    def id(self):
        """Element's unique ID."""
        return self._id

    @id.setter
    def id(self,new_id):
        self._id=new_id

    @property
    def name(self):
        """Element's name."""
        return self._name

    @name.setter
    def name(self,new_name):
        self._name=new_name

    @property
    def properties(self):
        """Dictionary of additional element properties."""
        return self._properties

    @properties.setter
    def properties(self,new_properties):
        if isinstance(new_properties,dict):
            self._properties = new_properties
        else:
            raise TypeError('Hypergraph element properties should be given as a dictionary.')

    def __hash__(self):
        """
        Returns the element's hash key.
        """
        return self.hash_key

    def __eq__(x,y):
        """
        Two elements are considered equal if they have the same hash key.
        """
        return isinstance(x,GraphElement) and isinstance(y,GraphElement) and (x.hash_key == y.hash_key)
        #return isinstance(y,GraphElement) and (x.hash_key == y.hash_key)

    def __ne__(self, other):
        return not self == other

class RAOStarGraphNode(GraphElement):
    """
    Class for nodes in the RAO* hypergraph.
    """
    def __init__(self,value,state,best_action=None,terminal=False,name=None,node_id=None,
                 properties={}):
        super(RAOStarGraphNode,self).__init__(name,node_id,properties)
        self.value = value         #Q value
        self.terminal = terminal   #Terminal flag
        self.state = state         #Belief state
        self.best_action = best_action #Best action at the node


        #TODO: fixing this hashing is key to allow nodes with the same Belief
        #state to be identified as being the same.

        #Dictionary key used to detect that tuwo nodes have the same belief state
        #self.hash_key = hl.md5(str(self.state).encode('utf-8')).hexdigest()
        #self.hash_key = hash(str(self.state))

        self.hash_key = self.state.hash_key#Uses the same hash key as the belief states.

        #self.hash_key = hash(''.join(sorted(str(self.state))))#Belief state string in alphabetical order

        self.risk = 0.0             #Belief state risk
        self.exec_risk = 0.0        #Execution risk
        self.exec_risk_bound = 1.0  #Bound on execution risk
        self.depth = 0              #Initial depth

    @property
    def depth(self):
        """Distance of this node to the root."""
        return self._depth

    @depth.setter
    def depth(self,new_depth):
        """Sets new non-negative depth"""
        if new_depth>=0:
            self._depth = new_depth
        else:
            raise ValueError('Node depths must be non-negative!')

    @property
    def value(self):
        """Q value associated with this node"""
        return self._value

    @value.setter
    def value(self,new_value):
        self._value = new_value

    @property
    def state(self):
        """Belief state associated with the node"""
        return self._state

    @state.setter
    def state(self,new_state):
        self._state = new_state

    @property
    def best_action(self):
        """Best operator at the node."""
        return self._best_action

    @best_action.setter
    def best_action(self,new_best_action):
        if new_best_action==None or isinstance(new_best_action,RAOStarGraphOperator):
            self._best_action = new_best_action
        else:
            raise TypeError('The best action at a node should be of type RAOStarGraphOperator.')

    @property
    def terminal(self):
        """Terminal node flag."""
        return self._is_terminal

    @terminal.setter
    def terminal(self,is_terminal):
        self._is_terminal = is_terminal

    @property
    def risk(self):
        """Risk associated with the node's belief state."""
        return self._risk

    @risk.setter
    def risk(self,new_risk):
        if new_risk>=0.0 and new_risk<=1.0:
            self._risk = new_risk
        else:
            raise ValueError('Invalid risk value: %f'%(new_risk))

    @property
    def exec_risk(self):
        """Execution risk associated with the node."""
        return self._e_risk

    @exec_risk.setter
    def exec_risk(self,new_exec_risk):
        if new_exec_risk>=0.0 and new_exec_risk<=1.0:
            self._e_risk = new_exec_risk
        else:
            raise ValueError('Invalid execution risk: %f'%(new_exec_risk))

    @property
    def exec_risk_bound(self):
        """Execution risk bound associated with the node."""
        return self._e_risk_bound

    @exec_risk_bound.setter
    def exec_risk_bound(self,new_exec_risk_bound):
        if new_exec_risk_bound>=0.0 and new_exec_risk_bound<=1.0:
            self._e_risk_bound = new_exec_risk_bound
        else:
            raise ValueError('Invalid execution risk bound: %f'%(new_exec_risk_bound))

    def __str__(self):
        """String representation of a Hypergraph node."""
        return self._name


class RAOStarGraphOperator(GraphElement):
    """
    Class for operators associated to hyperedge in the RAO* hypergraph.
    """
    def __init__(self,name=None,op_id=None,op_value=0.0,properties={}):
        super(RAOStarGraphOperator,self).__init__(name,op_id,properties)

        self.op_value = op_value #Value associated with this operator

        #Dictionary key used to detect that two operators are equal
        #TODO: This fixes the problem with duplicated operators, but doesn't
        #answer the question: how can unduplicated actions give rise to
        #duplicated operators? Answer: because the hypergraph is a graph, and the
        #same node was being expanded through different paths.

        self.hash_key = hash(self.name)

    @property
    def op_value(self):
        """Value associated with executing this operator (reward or cost)."""
        return self._op_value

    @op_value.setter
    def op_value(self,new_value):
        self._op_value = new_value

    def __str__(self):
        """String representation of a hypergraph operator."""
        return str(self.id)+'_'+str(self.properties)


class RAOStarHyperGraph(GraphElement):
    """
    Class representing an RAO* hypergraph.
    """
    def __init__(self,name=None,graph_id=None,properties={}):
        super(RAOStarHyperGraph,self).__init__(name,graph_id,properties)
        #Dictionary of nodes mapping their hash keys to themselves
        self.nodes={}
        #Dictionary of operators mapping their hash keys to themselves
        self.operators={}
        #Nested dictionary {parent_key: {operator_key: successors}}
        self.hyperedges = {}
        #Dictionary from child to sets of parents {child_key: set(parents)}
        self.parents = {}

    @property
    def nodes(self):
        """Nodes in the hypergraph."""
        return self._nodes

    @nodes.setter
    def nodes(self,new_nodes):
        self._nodes = new_nodes

    @property
    def operators(self):
        """Operators in the hypergraph."""
        return self._operators

    @operators.setter
    def operators(self,new_operators):
        self._operators = new_operators

    @property
    def hyperedges(self):
        """Dictionary representing the hyperedges in the graph."""
        return self._hyperedge_dict

    @hyperedges.setter
    def hyperedges(self,new_hyperedges):
        if isinstance(new_hyperedges,dict):
            self._hyperedge_dict = new_hyperedges
        else:
            raise TypeError('Hyperedges should be given in dictionary form')

    @property
    def parents(self):
        """Dictionary mapping from children and actions to parents."""
        return self._parent_dict

    @parents.setter
    def parents(self,new_parents):
        if isinstance(new_parents,dict):
            self._parent_dict = new_parents
        else:
            raise TypeError('Node parents should be given in dictionary form')

    @property
    def root(self):
        """Root of the hypergraph, corresponding to the initial belief state."""
        return self._root

    @root.setter
    def root(self,new_root):
        if isinstance(new_root,RAOStarGraphNode):
            self._root = new_root
            self.add_node(self._root)
        else:
            raise TypeError('The root of the hypergraph must be of type RAOStarGraphNode.')

    def add_node(self,node):
        """Adds a node to the hypergraph."""
        if not node in self.nodes:
            self.nodes[node]=node

    def add_operator(self,op):
        """Adds an operators to the hypergraph."""
        if not op in self.operators:
            self.operators[op]=op

    def add_hyperedge(self,parent_obj,child_obj_list,op_obj):
        """Adds a hyperedge between a parent and a list of child nodes, adding
        the nodes to the graph if necessary."""
        #Makes sure all nodes and operator are part of the graph.
        self.add_node(parent_obj)
        self.add_operator(op_obj)
        for child in child_obj_list:
            self.add_node(child)

        #Adds the hyperedge
        #TODO: check if the hashing is being done correctly here by __hash__
        if parent_obj in self.hyperedges:

            # #TODO:
            # #Symptom: operators at the hypergraph nodes where being duplicated,
            # #even though actions the hypergraph models (the operator names),
            # #were not.
            # #
            # #Debug conclusions: operators were using their memory ids as hash
            # #keys, causing operators with the same action (name) to be considered
            # #different objects. The duplication would manifest itself when a node
            # #already with outgoing hyperedges (parent_obj in self.hyperedges) was
            # #later dequed and given the same operator. Fortunately, the tests
            # #revealed that different operators with the same action would yield
            # #the same set of children nodes, which indicates that the expansion
            # #is correctly implemented.
            #
            # if op_obj in self.hyperedges[parent_obj]:
            #     #TODO: this should be removed, once I'm confident that the algorithm
            #     #is handling the hypergraph correctly. It checks whether the two
            #     #copies of the same operator yielded the same children at the same
            #     #parent node (which is a requirement), and opens a debugger if they
            #     #don't
            #     prev_children = self.hyperedges[parent_obj][op_obj]
            #     if len(prev_children)!=len(child_obj_list):
            #         print('WARNING: operator %s at node %s yielded children lists with different lengths'%(op_obj.name,parent_obj.name))
            #         import ipdb; ipdb.set_trace()
            #         pass
            #
            #     for child in child_obj_list:
            #         if not child in prev_children:
            #             print('WARNING: operator %s at node %s yielded different sets of children'%(op_obj.name,parent_obj.name))
            #             import ipdb; ipdb.set_trace()
            #             pass

            self.hyperedges[parent_obj][op_obj]=child_obj_list
        else:
            self.hyperedges[parent_obj]={op_obj:child_obj_list}

        #Records the mapping from children to parent nodes
        for child in child_obj_list:
            if not (child in self.parents):
                self.parents[child]=set()
            self.parents[child].add(parent_obj)

    def remove_all_hyperedges(self,node):
        """Removes all hyperedges at a node."""
        if node in self.hyperedges:
            del self.hyperedges[node]

    def all_node_operators(self,node):
        """List of all operators at a node."""
        return list(self.hyperedges[node].keys()) if node in self.hyperedges else []

    def all_node_ancestors(self,node):
        """Set of all node parents, considering all hyperedges."""
        if node in self.parents:
            return self.parents[node]
        else:
            return set()

    def hyperedge_successors(self,node,act):
        """List of children associated to a hyperedge."""
        if node in self.hyperedges and act in self.hyperedges[node]:
            return self.hyperedges[node][act]
        else:
            return []

    def has_node(self,node):
        """Returns whether the hypergraph contains the node"""
        return (node in self.nodes)

    def has_operator(self,op):
        """Returns whether the hypergraph contains the operator."""
        return (op in self.operators)

    def has_ancestor(self,node):
        """Whether a node has ancestors in the graph."""
        return (node in self.parents)
