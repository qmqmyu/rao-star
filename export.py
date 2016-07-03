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

Functions for exporting RAO* data structures into other formats.

@author: Pedro Santana (psantana@mit.edu).
"""
try:
    from rmpyl.rmpyl import RMPyL, Episode
    from rmpyl.constraints import TemporalConstraint,ChanceConstraint
    _RMPYL_FOUND=True
except ImportError:
    _RMPYL_FOUND=False; print('RMPyL not found. Cannot export policies to RMPyL programs or TPNs.')

try:
    import pydot
    _PYDOT_FOUND=True
except ImportError:
    _PYDOT_FOUND=False; print('PyDot not found. Cannot export policies to Graphviz.')

if _PYDOT_FOUND:
    def policy_to_dot(G,policy):
        """Returns a Dot representation of the RAO* policy graph."""

        def _make_node_label(node):
            label= "State:%s\nValue:%.4f Entropy:%.3f\nR:%.4f ER:%.4f ERB:%.4f"%(node.name[-5:-1],
                                                                                 node.value,
                                                                                 node.state.entropy,
                                                                                 node.risk,
                                                                                 node.exec_risk,
                                                                                 node.exec_risk_bound)
            if node in policy:
                label += "\nOp: %s, OpValue: %.4f"%(policy[node].name,policy[node].op_value)

            return label

        #DIrected graph, Left to Right (LR)
        #dot_graph = pydot.Dot(graph_type="digraph",rank="same",rankdir="LR")
        dot_graph = pydot.Dot(graph_type="digraph")

        for node in [G.nodes[n] for n in policy]:
            node_label = _make_node_label(node)

            dot_graph.add_node(pydot.Node(node.name,
                                          style="filled",
                                          fillcolor="white",
                                          label=node_label))
            op = policy[node]
            for child_idx,child in enumerate(G.hyperedge_successors(node,op)):
                child_label = _make_node_label(child)

                dot_graph.add_node(pydot.Node(child.name,
                                              style="filled",
                                              fillcolor="white",
                                              label=child_label))
                child_label = str(op.properties['prob'][child_idx])
                if len(op.properties['obs'][child_idx])>0:
                    child_label +='\n'+op.properties['obs'][child_idx]

                dot_graph.add_edge(pydot.Edge(src=node.name,dst=child.name,
                                              label=child_label))
        return dot_graph
else:
    def policy_to_dot(*args):
        print('WARNING: PyDot not found. Cannot export policies to Graphviz.')
        return None

if _RMPYL_FOUND:

    def policy_to_rmpyl(G,policy,name='run()',constraint_fields=['constraints'],
                        global_start=None,global_end=None):
        """
        Returns an RMPyL program corresponding to the RAO* policy, which can
        be subsequently converted into a pTPN.
        """
        prog = RMPyL(name=name)
        constraints = set()
        prog *= _recursive_convert(G,prog,G.root,policy,constraints,constraint_fields,global_end)
        if global_start!=None:
            constraints.add(TemporalConstraint(start=global_start,end=prog.first_event,
                                               ctype='controllable',lb=0.0,ub=float('inf')))
        _add_constraints(prog,constraints)
        return prog

    def _recursive_convert(G,prog,node,policy,constraints,constraint_fields,global_end):
        """Recursively converts a policy graph into an RMPyL program."""
        #Collects constraints in the node
        constraints.update(_get_state_constraints(node,constraint_fields))

        if node.terminal:
            stop_episode = Episode(action='__stop__') #Stop action
            if global_end != None:
                constraints.add(TemporalConstraint(start=stop_episode.end,
                                                    end=global_end,
                                                    ctype='controllable',
                                                    lb=0.0,ub=float('inf')))
            return stop_episode
        else:
            op = policy[node] #Operator at current node
            children = G.hyperedge_successors(node,op)
            obs_domain = op.properties['obs']
            obs_probs = op.properties['prob']

            if op.properties['episode']==None:
                ep = Episode(action=op.name)
            else:
                ep = op.properties['episode']

            #Probabilistic transitions with more than one option require an
            #observation node.
            if len(obs_domain)>1:
                return prog.sequence(
                        ep,
                        prog.observe({'name':'observe-'+op.name,
                                      'ctype':'probabilistic',
                                      'domain':obs_domain,
                                      'probability':obs_probs},
                                      *[_recursive_convert(G,prog,c,policy,
                                                           constraints,
                                                           constraint_fields,
                                                           global_end) for c in children]))
            #If there is only a single transition, no need for an observation.
            elif len(obs_domain)==1:
                return prog.sequence(ep,_recursive_convert(G,prog,children[0],
                                                           policy,constraints,
                                                           constraint_fields,
                                                           global_end))
            else:
                raise ValueError('Cannot have a probabilitistic transition with 0 children!')

    def _get_state_constraints(node,constraint_fields):
        """
        Extracts constraint objects from the constraint fields in a node.
        """
        constraint_set=set()
        for particle_key,(state,prob) in node.state.belief.items():
            for cf in constraint_fields:
                if cf in state:
                    for c in state[cf]:
                        if isinstance(c,(TemporalConstraint,ChanceConstraint)):
                            constraint_set.add(c)
                        else:
                            raise TypeError('Invalid type of constraint or not implemented!')
        return constraint_set

    def _add_constraints(prog,constraints):
        """
        Adds constraints to an RMPyL program
        """
        prog_tc = prog.temporal_constraints
        for c in constraints:
            if isinstance(c,TemporalConstraint):
                add_tc=True
                for existing_tc in prog_tc:
                    if existing_tc.is_equivalent(c):
                        add_tc = False; break
                if add_tc:
                    prog.add_temporal_constraint(c)
            elif isinstance(c,ChanceConstraint):
                prog.add_chance_constraint(c)
            else:
                raise TypeError('Invalid type of constraint or not implemented!')

else:
    def policy_to_rmpyl(*args):
        print('WARNING: RMPyL not found. Cannot export policies to RMPyL programs or TPNs.')
        return None
