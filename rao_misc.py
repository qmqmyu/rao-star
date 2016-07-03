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

Miscellaneous tools.

@author: Pedro Santana (psantana@mit.edu).
"""
import numpy as np

def hash_complex_type(complex_t):
    """
    Hashes a potentially nested dictionary.
    """
    # return hash(hl.md5(str(complex_t).encode('utf-8')).hexdigest())
    if isinstance(complex_t,(list,tuple,set,np.ndarray)):
        return hash(frozenset([hash_complex_type(e) for e in complex_t]))
    elif isinstance(complex_t,dict):
        hash_dict={k:hash_complex_type(v) for k,v in complex_t.items()}
        return hash(frozenset(hash_dict.items()))
    else:
        return hash(complex_t)


def limit_precision(belief,decimals):
    """Limits the precison of probabilities in a belief."""
    max_hash = max(belief.keys()); acc_prob=0.0
    for k,v in belief.items():
        prob = np.round(v[1],decimals=decimals)
        belief[k][1]=prob
        acc_prob+=prob

    #The state with the highest hash will have the normalized probability
    belief[max_hash][1] = 1.0-acc_prob+belief[max_hash][1]
    return belief

def policy_entropy(policy,max_entropy=True):
    """
    Returns the entropies of belief states down a policy.
    """
    nodes = policy.keys()
    max_depth=max([n.depth for n in nodes])
    entropy_list=[[] for i in range(max_depth+1)]
    for node in policy.keys():
        entropy_list[node.depth].append(node.state.entropy)

    #Either returns the maximum or the average entropy
    func = max if max_entropy else lambda x: sum(x)/len(x)
    return [func(l) for l in entropy_list]
