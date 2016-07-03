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

Module that parses PDDL descriptions, with the option of dumping them to pickle
files.

@author: Pedro Santana (psantana@mit.edu).
"""
from rao.pddl.parser import Parser
import rao.pddl.grounding as grounding
import pickle

def model_parser(domain_file,prob_file,pddl_pickle='',remove_static=True,verbose=False):
    """
    Function that parses a PDDL domain and problem description and returns the
    corresponding objects. Alternatively, it can dump the PDDL task description
    to a pickle file.
    """
    pddlparser = Parser(domFile=domain_file)
    domain = pddlparser.parse_domain()

    #PDDL problem
    pddlparser.set_prob_file(prob_file)
    problem =pddlparser.parse_problem(domain)

    #Grounded PDDL task
    task = grounding.ground(problem,remove_static=remove_static)

    if len(pddl_pickle)>0:
        print('\n### Dumping PDDL objects to file '+pddl_pickle)
        with open(pddl_pickle,'wb') as f:
            pickle.dump((domain,problem,task),f,protocol=2)
        print('\n### Pickling complete.')

    if verbose:
        print('\n### PDDL domain from %s:'%(domain_file))
        print(domain)
        print('\n### PDDL problem from %s:'%(prob_file))
        print(problem)
        print('\n### Grounded PDDL task:')
        print(task)

    return domain,problem,task
