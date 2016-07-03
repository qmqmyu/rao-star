#!/usr/bin/env python
#
#  Copyright (c) 2016 MIT. All rights reserved.
#
#   author: Pedro Santana, Tiago Vaquero
#   e-mail: psantana@mit.edu, tvaquero@mit.edu
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

Demo of RAO* being used to generate plans for the Mitsubishi demo.

@author: Pedro Santana (psantana@mit.edu), Tiago Vaquero (tvaquero@mit.edu).
"""

from .pddl_planner import PDDL_Planner
from rao.pddl.model_parser import model_parser

import sys
import os
import subprocess

class External_PDDL_Planner(PDDL_Planner):
    """
    A PDDL planner that is called externally (executable) through a subprocess.
    """
    def __init__(self,domain_file, problem_file, arguments=['domain','problem','output'], domain=None, problem=None, task=None, folder='popf',executable='plan'):
        self.name = executable
        self.domain_file = domain_file
        self.problem_file = problem_file
        self.domain = domain
        self.problem = problem
        self.task = task

        self.arguments = arguments

        self.path = os.path.dirname(os.path.abspath(__file__))
        #TODO: set planner executable as a parameter
        self.executable_dir = os.path.join(self.path,folder)
        self.executable_path = os.path.join(self.executable_dir,executable)
        self.plan_file = os.path.join(self.executable_dir,"plan.soln")

        if domain == None or problem == None or task == None:
            self.domain,self.problem,self.task = model_parser(self.domain_file,
                                                              self.problem_file,
                                                              pddl_pickle=prob_file+'_task.pickle',
                                                              verbose=True)

    def plan(self, initial_state=None):
        """
        Calls a PDDL planner
        """
        if not initial_state == None:
            self._generate_pddl_problem(initial_state)
        #self._generate_pddl_problem(initial_state)

        # clear plan
        plan = []
        # delete/clear current solution files
        if os.path.isfile(self.plan_file):
            os.remove(self.plan_file)

        #print("Domain:" + self.domain_file)
        #print("Problem:" + self.problem_file)
        # call the planner
        if os.path.exists(self.domain_file) and os.path.exists(self.problem_file):
            #print("Calling %s planner" % self.name)
            #subprocess.call([self.executable_path,self.domain_file, self.problem_file, plan_file],cwd=os.path.expanduser(self.executable_dir)
            FNULL = open(os.devnull, 'w')
            subprocess.call(self._get_arguments(),cwd=os.path.expanduser(self.executable_dir),stdout=FNULL, stderr=subprocess.STDOUT)
            #subprocess.call([self.executable_path, '--domain',self.domain_file, '--problem', self.problem_file, '--output',plan_file],cwd=os.path.expanduser(self.executable_dir), stdout=FNULL, stderr=subprocess.STDOUT)
            if os.path.isfile(self.plan_file):
                plan_str = ""
                with open(self.plan_file) as pddl_plan_content_file:
                    plan_str = pddl_plan_content_file.read().strip().lower()

                #print("PDDL Plan: \n" + plan_str)
                # remove the plan file
                os.remove(self.plan_file)


                plan = self._parse_pddl_plan(plan_str)
        else:
            print("Domain file or Problem Instance file could not be found. Aborting planning!")

        return plan


    def _parse_pddl_plan(self,plan_str):
        """
        Parse a PDDL plan to a list of actions in which each element is in the form
        (action, start_time, duration)
        """
        plan = []
        for line in plan_str.splitlines():
            line_str = line.strip()
            if line_str == "":
                continue
            # check if the line is not a comment
            if not line_str.startswith(";"):
                if line_str.find(':') >= 0 and line_str.find('['):
                    start_time = float(line_str[:line_str.index(":")])
                    action = line_str[line_str.index("(") : line_str.index(")") + 1]
                    duration = float(line_str[line_str.index("[")  + 1: line_str.index("]") ])
                    plan.append((action, start_time, duration))
                else:
                    plan.append(line_str)

        #print plan
        return plan


    def _generate_pddl_problem(self,initial_state=None):
        """
        Parse a Problem object back to a PDDL problem string.
        """

        #if not initial_state == None and len(initial_state) > 0:
        #    self.problem.initial_state = initial_state

        problem_str = '(define (problem ' + self.problem.name + ') \n'
        problem_str += '    (:domain ' + self.domain.name + ') \n'
        # adding objects
        problem_str += '    (:objects \n'
        for obj in self.problem.objects:
            obj_name = obj
            obj_type = self.problem.objects[obj]
            problem_str += '        ' + obj_name + ' - ' + obj_type.name +' \n'
        problem_str += '    ) \n'

        #adding initial state
        problem_str += '    (:init \n'
        init = initial_state.union(self.task.static)
        for predicate in init:
            problem_str += '        ' + self._to_predicate_str(predicate) +' \n'
        problem_str += '    ) \n'

        #adding goals
        problem_str += '    (:goal \n'
        problem_str += '        (and \n'
        for goal in self.problem.goal:
            problem_str += '            ' + self._to_predicate_str(goal) +' \n'
        problem_str += '        ) \n'
        problem_str += '    ) \n'
        problem_str += ')'


        #print(problem_str)
        with open(self.problem_file, 'w') as problem_file:
            problem_file.write(problem_str)


    def _to_predicate_str(self,predicate=None):
        """
        Parses a Predicate object back to a PDDL predicate string.
        """
        predicate_str = ''
        if not predicate == None:
            if type(predicate) == str: #check it it is already in a str format
                predicate_str = predicate
            else:
                predicate_str = '('+ predicate.name
                for parameter in predicate.signature:
                    predicate_str += ' ' + parameter[0] #get the object name, parameter[1] would give the type
                predicate_str += ')'
        return predicate_str


    def _get_arguments(self):
        """
        Builds the argument list to run the planner
        """
        argument_list=[self.executable_path]
        if self.arguments == [] or self.arguments == ['domain','problem','output']:
            argument_list = [self.executable_path,self.domain_file, self.problem_file, self.plan_file]
        else:
            for arg in self.arguments:
                if arg == 'domain':
                    argument_list.append(self.domain_file)
                elif arg == 'problem':
                    argument_list.append(self.problem_file)
                elif arg == 'output':
                    argument_list.append(self.plan_file)
                else:
                    argument_list.append(arg)

        return argument_list



dom_file = '/home/tiago/mers/rss/catkin_ws/src/rss_work/rss_git/enterprise/ros/model/domain-strips.pddl'#'rss-domain-strips.pddl')
prob_file = '/home/tiago/mers/rss/catkin_ws/src/rss_work/rss_git/enterprise/ros/model/current_problem.pddl'#'rss-current-problem-strips.pddl')

if __name__ == '__main__':

    planner = External_PDDL_Planner(domain_file=dom_file,problem_file=prob_file)

    plan = planner.plan()
    print(plan)
