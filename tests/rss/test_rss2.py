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

Demo of RAO* being used to generate plans for the Mitsubishi demo.

@author: Pedro Santana (psantana@mit.edu).
"""
import os, sys
import time
from rao.raostar import RAOStar
from rao.external_planners.pysat import PySAT
from rao.export import policy_to_dot,policy_to_rmpyl
from rao.models.rss_durative_pddl import RSSDurativePDDL, time_window_constraints
from rao.pddl.model_parser import model_parser

from rmpyl.rmpyl import RMPyL
from rmpyl.episodes import Episode
from rmpyl.defs import Event
from rmpyl.constraints import TemporalConstraint

from rss_model_utils import rss_duration_func,rss_time_window_func
from pytemporal.paris import PARIS
import ipdb

#TODO: this should be passed as arguments
#PDDL files for the RSS demo
# path = os.path.dirname(os.path.abspath(__file__))
# dom_file = os.path.join(path, '/home/tiago/mers/rss/catkin_ws/src/rss_work/rss_git/enterprise/ros/model/domain-strips.pddl')#'rss-domain-strips.pddl')
# prob_file = os.path.join(path, '/home/tiago/mers/rss/catkin_ws/src/rss_work/rss_git/enterprise/ros/model/current_problem.pddl')#'rss-current-problem-strips.pddl')
# output_file = '/home/tiago/mers/rss/catkin_ws/src/rss_work/rss_git/enterprise/ros/model/plan.tpn' #os.path.join(path,'rss_policy_rmpyl_ptpn.tpn')
# output_graph_file = '/home/tiago/mers/rss/catkin_ws/src/rss_work/rss_git/enterprise/ros/model/plan_graph.svg' #os.path.join(path,'rss_policy.svg')

duration_model = {}
time_windows = {}


def rss_duration_model_func(action):
    """
    Function mapping from PDDL (string) operators in the RSS domain to activity
    durations.
    """
    act_tokens = action.strip('() ').split(' ')
    duration_dict = None
    if act_tokens[0].lower() in duration_model:
        if act_tokens[0].lower() == 'move':
            if act_tokens[2]==act_tokens[3]:
                duration_dict = {'ctype':'uncontrollable_bounded','params':{'lb':0,'ub':0}}
            for i in [0,1]:
                if (act_tokens[i+2],act_tokens[3-i]) in duration_model['move']:
                    duration_dict = duration_model['move'][(act_tokens[i+2],act_tokens[3-i])]
                    break
        else:
            duration_dict = duration_model[act_tokens[0]]

    #print(duration_dict)
    if duration_dict ==None:
        raise ValueError('No duration information for '+action)
    else:
        rmpyl_dict = {'ctype':duration_dict['ctype']}
        rmpyl_dict.update(duration_dict['params'])
        return rmpyl_dict


def rss_time_window_model_func(action):
    """
    Function mapping from PDDL (string) operators in the RSS domain to execution
    time windows
    Example of time_windows dictionaly:
    {'activity_dependencies':
                     {'transmit_data': {'orbiter_communication_available': {
                                                'temporal_constraint': ['overall'],
                                                'parameters_mapping': [[1, 0]]}}},
     'time_windows': {'orbiter_communication_available': {tuple({'l2'}): {'start': 10, 'end': '2188'},
                                                          tuple({'l4'}): {'start': 10, 'end': '2188'}}}}
    """

    #default values
    constraint_type = 'overall'  #TODO: this should be a list
    bounds = [0.0,float('inf')]

    no_time_window_case = False

    # print(time_windows)
    activity_dependencies = time_windows['activity_dependencies']
    current_time_windows = time_windows['time_windows']

    act_tokens = action.strip('() ').split(' ')
    operator = act_tokens[0].lower()
    if operator in activity_dependencies:
        precond_list = activity_dependencies[operator]
        precond_name = list(precond_list.keys())[0]
        precond_info = precond_list[precond_name]
        #try to find the precondition in the current time windows
        if precond_name in current_time_windows:
            precond_time_window = current_time_windows[precond_name]

            constraint_type = precond_info['temporal_constraint'][0] #TODO: this should be a list, we shouldl not pick the first one only
            parameters_mapping = precond_info['parameters_mapping'][0] #TODO: this should handle a list of parameters not just the first
            #get the parameter
            precond_parameter = act_tokens[1:][parameters_mapping[0]]
            #print(precond_parameter)

            if frozenset([precond_parameter]) in precond_time_window:
                lb_up = precond_time_window[frozenset([precond_parameter])]
                if 'start' in lb_up:
                    bounds[0] = float(lb_up['start'])
                if 'end' in lb_up:
                    bounds[1] = float(lb_up['end'])
            else:
                no_time_window_case = True
        else:
            no_time_window_case = True

    if no_time_window_case:
        print('TIME WINDOW FOR ACTION %s NOT FOUND!' % action)
        #TODO: what should we set here?

    #print(action)
    #print(bounds)
    return bounds,constraint_type

def make_episode_id(t,op):
    return ('%s-%d'%(op,t)).replace(' ','-').replace('(','').replace(')','')

if __name__ == '__main__':

    dom_file = sys.argv[1]
    prob_file = sys.argv[2]
    duration_model_file = sys.argv[3]
    time_windows_file = sys.argv[4]
    output_file = sys.argv[5]
    output_graph_file = output_file + '.svg'

    # reads activity duration model (as a dictionary) from file
    with open(duration_model_file) as content_file:
        duration_model = eval(content_file.read())

    # reads time wondows model (as a dictionary) from file
    with open(time_windows_file) as content_file:
        time_windows = eval(content_file.read())


    py_sat = PySAT(dom_file,prob_file,precompute_steps=40,remove_static=True,verbose=True)
    domain,problem,task = model_parser(dom_file,prob_file,remove_static=True)

    start = time.time()
    # sat_plans = py_sat.plan(task.initial_state,task.goals,time_steps=30,find_shortest=True) #find optimal plan size
    sat_plans = py_sat.plan(task.initial_state,task.goals,time_steps=18,find_shortest=True) # find sub-optimal plan
    # print("---------plan: %d" % len(sat_plans))
    if len(sat_plans)>0:
        plan = sat_plans[0] # get the first plan (default returns a list with one plan)
        for t,op in enumerate(plan):
            print('%d: %s'%(t,op))

        elapsed = time.time()-start
        print('\n##### All solving took %.4f s'%(elapsed))

        prog = RMPyL(name='run()')
        pddl_episodes = [Episode(id=make_episode_id(t,op),
                                 start=Event(name='start-of-%d-%s'%(t,op)),
                                 end=Event(name='end-of-%d-%s'%(t,op)),
                                 action=op,
                                 duration=rss_duration_model_func(op)) for t,op in enumerate(plan)]
        prog.plan = prog.sequence(*pddl_episodes)
        # prog.add_overall_temporal_constraint(ctype='controllable',lb=0.0,ub=2000.0)
        #Adds temporal window to the plan
        for t,op in enumerate(plan):
            bounds, tc_type = rss_time_window_model_func(op)
            for tc in time_window_constraints(tc_type,bounds,prog.first_event,prog.episode_by_id(make_episode_id(t,op))):
                prog.add_temporal_constraint(tc)

        #Dummy episodes that enable transmissions
        activation_episodes=[]
        activation_tcs=[]
        global_start=Event(name='global-start')

        for op_name,op_param_dict in time_windows['time_windows'].items():
            for arg_set,window_dict in op_param_dict.items():
                for ev_type,time_bound in window_dict.items():
                    orb_ep_id='%s_event_%s-%s'%(ev_type,op_name,'-'.join(arg_set))
                    activation_episodes.append(Episode(id=orb_ep_id,
                                                    action='(%s)'%(orb_ep_id.replace('-',' ')),
                                                    duration={'ctype':'controllable','lb':0.005,'ub':0.1}))
                    activation_tcs.append(TemporalConstraint(start=global_start,
                                                             end=activation_episodes[-1].start,
                                                             ctype='controllable',
                                                             lb=float(time_bound),
                                                             ub=float(time_bound)))

        global_prog = RMPyL(name='run()')
        global_prog *= global_prog.parallel(prog.plan,*activation_episodes,start=global_start)

        activation_tcs.append(TemporalConstraint(start=global_start,end=prog.first_event,
                                                 ctype='controllable',lb=0.0,ub=0.005))

        for tc in activation_tcs:
            global_prog.add_temporal_constraint(tc)

        # rmpyl_policy.to_ptpn(filename=output_file)
        #ipdb.set_trace()

        global_prog.to_ptpn(filename=output_file+'_test')
        paris = PARIS()

        # risk_bound,sc_schedule = paris.stnu_reformulation(rmpyl_policy,makespan=True,cc=0.001)

        risk_bound,sc_schedule = paris.stnu_reformulation(global_prog,makespan=True,cc=0.001)

        if risk_bound != None:
            risk_bound = min(risk_bound,1.0)
            print('\nSuccessfully performed STNU reformulation with scheduling risk %f %%!'%(risk_bound*100.0))


            for tc in global_prog.temporal_constraints:
                if tc.type == 'uncontrollable_bounded':
                    tc.type = 'controllable'

            for i in range(len(pddl_episodes)-1):
                global_prog.add_temporal_constraint(TemporalConstraint(start=pddl_episodes[i].end,end=pddl_episodes[i+1].start,
                                                                       ctype='controllable',lb=1.005,ub=2.1))

            global_prog.simplify_temporal_constraints()


            # activation_episode_end_events = [ep.end for ep in activation_episodes]
            # for tc in global_prog.temporal_constraints:
            #     if tc.start in activation_episode_end_events:
            #         global_prog.remove_temporal_constraint(tc)

            global_prog.to_ptpn(filename=output_file)

            print('\nThis is the schedule:')
            for e,t in sorted([(e,t) for e,t in sc_schedule.items()],key=lambda x: x[1]):
                print('\t%s: %.2f s'%(e,t))

        else:
            print('\nFailed to perform STNU reformulation...')
    else:
        print('Oh, sweet Jesus! There is no plan! HEEEEELP!')



    # rss_model = RSSDurativePDDL(domain_file=dom_file,prob_file=prob_file,
    #                             perform_scheduling=True,
    #                             duration_func=rss_duration_model_func,
    #                             time_window_func=rss_time_window_model_func,
    #                             verbose=1)

    # time_window = TemporalConstraint(start=rss_model.global_start_event,
    #                                 end=rss_model.global_end_event,
    #                                 ctype='controllable',lb=0.0,ub=610.0)

    # b0 = rss_model.get_initial_belief(constraints=[time_window])
    #
    # planner = RAOStar(rss_model,node_name='id',cc=0.05,cc_type='overall',
    #                   terminal_prob=1.0,randomization=0.0,propagate_risk=True,
    #                   verbose=1,log=False)

    # policy,explicit,performance = planner.search(b0)
    #
    # dot_policy = policy_to_dot(explicit,policy)
    # dot_policy.write(output_graph_file,format='svg')

    # rmpyl_policy = policy_to_rmpyl(explicit,policy,
    #                                constraint_fields=['constraints'],
    #                                global_end=rss_model.global_end_event)




    # rss_model = RSSDurativePDDL(domain_file=dom_file,prob_file=prob_file,
    #                             perform_scheduling=False,
    #                             duration_func=rss_duration_func,
    #                             time_window_func=rss_time_window_func,
    #                             verbose=1)
    #
    # time_window = TemporalConstraint(start=rss_model.global_start_event,
    #                                  end=rss_model.global_end_event,
    #                                  ctype='controllable',lb=0.0,ub=610.0)
    #
    # b0 = rss_model.get_initial_belief(constraints=[time_window])
    #
    # planner = RAOStar(rss_model,node_name='id',cc=1.0,cc_type='overall',
    #                   terminal_prob=1.0,randomization=0.0,propagate_risk=True,
    #                   verbose=1,log=False)
    #
    #
    # policy,explicit,performance = planner.search(b0)
    # dot_policy = policy_to_dot(explicit,policy)
    # rmpyl_policy = policy_to_rmpyl(explicit,policy,
    #                                constraint_fields=['constraints'],
    #                                global_end=rss_model.global_end_event)
    #
    # dot_policy.write(output_graph_file,format='svg')
    #
    # paris = PARIS()
    # risk_bound,sc_schedule = paris.stnu_reformulation(rmpyl_policy)
    # if risk_bound != None:
    #     print('\nSuccessfully performed STNU reformulation!')
    #     rmpyl_policy.to_ptpn(filename=output_file)
    # else:
    #     print('\nFailed to perform STNU reformulation...')
