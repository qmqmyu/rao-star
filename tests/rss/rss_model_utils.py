#!/usr/bin/env python
#
#  Copyright (c) 2016 MIT. All rights reserved.
#
#   author: Tiago Vaquero, Pedro Santana
#   e-mail: tvaquero@mit.edu, psantana@mit.edu
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
"""

#Dictionary containing the temporal properties of traversal activities
_traversal_dict = {  ('unknown','l1'):{  'type':'uncontrollable_probabilistic',
                                        'params':{'distribution':{'type':'gaussian','mean':101,'variance':10}}},
                    ('unknown','l3'):{  'type':'uncontrollable_probabilistic',
                                        'params':{'distribution':{'type':'gaussian','mean':3,'variance':1}}},
                    ('unknown','l2'):{  'type':'uncontrollable_probabilistic',
                                        'params':{'distribution':{'type':'gaussian','mean':169,'variance':15}}},
                    ('unknown','l5'):{  'type':'uncontrollable_probabilistic',
                                        'params':{'distribution':{'type':'gaussian','mean':281,'variance':20}}},
                    ('unknown','l4'):{  'type':'uncontrollable_probabilistic',
                                        'params':{'distribution':{'type':'gaussian','mean':160,'variance':9}}},
                    ('l1','l2'):{  'type':'uncontrollable_probabilistic',
                                        'params':{'distribution':{'type':'gaussian','mean':217,'variance':5}}},
                    ('l1','l3'):{  'type':'uncontrollable_probabilistic',
                                        'params':{'distribution':{'type':'gaussian','mean':105,'variance':5}}},
                    ('l1','l4'):{  'type':'uncontrollable_probabilistic',
                                        'params':{'distribution':{'type':'gaussian','mean':122,'variance':5}}},
                    ('l1','l5'):{  'type':'uncontrollable_probabilistic',
                                        'params':{'distribution':{'type':'gaussian','mean':258,'variance':5}}},
                    ('l2','l3'):{  'type':'uncontrollable_probabilistic',
                                        'params':{'distribution':{'type':'gaussian','mean':171,'variance':5}}},
                    ('l2','l4'):{  'type':'uncontrollable_probabilistic',
                                        'params':{'distribution':{'type':'gaussian','mean':166,'variance':5}}},
                    ('l2','l5'):{  'type':'uncontrollable_probabilistic',
                                        'params':{'distribution':{'type':'gaussian','mean':202,'variance':5}}},
                    ('l3','l4'):{  'type':'uncontrollable_probabilistic',
                                        'params':{'distribution':{'type':'gaussian','mean':163,'variance':5}}},
                    ('l3','l5'):{  'type':'uncontrollable_probabilistic',
                                        'params':{'distribution':{'type':'gaussian','mean':284,'variance':5}}},
                    ('l4','l5'):{  'type':'uncontrollable_probabilistic',
                                        'params':{'distribution':{'type':'gaussian','mean':137,'variance':5}}}}

#Dictionary mapping durative PDDL activities to their temporal parameters
_duration_model = { 'take_pictures_mastcam':{   'type':'uncontrollable_bounded',
                                                'params':{'lb':15,'ub':20}},
                    'take_pictures_hazcam':{    'type':'uncontrollable_bounded',
                                                'params':{'lb':15,'ub':20}},
                    'transmit_data':{           'type':'uncontrollable_probabilistic',
                                                'params':{'distribution':{'type':'uniform','lb':20,'ub':30}}},
                    'collect_rock_sample':{     'type':'uncontrollable_bounded',
                                                'params':{'lb':40,'ub':50}},
                    'survey_location':{         'type':'uncontrollable_bounded',
                                                'params':{'lb':40,'ub':50}},
                    'turnon_mastcam':{          'type':'uncontrollable_probabilistic',
                                                'params':{'distribution':{'type':'uniform','lb':15,'ub':20}}},
                    'turnon_hazcam':{          'type':'uncontrollable_probabilistic',
                                                'params':{'distribution':{'type':'uniform','lb':15,'ub':20}}}}
_duration_model['move']=_traversal_dict

def _duration_predictor(duration_model,durative_pddl_action):
    """
    Takes in a durative PDDL action and returns the type and parameters of its
    duration model.
    """
    act_tokens = durative_pddl_action.strip('() ').split(' ')
    duration_dict = None
    if act_tokens[0].lower() in duration_model:
        if act_tokens[0].lower() == 'move':
            if act_tokens[2]==act_tokens[3]:
                duration_dict = {'type':'uncontrollable_bounded','params':{'lb':0,'ub':0}}
            for i in [0,1]:
                if (act_tokens[i+2],act_tokens[3-i]) in duration_model['move']:
                    duration_dict = duration_model['move'][(act_tokens[i+2],act_tokens[3-i])]
                    break
        else:
            duration_dict = duration_model[act_tokens[0]]

    if duration_dict ==None:
        raise ValueError('No duration information for '+durative_pddl_action)
    else:
        rmpyl_dict = {'ctype':duration_dict['type']}
        rmpyl_dict.update(duration_dict['params'])
        return rmpyl_dict

def rss_duration_func(action):
    """
    Function mapping from PDDL (string) operators in the RSS domain to activity
    durations.
    """
    return _duration_predictor(_duration_model,action)


def rss_time_window_func(action):
    """
    Function mapping from PDDL (string) operators in the RSS domain to execution
    time windows
    """
    return [0.0,2200.0],'overall'

# time_window_predicate_map = {'orbiter_communication_available':{('l2',):[50,1800],
#                                                                 ('l4',):[50,1800]}}
#
#
# time_window_dict = {'transmit_data':['orbiter_communication_available','at-start']}


# def time_window_assigner(time_window_dict,time_window_predicate_map,durative_pddl_action):
#     act_tokens = durative_pddl_action.strip('() ').split(' ')
#     if act_tokens == 'transmit_data':
#         time_window_predicate_map[time_window_dict[act_tokens[0]]][act_tokens[2]]
