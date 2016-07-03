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
RMPyL program modeling a collaborative pick-and-place task between a human and a robot.
"""
from rmpyl.rmpyl import RMPyL, Episode
import time
import pickle

def say(text):
    """Final message to the human."""
    return Episode(action=('(say \"%s\")'%text).lower())

def pick_and_place_block(prog,block,pick_loc,place_loc,manip,agent,dur_dict=None):
    """"Picks a block and places it somewhere."""
    obj=block+'Component'

    duration = dur_dict if dur_dict != None else {'ctype':'controllable','lb':0.0,'ub':float('inf')}

    return prog.sequence(say('Going to pick %s'%obj),
                         Episode(action=('(pick %s %s %s %s)'%(obj,manip,pick_loc,agent)).lower(),
                                 duration=duration),
                         Episode(action=('(place %s %s %s %s)'%(obj,manip,place_loc,agent)).lower(),
                                 duration=duration))

def observe_decide_act(prog,blocks,manip,agent,dur_dict=None):
    """Final message to the human."""
    if len(blocks)>0:
        #Human helped with one of the blocks
        human_help = [observe_decide_act(prog,[ob for ob in blocks if ob!=b],manip,agent) for b in blocks]
        #No help from the human
        no_human_help = prog.sequence(
                            prog.decide(
                                {'name':'block-choice',
                                 'domain':blocks,
                                 'utility':range(len(blocks))},
                                 *[prog.sequence(
                                    pick_and_place_block(prog,b,b+'Bin',b+'Target',manip,agent,dur_dict),
                                    observe_decide_act(prog,[ob for ob in blocks if ob!=b],manip,agent)) for b in blocks]))

        #All episodes
        all_episodes = human_help+[no_human_help]

        #Observe each one of the blocks
        observations=blocks+['none']
        say_text = 'Checking if the human moved %s components or %s'%(','.join(observations[:-1]),observations[-1])
        return prog.sequence(
                    say(say_text),
                    prog.observe({'name':'observe-human-%d'%(len(blocks)),
                                  'ctype':'probabilistic',
                                  'domain':observations,
                                  'probability':([0.3/len(blocks)]*len(blocks))+[0.7]},
                                  *all_episodes))

    else:
        return say('All done!')

def nominal_case(blocks,time_window=-1,dur_dict=None):
    """
    Nominal case, where the robot observes what the human has already completed,
    and acts accordingly
    """
    agent='Baxter'
    manip='BaxterRight'

    prog = RMPyL(name='run()')

    prog *= prog.sequence(say('Should I start?'),
                          prog.observe({'name':'ask-human',
                                        'ctype':'probabilistic',
                                        'domain':['YES','NO'],
                                        'probability':[0.9,0.1]},
                                        observe_decide_act(prog,blocks,manip,agent,dur_dict),
                                        say('All done!')))
    if time_window>0.0:
        prog.add_overall_temporal_constraint(ctype='controllable',lb=0.0,ub=time_window)

    return prog

def collaborative_pick_and_place(blocks,time_window=-1.0,dur_dict=None,write_tpn=True,write_pickle=False):
    """
    Collaborative pick-and-place task described in RMPyL.
    """
    print('Generating RMPyL program.'); start_t=time.time()
    prog = nominal_case(blocks,time_window,dur_dict)
    print('Done in %.3f s'%(time.time()-start_t))

    print('\n##### Plan stats:')
    print('Name: %s'%(prog.name))
    print('Events: %d'%(len(prog.events)))
    print('Primitive episodes: %d'%(len(prog.primitive_episodes)))
    print('Choices: %d'%(len(prog.choices)))
    print('Temporal constraints: %d'%(len(prog.temporal_constraints)))

    if write_tpn:
        filename='observe_blocks_and_act_%d_blocks'%(len(blocks))
        print('\nDumping program to TPN file...'); start_t=time.time()
        prog.to_ptpn(filename=filename+'.tpn')
        print('Done in %.3f s'%(time.time()-start_t))

    if write_pickle:
        print('\nDumping program to Pickle file...'); start_t=time.time()
        with open(filename+'.pickle','wb') as f:
            pickle.dump(prog,f)
        print('Done in %.3f s'%(time.time()-start_t))

    return prog


if __name__=='__main__':
    objects=['Red','Green']
    collaborative_pick_and_place(objects,write_tpn=True,write_pickle=True)
