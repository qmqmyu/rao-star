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
RMPyL programs implementing the contingent plans in the AFOSR demos
"""
from rmpyl.rmpyl import RMPyL, Episode
import time
import pickle

def say(text):
    """Final message to the human."""
    return Episode(action=('(say \"%s\")'%text).lower())

def pick_and_place_block(prog,block,pick_loc,place_loc,manip,agent):
    """"Picks a block and places it somewhere."""
    obj=block+'Component'
    return prog.sequence(say('Going to pick %s'%obj),
                         Episode(action=('(pick %s %s %s %s)'%(obj,manip,pick_loc,agent)).lower()),
                         Episode(action=('(place %s %s %s %s)'%(obj,manip,place_loc,agent)).lower()))

def observe_and_act(prog,blocks,manip,agent):
    """Final message to the human."""
    if len(blocks)>0:
        #Human helped with one of the blocks
        human_help = [observe_and_act(prog,[ob for ob in blocks if ob!=b],manip,agent) for b in blocks]
        #No help from the human
        no_human_help = prog.sequence(pick_and_place_block(prog,blocks[0],blocks[0]+'Bin',blocks[0]+'Target',manip,agent),
                                      observe_and_act(prog,blocks[1:],manip,agent))
        #All episodes
        all_episodes = human_help+[no_human_help]

        #Observe each one of the blocks
        observations=blocks+['none']
        return prog.sequence(say('Checking if the human moved %s components or %s'%(','.join(observations[:-1]),observations[-1])),
                             prog.observe({'name':'observe-human-%d'%(len(blocks)),
                                           'ctype':'uncontrollable',
                                           'domain':observations},
                                           *all_episodes))

    else:
        return say('All done!')

def nominal_case(blocks):
    """
    Nominal case, where the robot observes what the human has already completed,
    and acts accordingly
    """
    agent='Baxter'
    manip='BaxterRight'

    prog = RMPyL(name='run()')

    prog *= prog.sequence(say('Should I start?'),
                          prog.observe({'name':'observe-human-%d'%(len(blocks)),
                                        'ctype':'uncontrollable',
                                        'domain':['YES','NO']},
                                        observe_and_act(prog,blocks,manip,agent),
                                        say('All done!')))
    return prog

if __name__=='__main__':

    #blocks=['Red','Green','Blue','Yellow']
    #blocks=['Red','Green','Blue']
    blocks=['Red','Green']

    print('\"Compiling\" the cRMPL program.'); start_t=time.time()
    prog = nominal_case(blocks)
    print('Done in %.3f s'%(time.time()-start_t))

    print('\n##### Plan stats:')
    print('Name: %s'%(prog.name))
    print('Events: %d'%(len(prog.events)))
    print('Primitive episodes: %d'%(len(prog.primitive_episodes)))
    print('Choices: %d'%(len(prog.choices)))
    print('Temporal constraints: %d'%(len(prog.temporal_constraints)))

    filename='observe_blocks_and_act_%d_blocks'%(len(blocks))
    print('\nDumping program to TPN file...'); start_t=time.time()
    prog.to_ptpn(filename=filename+'.tpn')
    print('Done in %.3f s'%(time.time()-start_t))

    print('\nDumping program to Pickle file...'); start_t=time.time()
    with open(filename+'.pickle','wb') as f:
        pickle.dump(prog,f)
    print('Done in %.3f s'%(time.time()-start_t))
