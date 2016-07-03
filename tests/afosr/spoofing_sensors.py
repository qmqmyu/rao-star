#!/usr/bin/env python
#
#  A simple Python simulator for power networks.
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
Manually spoofs the uncontrollable choices in an RMPyL program.
"""
import rospy
from pike_msgs.msg import UncontrollableChoiceSensed
from collections import deque
import sys
import pickle

class PikeSensorSpoofer(object):
    """
    Handles the spoofing of uncontrollable choices (sensors) in Pike.
    """
    def __init__(self,prog,period=0.1,verbose=1):
        self.obs_pub = rospy.Publisher('pike/uncontrollable_choice_outcomes',
                                        UncontrollableChoiceSensed,queue_size=1)

        self.obs_echo = rospy.Subscriber('pike/uncontrollable_choice_outcomes',
                                        UncontrollableChoiceSensed,self._observation_echo)

        self.send_queue = deque() #Queue of observations to be sent
        self.last_unconfirmed = None #Last unconfirmed observation message

        #Dictionary mapping observation ID's to domain values
        self.obs_dict={ob.id:ob.domain for ob in prog.observations}
        self.verbose=verbose

        #Spawns the periodic queue management activity
        rospy.Timer(rospy.Duration(period),self._queue_management)

    def __del__(self):
        print('\nShame on you for spoofing sensor inputs. SHAME!')

    def spoof(self,choice_id):
        """
        Listens to the output of Pike and checks if it is currently waiting on
        an observation (uncontrollable choice) to be published.
        """
        if choice_id in self.obs_dict:
            if self.verbose>=1:
                rospy.logwarn('PIKE SPOOFER: Spoofing choice %s'%(choice_id))
            domain = self.obs_dict[choice_id]
            val_idx = int(input('Select value by index [%s]: '%(' '.join(['%s (%d)'%(val,idx) for idx,val in enumerate(domain)]))))
            if val_idx>=0 and val_idx<len(domain):
                self.send_queue.appendleft([choice_id,domain[val_idx]])
            else:
                rospy.logwarn('\nInvalid index!\n')
        else:
            if self.verbose>=2:
                rospy.logwarn('\nInvalid choice ID!\n')

    def _queue_management(self,msg):
        """
        Manages the queue of observations to be published.
        """
        send_message=True
        if self.last_unconfirmed==None:
            if len(self.send_queue)>0:
                self.last_unconfirmed = self.send_queue.pop()
            else:
                send_message=False

        if send_message:
            choice_id,value = self.last_unconfirmed
            self._send_observation(choice_id,value)

    def _send_observation(self,choice_id,value):
        """
        Sends the observation to Pike.
        """
        tpn_choice_id = self._tpn_choice_id(choice_id)

        observation = UncontrollableChoiceSensed(id=tpn_choice_id,value=value)
        self.obs_pub.publish(observation)
        if self.verbose>=1:
            rospy.loginfo('Publishing %s = %s'%(tpn_choice_id,value))
        rospy.sleep(1)

    def _observation_echo(self,msg):
        """
        Confirms that the observation message was published
        """
        if msg.id == self._tpn_choice_id(self.last_unconfirmed[0]):
            self.last_unconfirmed = None #Message was confirmed
            if self.verbose>=1:
                rospy.loginfo('Observation %s = %s confirmed!'%(msg.id,msg.value))

    def _tpn_choice_id(self,choice_id):
        """
        TPN's have an additional 'C' in choice ID's
        """
        return choice_id+'C'


if __name__=='__main__':

    if len(sys.argv) != 2:
        print('Please provide a pickle file where the RMPyL program is stored.')
        sys.exit(0)

    with open(sys.argv[1],'rb') as f:
        prog = pickle.load(f)

    rospy.init_node('spoof_sensors')
    pike_spoof = PikeSensorSpoofer(prog)

    while not rospy.is_shutdown():
        user_input = raw_input('Input choice ID (e.g., Choice_0x7f8c8aa22a50): ')
        if user_input.lower() in ['quit','exit']:
            break
        else:
            pike_spoof.spoof(user_input)
