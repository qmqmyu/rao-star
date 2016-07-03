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
Executes Pike as a subprocess and spoofs the interaction with the user, as well
as the uncontrollable choices.
"""
import rospy
from spoofing_sensors import PikeSensorSpoofer
import sys
import os
import subprocess
import pickle
import time
import re
from threading  import Thread
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty  # python 3.x

def enqueue_output(out, queue):
    """
    Constantly reads from the output stream and enqueues it.
    """
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

if __name__=='__main__':

    if len(sys.argv) != 3 or not (os.path.isfile(sys.argv[1]) and os.path.isfile(sys.argv[2])):
        print('Please provide a TPN file and a pickle file with the RMPyL program as two arguments')
        sys.exit(0)

    with open(sys.argv[2],'rb') as f:
        prog = pickle.load(f)

    rospy.init_node('pike_spoofed')
    pike_spoof = PikeSensorSpoofer(prog)

    pike_cmd='rosrun pike pike %s'%(sys.argv[1])
    pike_proc = subprocess.Popen(args=pike_cmd,bufsize=-1,shell=True,stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,stderr=subprocess.PIPE)

    #Thread that continuously reads output from Pike and enqueues it
    pike_output_q = Queue()
    t = Thread(target=enqueue_output, args=(pike_proc.stdout, pike_output_q))
    t.daemon = True # thread dies with the program
    t.start()

    #Uses regular expression to remove the terminal color characters
    remove_terminal_colors = re.compile(r'\x1b[^m]*m')

    terminate=False
    while not rospy.is_shutdown() and not terminate:
        try:
            line = pike_output_q.get_nowait()

            if line:
                sys.stdout.write(line) #Echoes Pike's output
                sys.stdout.flush()
                line = remove_terminal_colors.sub('',line)
                if line.find('[ENTER]')>0: #Starts execution
                    rospy.loginfo('Sending [ENTER] to Pike')
                    time.sleep(0.5)
                    pike_proc.stdin.write('\n\n')
                    pike_proc.stdin.flush()
                elif line.find('Executing event')>0:
                    start = line.find('Executing event ')+len('Executing event ')
                    end = line.find(' ',start)
                    event_id=line[start:end].split()[0].strip(' \n\r\t\x1bq')
                    pike_spoof.spoof(event_id)
                elif line.find('Execution complete')>0: #Execution complete
                    terminate=True
                else:
                    pass
        except Empty:
            pass
        time.sleep(0.5)

    pike_proc.terminate() #Kills the Pike process
    time.sleep(1.0)
