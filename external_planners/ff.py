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
Python interface for the FF planner written in C.

@author: Pedro Santana (psantana@mit.edu).
"""
import ctypes
import ipdb

class PyFF(object):
    """
    Python interface to the FF planner.
    """
    def __init__(self,ff_lib='/usr/local/lib/libff.so',max_plan_str_length=8192):
        self.cff = ctypes.cdll.LoadLibrary(ff_lib)
        self.max_plan_str_length=max_plan_str_length

    def plan(self,domain_file,problem_file):
        self.cff = ctypes.cdll.LoadLibrary('/usr/local/lib/libff.so')
        cmd_tokens = ("ff -o %s -f %s"%(domain_file,problem_file)).split()
        arr = (ctypes.c_char_p * len(cmd_tokens))()
        arr[:]=cmd_tokens
        plan_str = ctypes.create_string_buffer(self.max_plan_str_length)
        #ipdb.set_trace()
        if self.cff.ff_plan(ctypes.pointer(plan_str), len(cmd_tokens), arr):
            print('Found plan')
            return plan_str.value.split('\n')
        else:
            print('Failed to find plan.')
            return []
