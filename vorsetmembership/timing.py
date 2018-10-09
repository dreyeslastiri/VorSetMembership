# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:58:56 2015

@author: PaulMcGuire (original), Nicojo (update to Python 3)
http://stackoverflow.com/questions/1557571/
how-to-get-time-of-a-python-program-execution/1557906#1557906
& Daniel Reyes Lastiri
"""
# My version
# On Windows use time.clock.
# On linux use time.time
# Use of clock and time is deprecated
# http://stackoverflow.com/questions/85451/python-time-clock-vs-time-time-accuracy

from time import clock, perf_counter
from datetime import timedelta

def secondsToStr(t):
    return str(timedelta(seconds=t))

line = "="*40
def log(string, timeRead, elapsed=None):
    print(line)
    print( string, '-', secondsToStr(timeRead) )
    if elapsed:
        print( "Elapsed time:", secondsToStr(elapsed) )
    print(line)

def startlog(startTime):
    log("Start time",startTime)   
    
def endlog(startTime):
    endTime = perf_counter()
    elapsed = endTime-startTime
    log("End time", endTime, elapsed)
    return elapsed

def now():
    return secondsToStr(perf_counter())

def start():
    self=perf_counter()
    return self

def end(start_time, print_elapsed=False):
    end_time = perf_counter()
    elapsed = end_time-start_time
    if print_elapsed:
        print('t_elapsed = ', secondsToStr(elapsed))
    return elapsed
    

def time_it(func):
    def wrapper(*args,**kwargs):
        tstart = perf_counter()
        result = func(*args,**kwargs)
        tend = perf_counter()
        print(func.__name__, ' t_elapsed = ', secondsToStr(tend-tstart))
        return result
    return wrapper

# Test for debugging
if __name__ == '__main__':
    startTime=start()
    startlog(startTime)
    endlog(startTime)
    x=secondsToStr(126)
    print(x)

# Original version
'''
import atexit
from time import clock
from datetime import timedelta

def secondsToStr(t):
    return str(timedelta(seconds=t))

line = "="*40
def log(s, elapsed=None):
    print(line)
    print(secondsToStr(clock()), '-', s)
    if elapsed:
        print("Elapsed time:", elapsed)
    print(line)
    print()

def endlog():
    end = clock()
    elapsed = end-start
    log("End Program", secondsToStr(elapsed))

def now():
    return secondsToStr(clock())

start = clock()
atexit.register(endlog,args=() )
log("Start Program")
'''