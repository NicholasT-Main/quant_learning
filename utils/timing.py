import time

def timer_function(function):
    start_time = time.time()
    results=function()
    time_taken=time.time()-start_time
    return time_taken,results