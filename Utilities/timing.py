import time


def timer_function(function, unpack=False):
    """
    Times the execution of a function and returns the elapsed time with results.

    Args:
        function: A callable function with no arguments to be timed
        unpack: If True, unpacks function results; if False, returns as single value

    Returns:
        tuple: (time_taken, *results) if unpack=True, (time_taken, results) if unpack=False
    """
    start_time = time.time()
    results = function()
    time_taken = time.time() - start_time

    if unpack:
        return time_taken, *results
    else:
        return time_taken, results