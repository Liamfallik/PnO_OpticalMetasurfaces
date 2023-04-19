import time

# Define a TimerError class. The (Exception) notation means that TimerError inherits
# from another class called Exception. Python uses this built-in class for error handling.
# You don’t need to add any attributes or methods to TimerError, but having a custom error
# will give you more flexibility to handle problems inside Timer.
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    # When you first create or instantiate an object from a class, your code calls the special method .__init__().
    # In this first version of Timer, you only initialize the ._start_time attribute, which you’ll use to track the
    # state of your Python timer. It has the value None when the timer isn’t running. Once the timer is running,
    # ._start_time keeps track of when the timer started.
    def __init__(self):
        self._start_time = None

    # When you call .start() to start a new Python timer, you first check that the timer isn’t already running.
    # Then you store the current value of perf_counter() in ._start_time.
    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    # when you call .stop(), you first check that the Python timer is running. If it is, then you calculate
    # the elapsed time as the difference between the current value of perf_counter() and the one that you stored
    # in ._start_time. Finally, you reset ._start_time so that the timer can be restarted, and print the elapsed time.
    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")