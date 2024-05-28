import threading


def function_with_timeout(func, args=(), kwargs=None, timeout=30):
    """Run a function with a timeout.

    Args:
    - func (callable): The function to run.
    - args (tuple, optional): Positional arguments to pass to the function. Default is ().
    - kwargs (dict, optional): Keyword arguments to pass to the function. Default is {}.
    - timeout (int, optional): Time (in seconds) before the function is interrupted. Default is 5 seconds.

    Returns:
    - Any: The return value of the function if it completes within the timeout. Otherwise, raises an exception.
    """

    if kwargs is None:
        kwargs = {}
    result = [None]
    exception = [None]

    def worker():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        # thread.join()  # you can still allow thread to complete even if it exceeded the timeout
        raise TimeoutError(f"Function {func.__name__} exceeded {timeout} second timeout.")
    if exception[0] is not None:
        raise exception[0]
    return result[0]
