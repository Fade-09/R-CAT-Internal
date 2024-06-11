import time

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return result
    return wrapper

def example_function():
    time.sleep(2)
    print("Function executed.")

example_function = time_it(example_function)

example_function()
