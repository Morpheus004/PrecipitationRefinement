import concurrent.futures
import os
import time

def infinite_task(counter):
    print(f"Process {os.getpid()} running task {counter}")
    time.sleep(1)
    return counter

def counter_generator():
    counter = 1
    while True:
        yield counter
        counter += 1

def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        # Use map with an infinite generator
        for result in executor.map(infinite_task, counter_generator()):
            pass  # The loop will run forever

if __name__ == "__main__":
    main()

