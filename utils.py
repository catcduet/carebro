from __future__ import print_function
import time


class Timer:

    def __init__(self):
        self.start_time = time.time()
        self.job = None

    def start(self, job):
        if job is None:
            return None
        self.start_time = time.time()
        self.job = job
        print("\033[92m[INFO] {job} started.\033[0m".format(job=self.job))

    def stop(self):
        if self.job is None:
            return None
        elapsed_time = time.time() - self.start_time
        print("\033[92m[INFO] {job} finished in {elapsed_time:0.3f} s.\033[0m"
              .format(job=self.job, elapsed_time=elapsed_time))
        self.job = None


if __name__ == "__main__":
    timer = Timer()
    timer.start("Test")
    timer.stop()
