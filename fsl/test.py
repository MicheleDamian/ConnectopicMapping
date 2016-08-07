from multiprocessing import Pool
import time
import numpy

if __name__ == '__main__':

    def f(x):
        print('Worker {0}...'.format(x), end="", flush=True)
        t = numpy.random.randint(0, 10)
        time.sleep(t)
        print('\rWorker {0}... {0}'.format(x), flush=True)

    with Pool(10) as p:
        p.map(f, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
