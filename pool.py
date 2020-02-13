import time
import multiprocessing


def basic_func(x):
    if x == 0:
        return 'zero'
    elif x % 2 == 0:
        return 'even'
    else:
        return 'odd'


def multiprocessing_func():
    y = i * i
    time.sleep(2)
    print('{} squared results in a/an {} number'.format(i, basic_func(y)))


if __name__ == '__main__':
    starttime = time.time()
    processes = []
    for i in range(0, 10):
        p = multiprocessing.Process(target=multiprocessing_func)
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print('That took {} seconds'.format(time.time() - starttime))