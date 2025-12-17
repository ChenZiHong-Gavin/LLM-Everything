# multiprocessing模块

### 1 Python为什么慢

我们经常听程序员争论”XXX才是世界上最好的语言！”有一个理由常常让Python的拥趸者哑口无言“Python太慢了！”

以下是 The Benchmarks Game 面向主流编程语言设计的性能测试榜单，灰色反映的是时间效率，越短代表性能越好，棕色则是基于执行时间和内存开销的加权值。

<figure><img src="../../.gitbook/assets/image (12) (1).png" alt=""><figcaption></figcaption></figure>

Pthon3处于垫底水平，那Python为什么会慢呢？

1. 开发者写的代码不够pythonic

*   对象复制

    下面两种对象复制方法有数倍的速度差异。

    ```python
    from random import randint
    from time import time

    data = [str(randint(1000000, 9999999)) for _ in range(10000000)]

    s = time()
    all_add = ''
    for x in data:
        all_add += x
    e = time()
    print(e-s) # 1.4891037940979004

    s = time()
    all_join = ''.join(data)
    e = time()
    print(e-s) # 0.1853940486907959
    ```
*   迭代器

    用好迭代器能加速部分场景。

    迭代器有**惰性求值**的特性，并非一次性将所有元素加载到内存中，而是在每次需要时才计算并返回下一个元素。

    运行下面的程序，会发现get\_data\_1与get\_data\_2所用的时间类似，但是get\_data\_1的三个url几乎是同时输出，而get\_data\_2的url是一秒一个。

    ```python
    from time import time, sleep

    urls = [
        "url1",
        "url2",
        "url3",
    ]

    def get_data_1(urls):
        results = []
        for url in urls:
            sleep(1)
            results.append("data for " + url)
        return results

    def get_data_2(urls):
        for url in urls:
            sleep(1)
            yield "data for " + url

    s = time()
    for data in get_data_1(urls):
        print(data)
    e = time()
    print(e-s) # 3.0031187534332275

    s = time()
    for data in get_data_2(urls):
        print(data)
    e = time()
    print(e-s) # 3.003131628036499
    ```

    把代码修改成这样应该很容易看出来了：

    ```python
    from time import time, sleep

    urls = [
        "url1",
        "url2",
        "url3",
    ]

    def get_data_1(urls):
        results = []
        for url in urls:
            sleep(1)
            results.append("data for " + url)
        return results

    def get_data_2(urls):
        for url in urls:
            sleep(1)
            yield "data for " + url

    s = time()
    for i, data in enumerate(get_data_1(urls)):
        if i == 0:
            break
        print(i, data)
    e = time()
    print(e-s) # 3.0031187534332275

    s = time()
    for i, data in enumerate(get_data_2(urls)):
        if i == 0:
            break
        print(i, data)
    e = time()
    print(e-s) # 1.0010180473327637
    ```

    不难看到，迭代器的i=0后面的程序就没有再运行了。

    *   数据结构

        不同的数据结构容易造成较大的性能差异，比如：

        ```python
        from time import time
        data1 = list(range(0, 100000))

        data2 = set(range(0, 100000))

        s1 = time()
        99999 in data1
        e1 = time()
        print(e1-s1) # 0.0006175041198730469

        s2 = time()
        99999 in data2 # 1.1920928955078125e-06
        e2 = time()
        print(e2-s2)
        ```

        set是用hash table实现的，hash表查询速度更快
* GIL (Global Interpret Lock)

<figure><img src="../../.gitbook/assets/image (13).png" alt=""><figcaption></figcaption></figure>

全局解释器锁，确保在**任何一个时刻，同一个Python进程中只有一个线程能够执行Python字节码。因此如果两个线程并发的话还是要做序列化，所以我们才会说Python中无法实现真正的多线程。**

下面是一个例子：

```python
from threading import Thread
from time import time

def empty_loop():
    for _ in range(100000000):
        pass # 耗时程序

def test(thread_count=1, func=empty_loop):
    thread_list = [Thread(target=func) for _ in range(thread_count)]

    s = time()
    for task in thread_list:
        task.start()
    for task in thread_list:
        task.join()
    e = time()

    print('{} threads, {} seconds'.format(thread_count, int(e-s)))

if __name__ == "__main__":
    test(1) # 1 thread, 1 seconds
    test(2) # 2 threads, 2 seconds
    test(3) # 3 threads, 3 seconds
    
```

loop是占用了CPU，而同一进程内只能有一个线程执行，所以会慢，这个代码只是虚有多线程的壳子，没有起到多线程的作用。

## 2 Python并发解决方案

当然，对于IO密集型的任务（比如文件读写），上面的多线程代码是成立的，因为计算机在等待IO的过程中不占用CPU，可以切换下一个线程。针对IO密集型的任务可以使用各类异步编程库：asyncio, aiohttp, eventlet, twisted等，可以后续文章再细讲。

针对上述CPU密集型的任务（比如计算正则表达式），可以使用多进程。

将上述代码改成多进程，就可以顺利执行：

```python
from multiprocessing import Process
from time import time

def empty_loop():
    for _ in range(100000000):
        pass # 耗时程序

def test(thread_count=1, func=empty_loop):
    thread_list = [Process(target=func) for _ in range(thread_count)]

    s = time()
    for task in thread_list:
        task.start()
    for task in thread_list:
        task.join()
    e = time()

    print('{} process, {} seconds'.format(thread_count, int(e-s)))

if __name__ == "__main__":
    test(1) # 1 process, 1 seconds
    test(2) # 2 processes, 1 seconds
    test(3) # 3 processed, 1 seconds
```

### 2.1 并行与并发的区别

并发（Concurrency）：看起来在同时处理任务，实际可能是使用单个处理器通过快速切换交替执行，给人同时执行的错觉。

并行（Parallelism）：在多核CPU或者多设备上真正的同时执行。

### 2.2 multiprocessing模块

Process类

创建进程的类，由该类实例化得到的对象，表示一个子进程中的任务。

p=Process()

* p.start() 启动进程
* p.run() 进程启动时运行的方法，用于调用target来指定需要执行的函数；这个方法通常不应该由用户直接调用
* p.terminate() 强制终止进程p，但不会做任何的清理操作
* p.is\_alive() 判断p是否还在运行
* p.join(\[timeout]) 主线程等待p结束，确保主进程在子进程完成之后再继续执行

```python
import multiprocessing
import time

def worker(name):
    print(f"进程 {name} 开始工作...")
    time.sleep(3)
    print(f"进程 {name} 完成工作。")

if __name__ == '__main__':
    p = multiprocessing.Process(target=worker, args=('A',))
    
    print("主进程：准备启动子进程。")
    p.start()
   
    print(f"主进程：子进程已启动，状态 is_alive: {p.is_alive()}")
    print("主进程：等待子进程结束...")
    p.join() 
    print(f"主进程：子进程已结束，状态 is_alive: {p.is_alive()}")
    print("主进程：所有工作完成。")
```

Lock

多进程的本质是减少并发过程中锁的使用，但有些时候为了避免资源错乱，不得不用锁。

不加锁的后果：

```python
import multiprocessing

def worker_without_lock(shared_value):
    for _ in range(100000):
        current_value = shared_value.value
        shared_value.value = current_value + 1 # 对共享值进行累加

if __name__ == "__main__":
    shared_number = multiprocessing.Value('i', 0) 

    processes = []
    for _ in range(10):
        p = multiprocessing.Process(target=worker_without_lock, args=(shared_number,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # 预期结果：1000000
    print(f"实际结果: {shared_number.value}") # 172275，或者任意小于1000000的数字
```

加锁之后：

```python
import multiprocessing

def worker_without_lock(shared_value):
    for _ in range(100000):
        # 加锁
        with shared_value.get_lock():
            current_value = shared_value.value
            shared_value.value = current_value + 1 # 对共享值进行累加

if __name__ == "__main__":
    shared_number = multiprocessing.Value('i', 0) 

    processes = []
    for _ in range(10):
        p = multiprocessing.Process(target=worker_without_lock, args=(shared_number,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # 预期结果：1000000
    print(f"实际结果: {shared_number.value}") # 1000000
```

如果一个进程已经持有了某个 `Lock`，在再次尝试获取这个 `Lock` 时，会造成自**死锁**。

<figure><img src="../../.gitbook/assets/image (14).png" alt=""><figcaption></figcaption></figure>

相比于普通的Lock，Rlock允许同一个进程多次获取锁：

`RLock` 内部维护着一个**持有者标识**和一个**递归计数器**

1. 当一个进程第一次获取 `RLock` 时，`RLock` 会记录标识，并将计数器设为 1。
2. 当**同一个进程**再次获取这个 `RLock` 时，它会检查到持有者就是自己，于是不会阻塞，而是将计数器加 1。
3. 每次调用 `release()` ，计数器会减 1。
4. 当计数器减到 0 时，锁才会被真正释放，供其它进程获取。

Pool

手动创建进程需要一个一个start并join，略为繁琐，使用进程池可以简化这一过程。

```python
import multiprocessing
import time

def square(x):
    result = x * x
    time.sleep(1)
    return result

if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5, 6, 7, 8]
    start_time = time.time()

    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(square, numbers)
   
    end_time = time.time()
```

一个经典的并发过程如下图所示：

<figure><img src="../../.gitbook/assets/image (15).png" alt=""><figcaption></figcaption></figure>

## 4 条件变量与生产者消费者

上面的过程实际就是生产者与消费者的关系。

现在有两个队列，生产者队列负责添加数据，消费者队列负责取出数据。

在这个过程中，有一个固定大小的缓冲区：

* 当缓冲区已满时，生产者不应再向其中添加数据，否则会覆盖尚未被消费的数据。
* 当缓冲区为空时，消费者不应尝试从中取出数据，因为无数据可取。

如果有一些特殊需求需要满足：

1. 如果消费者要求必须含有至少3个数据才开始消费，应该怎么办
2. 同理生产者呢？
3. 如果每个消费者只能消费特定的数据，应该怎么办？

多进程中可以通过一个共享变量信号量解决：

```python
import multiprocessing
import time

BUFFER_SIZE = 5

def producer(buffer, empty_slots, filled_slots, mutex):
    """生产者进程"""
    for i in range(10):
        # P(empty_slots): 等待一个空位。如果empty_slots为0，则阻塞。
        empty_slots.acquire()
        
        # P(mutex): 获取互斥锁，保护缓冲区。
        mutex.acquire()
        
        # --- 临界区 ---
        # 使用共享内存数组作为循环队列
        buffer[i % BUFFER_SIZE] = i 
        print(f"生产了 -> {i}")
        # --- 临界区结束 ---
        
        # V(mutex): 释放互斥锁。
        mutex.release()
        
        # V(filled_slots): 通知消费者，物品数+1。
        filled_slots.release()
        
        time.sleep(0.1)

def consumer(buffer, empty_slots, filled_slots, mutex):
    """消费者进程"""
    for i in range(10):
        # P(filled_slots): 等待一个物品。如果filled_slots为0，则阻塞。
        filled_slots.acquire()
        
        # P(mutex): 获取互斥锁。
        mutex.acquire()
        
        # --- 临界区 ---
        item = buffer[i % BUFFER_SIZE]
        print(f"    消费了 <- {item}")
        # --- 临界区结束 ---
        
        # V(mutex): 释放互斥锁。
        mutex.release()
        
        # V(empty_slots): 通知生产者，空位数+1。
        empty_slots.release()
        
        time.sleep(0.4)

if __name__ == '__main__':
    # 创建共享内存
    shared_buffer = multiprocessing.Array('i', BUFFER_SIZE)
    
    # 1. 计数信号量，代表空槽位
    empty_slots = multiprocessing.Semaphore(BUFFER_SIZE)
    
    # 2. 计数信号量，代表已填充的槽位
    filled_slots = multiprocessing.Semaphore(0)
    
    # 3. 二进制信号量 (用Lock实现)，用于互斥访问缓冲区
    mutex = multiprocessing.Lock()
    
    # 创建并运行进程
    p = multiprocessing.Process(target=producer, args=(shared_buffer, empty_slots, filled_slots, mutex))
    c = multiprocessing.Process(target=consumer, args=(shared_buffer, empty_slots, filled_slots, mutex))
    
    p.start()
    c.start()
    
    p.join()
    c.join()
```

### 参考

1. [干货：深入浅出讲解Python并发编程（一）-阿里云开发者社区](https://developer.aliyun.com/article/941259)
2. [深入Python与日志服务，玩转大规模数据分析处理实战系列直播第四讲-云视频-阿里云开发者社区](https://developer.aliyun.com/live/932)
3. [各大主流编程语言性能PK，结果出乎意料-CSDN博客](https://blog.csdn.net/hollis_chuang/article/details/125592004)
