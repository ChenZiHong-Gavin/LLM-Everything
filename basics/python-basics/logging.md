# Logging模块

### **1 从结绳记事到日志**

在远古时代，人类使用**结绳记事**记录重要事件——绳结的大小、位置和颜色代表不同含义。随着文明发展，我们有了纸笔、印刷术，直到今天的电子日志系统。

Python的`logging`模块就是数字时代的"结绳记事"，帮助开发者记录程序运行的关键信息。

### 2 为什么需要日志

日志和保险一样，系统没有问题时没有存在感，出了问题后承担“行车记录仪”的功能。

日志记录应用程序和基础设施运行中的**每一个关键事件、操作和消息**，包括所发生事件的描述、相关时间戳、与事件相关的严重性级别类型以及其他相关元数据。日志对于调试问题、诊断错误和审核系统活动非常有用。它们提供了系统事件的文本叙述，使人们更容易理解所采取的操作顺序。

### 3 Python日志模块概述

<figure><img src="../../.gitbook/assets/image (5).png" alt=""><figcaption></figcaption></figure>

Python 的 **`logging`** 模块通过**层级化的日志器（Logger）**、**处理器（Handler）**、**过滤器（Filter）** 和 **格式化器（Formatter）** 实现灵活的日志管理。

*   Logger

    日志级别：**`DEBUG`** < **`INFO`** < **`WARNING`** < **`ERROR`** < **`CRITICAL`**，日志仅当级别≥日志器级别时才会处理。
* Filter
  * **精细化控制：**&#x57FA;于日志内容（如级别、消息文本、模块名）决定是否处理日志。
  * 绑定位置：可附加到日志器（全局过滤）或处理器（目标特定过滤）。
*   Handler

    负责将日志传递到不同目标：

    * **控制台处理器**：**`logging.StreamHandler`**（输出到 **`sys.stdout/stderr`**）。
    * **文件处理器**：**`logging.FileHandler`**（写入文件）、**`RotatingFileHandler`**（滚动文件）。
    * **其他处理器**：如 **`SMTPHandler`**（邮件）、**`SocketHandler`**（网络）等。
    * **多处理器支持**：一个日志器可绑定多个处理器（如同时输出到控制台和文件）。
*   Formatter

    * 将日志记录（**`LogRecord`**）转换为字符串（如添加时间、模块名）

    <figure><img src="../../.gitbook/assets/image (6).png" alt=""><figcaption></figcaption></figure>

### 4 日志使用常见错误

#### 4.1 没有log或者少打log

解决方案：用装饰器简化代码。

例如我们有三个函数， `add` 、 `mul` 、 `div`

如果我们需要记录函数的入口参数、返回结果和异常信息，以往需要在每个函数中重复编写日志代码，显得冗余且费力。

以下是优化后的实现：

```python
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from functools import wraps

def log_enter_exit(fn):
    @wraps(fn)
    def __wrapped(*args, **kwargs):
        logger.debug("enter %s: %s %s", fn.__name__, args, kwargs)
        try:
            ret = fn(*args, **kwargs)
            logger.debug("exit result: %s", ret)
            return ret
        except Exception as ex:
            logger.debug("exit got error: %s", ex, exc_info=True)
            raise ex
    return __wrapped

@log_enter_exit
def add(v1, v2):
    return v1 + v2

@log_enter_exit
def mul(v1, v2):
    return v1 * v2

@log_enter_exit
def div(v1, v2):
    return v1 / v2

add(1, 2)
mul(3, 4)
div(10, 0)
```

* 通过装饰器封装日志记录代码，消除每个函数中的重复日志语句
* 自动记录函数名、参数、返回值和异常信息

输出示例：

```python
2023-10-05 12:00:00 - __main__ - DEBUG - Enter add: args=(1, 2), kwargs={}
2023-10-05 12:00:00 - __main__ - DEBUG - Exit add: Result=3
2023-10-05 12:00:00 - __main__ - DEBUG - Enter mul: args=(3, 4), kwargs={}
2023-10-05 12:00:00 - __main__ - DEBUG - Exit mul: Result=12
2023-10-05 12:00:00 - __main__ - DEBUG - Enter div: args=(10, 0), kwargs={}
2023-10-05 12:00:00 - __main__ - DEBUG - Exit div with error: division by zero
Traceback (most recent call last):
  File "...", line 15, in wrapper
    ret = fn(*args, **kwargs)
  File "...", line 35, in div
    return v1 / v2
ZeroDivisionError: division by zero
```

#### 4.2 log信息不全

1.  缺少上下文信息

    解决方案：使用LogAdapter简化操作

    有时打印log时容易信息不全，容易混淆两个类的信息，例如：

    ```python
    import logging

    class A:
        def __init__(self, name):
            self.name = name

        def eat(self, food):
            logging.info(f"Eating {food}")

    class B:
        def __init__(self, name):
            self.name = name

        def eat(self, food):
            logging.info(f"Eating {food}")

    a = A("name1")
    b = B("name2")
    a.eat("apple")
    b.eat("apple")
    ```

    log后的信息都是 `Eating apple` ，同时，如果类的属性有删减，也需要一个一个修改logging的代码，非常麻烦。

    这时可以使用log的extra来简化操作，代码如下：

    ```python

    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(my_name)s - %(my_class)s - %(message)s')

    class A:
        def __init__(self, name):
            self.name = name
            self.extra = {
                'my_class': self.__class__.__name__,
                'my_name': self.name
            }

        def eat(self, food):
            logging.info(f"Eating {food}", extra=self.extra)

    a = A("name1")
    a.eat("apple")
    ```

    extra的内容也可以在logging模块中全局配置：

    ```python
    import logging
    import sys

    # 引用extra
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(ip)s - %(username)s - %(message)s")

    h_console = logging.StreamHandler(sys.stdout)
    h_console.setFormatter(fmt)
    init_logger = logging.getLogger("myPro")
    init_logger.setLevel(logging.DEBUG)
    init_logger.addHandler(h_console)

    extra_dict = {"ip": "1.2.3.4", "username": "Jack"}

    logger = logging.LoggerAdapter(init_logger, extra_dict)

    # 不需要写代码即可自动输出extra
    logger.info("test1!")

    # 也可以覆盖
    logger.info("test2!", extra={"ip": "113.208.78.29", "username": "Petter"})

    # 直接修改context
    logger.extra = {"ip": "113.208.78.29", "username": "Petter"}
    logger.info("test3")

    ```

    输出示例：

    ```python
    2025-06-14 17:33:04,232 - myPro - 1.2.3.4 - Jack - test1!
    2025-06-14 17:33:04,232 - myPro - 1.2.3.4 - Jack - test2!
    2025-06-14 17:33:04,233 - myPro - 113.208.78.29 - Petter - test3
    ```
2.  报错缺少堆栈信息

    解决方案：设置 `exc_info=True` ，代码示例如下：

    ```python
    import logging

    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s\\n%(exc_info)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()

    def risky_operation():
        return 1 / 0

    try:
        risky_operation()
    except Exception as ex:
        logger.error(f"操作失败: {ex}", exc_info=True)

    ```
3.  需要临时更换context

    解决方案：实现LoggingContext类，临时改变log的行为（monkey patch）

    ```python
    import logging

    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    class LoggingContext(object):
        def __init__(self, logger, level=None, handler=None, close=True):
            self.logger = logger
            self.level = level
            self.handler = handler
            self.close = close

        def __enter__(self):
            if self.level is not None:
                self.old_level = self.logger.level
                self.logger.setLevel(self.level)
            if self.handler:
                self.logger.addHandler(self.handler)

        def __exit__(self, et, ev, tb):
            if self.level is not None:
                self.logger.setLevel(self.old_level)
            if self.handler:
                self.logger.removeHandler(self.handler)
            if self.handler and self.close:
                self.handler.close()

    # info级别不打印
    def process_data1(s1):
        logger.info("entering process_data1")
        for x in s1:
            logger.info("processing1: %s", x)

    # 重要INFO打印
    def process_data2(s1):
        logger.info("entering process_data2")
        for x in s1:
            with LoggingContext(logger, level=logging.INFO):
                logger.info("processing2: %s", x)

    process_data1("abc")
    process_data2("abc")

    ```

#### 4.3 level不合适导致磁盘占满

解决方案：可以使用 `RotatingHandler` 或者 `TimeRotatingHandler`

#### 4.4 logging浪费过多CPU资源

解决方案：注意log的风格

Python logging 的[官方文档](https://docs.python.org/3/howto/logging.html#optimization)明确建议使用占位符格式：

> "将变量作为参数传递给日志记录方法，而非自行格式化字符串。"

* 当使用 **`%`** 或 **`{}`** 占位符 + 参数传递时，**仅在日志实际需要输出时**（如当前日志级别≥设置的级别）才会进行字符串格式化。
* 而 **`.format()`** 或 f-string 会**立即执行格式化**，即使日志被过滤（例如当前级别为 **`ERROR`**，但调用的是 **`INFO`** 日志）。
* **节省资源**：在高并发或频繁记录低级别日志（如 **`DEBUG`**）时，避免不必要的字符串操作。

下面是代码示例：

```python
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 两种风格
logger.info("processing: {0}".format(100))
logger.info("processing: %s", 100)  # 建议，当格式化字符串比较复杂时
```

### 5 总结

从远古的绳结到现代的日志系统，记录的核心目的从未改变——**保存重要信息，传递关键知识**。Python的`logging`模块提供了强大而灵活的日志解决方案，是每个Python开发者必须掌握的核心技能之一。

### 参考

1. [10个日志记录最佳实践 - 极道](https://www.jdon.com/76292.html)
2. [https://developer.aliyun.com/live/949](https://developer.aliyun.com/live/949)
3. [https://github.com/apache/incubator-hugegraph-ai/blob/main/hugegraph-python-client/src/pyhugegraph/utils/log.py](https://github.com/apache/incubator-hugegraph-ai/blob/main/hugegraph-python-client/src/pyhugegraph/utils/log.py)
4. [https://zhuanlan.zhihu.com/p/1916951996038099482](https://zhuanlan.zhihu.com/p/1916951996038099482)
