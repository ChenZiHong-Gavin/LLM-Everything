# Import模块

### 1 import原理

<figure><img src="../../.gitbook/assets/image (2) (1).png" alt=""><figcaption></figcaption></figure>

导入一个模块时，Python 首先会检查 `sys.modules` 这个字典。如果模块已经存在于 `sys.modules` 中（即已经被导入过），Python 会直接返回该模块，避免重复加载，确保了模块的单例性。

在 `import` 一个模块前后，观察 `sys.modules.keys()` 的变化，可以看到新导入的模块名被添加了进去。如下：

```python
import sys
list(sys.modules.keys())[:10]

'numpy' in sys.modules.keys() # False

import numpy
list(sys.modules.keys())[:10]

'numpy' in sys.modules.keys() # True
```

`import`语句不仅仅加载模块，还负责执行模块中所有的顶级语句。这意味着，模块中的任何初始化代码都将被执行。

### 2 三种import方法

* 方法一（**`import module_name`** ）：最常见的导入方式。
* 方法二（**`__import__('module_name')`**）：这是一个内置函数，底层由 Python 解释器调用，用于实现 `import` 语句的功能。
* 方法三（**`importlib.import_module('module_name')`**）**：** `importlib` 模块提供的一个函数，推荐在需要动态导入模块时使用。

### 3 如何让外部模块在不修改代码的情况下被 `import`

有时我们需要导入位于非标准路径的模块，或者替换系统自带的模块。在不直接修改代码中的 `import` 语句的情况下，有以下几种常用方法：

1.  修改 `sys.path`

    将模块所在的目录路径添加到 `sys.path` 中。由于 `sys.path` 是一个列表，我们可以使用 `sys.path.insert(0, 'your/module/path')` 将路径添加到列表的开头，确保你的模块优先被搜索到。
2.  通过 `PYTHONPATH` 环境变量 `PYTHONPATH` 是一个环境变量，可以用来指定额外的模块搜索路径。在启动 Python 解释器之前，设置 `PYTHONPATH` 变量，将模块所在的目录添加到其中，Python 解释器会自动将其内容添加到 `sys.path` 中。

    例如：

    ```python
    export PYTHONPATH="./your_lib_folder"; python -c "import your_lib"
    ```
3.  使用 `.pth` 文件

    在 `site-packages` 目录下创建以 `.pth` 为后缀的文件。这些文件中可以包含额外的模块搜索路径，每行一个路径。当 Python 解释器启动时，它会自动读取这些 `.pth` 文件，并将其中的路径添加到 `sys.path` 中。 `.pth`文件不仅可以包含路径，还可以包含Python代码。这对于更高级的配置非常有用，比如条件性地添加路径或者执行一些初始化代码。

    例如：

    ```python
    import sys
    if sys.platform == 'win32':
        sys.path.append('C:/Windows/System32')
    ```

### 4 `import` Hook

`import` Hook 是一种更高级的技巧，它允许我们干预 Python 的导入过程。通过实现 `import` Hook，我们可以在模块被查找、加载和执行的各个阶段插入自定义逻辑。

这使得我们可以实现一些自定义功能，例如：

#### 4.1 **检测 `import` 了哪些模块**

可以用来监控程序的依赖关系。

```python
impl = __import__
def my_importer(*args, **kwargs):
    print("** importing: " + args[0])
    return impl(*args, **kwargs)
__builtins__.__import__ = my_importer
```

#### 4.2 **模块替换（Monkey Patching）**

在不修改原始代码的情况下，用自定义的实现替换系统库或第三方库的某个部分。这在测试、调试或临时修复 Bug 时非常有用。

例如，替换 `urllib2.urlopen` 的行为，使其在本地测试时指向一个模拟服务。

```python
# 原始的urllib.request模块中的urlopen函数
import urllib.request

# 保存原始函数的引用
original_urlopen = urllib.request.urlopen

# 自定义的替代函数
def mock_urlopen(url, *args, **kwargs):
    print(f"Mock: 正在请求URL: {url}")
    
    # 模拟不同URL的返回内容
    if 'api/users' in str(url):
        import io
        return io.BytesIO(b'{"users": ["测试用户1", "测试用户2"]}')
    else:
        # 对于其他URL，调用原始函数
        return original_urlopen(url, *args, **kwargs)

# 替换原始函数
urllib.request.urlopen = mock_urlopen

# 现在，代码中任何对urllib.request.urlopen的调用都会使用mock_urlopen
# 例如:
response = urllib.request.urlopen('https://example.com/api/users')
print(response.read())
```



### 参考

1. [Python系列直播——深入Python与日志服务，玩转大规模数据分析处理实战系列直播第6六讲-云视频-阿里云开发者社区](https://developer.aliyun.com/live/969)
2. [Python 中的 import 机制 - 知乎](https://zhuanlan.zhihu.com/p/678539151)
3. [Python如何import不同文件夹下的文件(module) - 知乎](https://zhuanlan.zhihu.com/p/451438246)
4. [Python模块导入机制深度解析：importlib的10个关键使用技巧与高级特性 - CSDN文库](https://wenku.csdn.net/column/68tay0gm54)
