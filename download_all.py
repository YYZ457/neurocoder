"""
One-stop data download: more Python repos + Chinese text corpus.
Run: python download_all.py
"""
import os, subprocess, time, urllib.request, json
from pathlib import Path

DATA_DIR = "sample_data/source"
CHINESE_DIR = "sample_data/chinese_text"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHINESE_DIR, exist_ok=True)

def clone(repo, branch="main"):
    name = repo.split("/")[1].replace(".git", "")
    dest = Path(DATA_DIR) / name
    if dest.exists():
        print(f"  [Skip] {name}")
        return 0
    url = f"https://github.com/{repo}.git"
    print(f"  Clone {repo}...", end=" ", flush=True)
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--branch", branch, url, str(dest)],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180, check=True)
        n = len(list(dest.rglob("*.py")))
        print(f"[OK] {n} .py")
        return n
    except:
        print("[FAIL]")
        return 0

def write_file(path, content):
    Path(path).write_text(content.strip() + "\n", encoding="utf-8")
    print(f"  [TXT] {Path(path).name}")


# ====== TIER 1: More quality Python repos ======
PYTHON_REPOS = """
tensorflow/tensorflow
apache/spark
ansible/ansible
home-assistant/core
apache/airflow
localstack/localstack
apache/superset
getsentry/sentry
certbot/certbot
mitmproxy/mitmproxy
httpie/cli
cookiecutter/cookiecutter
streamlit/streamlit
gradio-app/gradio
fabric/fabric
boto/botocore
aws/aws-cli
docker/compose
kubernetes-client/python
saltstack/salt
ansible/ansible-lint
explosion/spaCy
RasaHQ/rasa
allenai/allennlp
deepset-ai/haystack
run-llama/llama_index
langchain-ai/langchain
nvbn/thefuck
spyder-ide/spyder
donnemartin/system-design-primer
donnemartin/data-science-ipython-notebooks
pennersd/django-allauth
pydata/xarray
HIPS/autograd
google/jax
fxsjy/jieba
Rokid/aliyun-iot-device-sdk-python
zhangslob/awesome-python-cn
"""

# ====== TIER 2: Chinese Python tutorials & docs ======
CHINESE_TUTORIALS = {
    "01_basics_cn.py": """
# Python 基础教程 - 中文版
# 变量、数据类型、运算符

# 这是单行注释
'''
这是多行注释
可以写多行文字
'''

# 变量赋值
name = "小明"  # 字符串
age = 18       # 整数
height = 1.75  # 浮点数
is_student = True  # 布尔值

# 字符串操作
greeting = "你好, " + name  # 字符串拼接
print(greeting)

# 字符串格式化
message = f"{name}今年{age}岁,身高{height}米"
print(message)

# 列表操作
fruits = ["苹果", "香蕉", "橘子"]
fruits.append("葡萄")  # 添加元素
fruits.remove("香蕉")  # 删除元素
print(f"水果列表: {fruits}")

# 字典操作
person = {
    "name": "张三",
    "age": 25,
    "city": "上海",
    "hobbies": ["编程", "读书", "跑步"]
}
print(f"{person['name']}住在{person['city']}")
""",

    "02_control_flow_cn.py": """
# Python 控制流 - 中文版
# if/else、for、while

# 条件判断
score = 85
if score >= 90:
    grade = "优秀"
elif score >= 80:
    grade = "良好"
elif score >= 70:
    grade = "中等"
elif score >= 60:
    grade = "及格"
else:
    grade = "不及格"
print(f"成绩: {score}, 等级: {grade}")

# for 循环 - 遍历列表
students = ["张三", "李四", "王五", "赵六"]
for student in students:
    print(f"同学: {student}")

# for 循环 - 使用 range
print("从1数到5:")
for i in range(1, 6):
    print(i, end=" ")
print()

# while 循环
count = 0
while count < 5:
    print(f"第{count + 1}次循环")
    count += 1

# 列表推导式
squares = [x**2 for x in range(1, 11)]
print(f"1到10的平方: {squares}")
""",

    "03_functions_cn.py": """
# Python 函数 - 中文版

def greet(name):
    """打招呼函数。参数 name: 姓名"""
    return f"你好, {name}!"

def add(a, b):
    """两数相加。返回和"""
    return a + b

def factorial(n):
    """计算 n 的阶乘。使用递归实现。"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    """生成斐波那契数列的前 n 项。"""
    result = [0, 1]
    for i in range(2, n):
        result.append(result[i-1] + result[i-2])
    return result

def is_prime(n):
    """判断一个数是否为质数。"""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# 测试函数
print(greet("小明"))
print(f"3 + 5 = {add(3, 5)}")
print(f"5! = {factorial(5)}")
print(f"斐波那契: {fibonacci(10)}")
print(f"质数检查: {[x for x in range(1, 21) if is_prime(x)]}")
""",

    "04_classes_cn.py": """
# Python 面向对象编程 - 中文版

class Animal:
    '''动物基类'''
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name}发出声音"

class Dog(Animal):
    '''狗类，继承自动物类'''
    def __init__(self, name, breed):
        super().__init__(name)  # 调用父类构造函数
        self.breed = breed

    def speak(self):
        return f"{self.name}({self.breed})说: 汪汪!"

class Cat(Animal):
    '''猫类'''
    def speak(self):
        return f"{self.name}说: 喵喵!"

class Calculator:
    '''计算器类 - 演示静态方法和类方法'''
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def multiply(a, b):
        return a * b

    @classmethod
    def create_double(cls, value):
        return cls(value * 2)

# 使用示例
dog = Dog("旺财", "金毛")
cat = Cat("咪咪")
print(dog.speak())
print(cat.speak())
print(f"计算: {Calculator.add(10, 20)}")
""",

    "05_file_io_cn.py": """
# Python 文件操作 - 中文版
import os
import json
import csv

def write_text_file(filepath, content):
    """把文本写入文件"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"文件已保存: {filepath}")

def read_text_file(filepath):
    """读取文本文件内容"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def write_json(filepath, data):
    """把数据保存为JSON文件"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_json(filepath):
    """读取JSON文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def read_csv(filepath):
    """读取CSV文件"""
    data = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

# 测试
test_data = {
    "name": "测试数据",
    "version": 1.0,
    "items": [1, 2, 3, 4, 5]
}
write_json("test_output.json", test_data)
loaded = read_json("test_output.json")
print(f"JSON读写测试: {loaded == test_data}")
""",

    "06_algorithms_cn.py": """
# 常用算法 - 中文版

def bubble_sort(arr):
    """冒泡排序。每次比较相邻的两个元素。"""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def binary_search(arr, target):
    """二分查找。在有序数组中查找目标值。"""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def dfs(graph, start, visited=None):
    """深度优先搜索。"""
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited

def bfs(graph, start):
    """广度优先搜索。"""
    visited = set([start])
    queue = [start]
    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited

# 图结构示例
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}
print(f"DFS: {dfs(graph, 'A')}")
print(f"BFS: {bfs(graph, 'A')}")
""",
}

CHINESE_TEXT_CORPUS = {
    "python_intro.txt": """
Python 是一种广泛使用的高级编程语言。由吉多·范罗苏姆在1989年创造，Python的设计理念强调代码的可读性和简洁的语法。

Python的特点包括：
1. 简洁易读: Python使用缩进来定义代码块,不需要花括号
2. 动态类型: 变量不需要声明类型,可以直接赋值
3. 自动内存管理: Python有垃圾回收机制
4. 丰富的标准库: Python自带"电池",内置大量模块
5. 跨平台: 可以在Windows、Linux、Mac等系统上运行
6. 面向对象: 支持面向对象编程范式
7. 可扩展: 可以用C/C++编写扩展模块

Python广泛应用于:
- Web开发(Django、Flask、FastAPI)
- 数据科学(NumPy、Pandas、Matplotlib)
- 机器学习与深度学习(TensorFlow、PyTorch、Scikit-learn)
- 自动化运维
- 网络爬虫(Scrapy、BeautifulSoup)
- 科学计算
- 人工智能
""",

    "python_advanced.txt": """
Python高级编程技巧

一、装饰器
装饰器是一种设计模式,可以在不修改函数代码的情况下添加功能。
使用@语法糖可以将装饰器应用到函数上。

二、生成器
生成器使用yield关键字,可以逐个产生值而不是一次性生成所有值。
这在处理大量数据时非常有用,可以节省内存。

三、上下文管理器
使用with语句和上下文管理器可以自动管理资源,
最常见的例子是文件操作: with open() as f:

四、多线程与多进程
Python的threading模块用于多线程编程,
multiprocessing模块用于多进程编程。
由于GIL(全局解释器锁)的限制,CPU密集型任务应该使用多进程。

五、异步编程
asyncio模块提供了异步I/O的支持,
使用async/await关键字可以编写高效的并发代码。
特别适合网络请求、数据库操作等I/O密集型任务。

六、类型提示
Python 3.5+支持类型提示(typing),
可以提高代码的可读性和可维护性。
""",

    "deep_learning_intro.txt": """
深度学习入门指南

深度学习是机器学习的一个分支,使用多层神经网络来学习数据中的模式。

神经网络的基本组件:
1. 神经元: 接收输入,经过权重和偏置计算后输出
2. 激活函数: 引入非线性,如ReLU、Sigmoid、Tanh
3. 损失函数: 衡量预测值与真实值的差距
4. 优化器: 更新网络参数以最小化损失

常见的神经网络类型:
- CNN(卷积神经网络): 适用于图像处理
- RNN(循环神经网络): 适用于序列数据
- Transformer: 基于注意力机制,适用于NLP任务

使用PyTorch构建神经网络的基本步骤:
1. 定义网络结构(class继承nn.Module)
2. 加载数据(使用DataLoader)
3. 定义损失函数和优化器
4. 训练循环(前向传播、计算损失、反向传播、更新参数)
5. 评估模型
""",
}


print("=" * 60)
print("Batch Download: Python Repos + Chinese Corpus")
print("=" * 60)

total_py = 0

# Phase 1: Python repos
print(f"\n--- Python Repos ---")
for line in PYTHON_REPOS.strip().split("\n"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    parts = line.split()
    repo = parts[0]
    branch = parts[1] if len(parts) > 1 else "main"
    n = clone(repo, branch)
    total_py += n

# Phase 2: Chinese Python tutorials
print(f"\n--- Chinese Tutorials ---")
for filename, code in CHINESE_TUTORIALS.items():
    write_file(os.path.join(CHINESE_DIR, filename), code)

# Phase 3: Chinese text corpus
print(f"\n--- Chinese Text Corpus ---")
for filename, text in CHINESE_TEXT_CORPUS.items():
    write_file(os.path.join(CHINESE_DIR, filename), text)

# Phase 4: Copy these to source too (so data loader picks them up)
print(f"\n--- Copying to source/ ---")
import shutil
for f in Path(CHINESE_DIR).glob("*.py"):
    shutil.copy2(f, Path(DATA_DIR) / ("_cn_" + f.name))
    print(f"  Copy _cn_{f.name}")
for f in Path(CHINESE_DIR).glob("*.txt"):
    shutil.copy2(f, Path(DATA_DIR) / ("_cn_" + f.name))
    print(f"  Copy _cn_{f.name}")

# Stats
import shutil
all_py = list(Path(DATA_DIR).rglob("*.py"))
all_txt = list(Path(DATA_DIR).rglob("*.txt")) + list(Path(DATA_DIR).rglob("*.md"))
total_mb = sum(f.stat().st_size for f in all_py) / (1024*1024)
print(f"\n{'=' * 60}")
print(f"Done! Total: {len(all_py)} .py + {len(all_txt)} .txt/.md = {len(all_py)+len(all_txt)} files")
print(f"Size: {total_mb:.1f} MB")
print(f"{'=' * 60}")
