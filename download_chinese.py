"""
Download Chinese conversational + technical corpus.
Focus: everyday Chinese, technical discussions, code descriptions in Chinese.
"""
import os, json, urllib.request, gzip, io
from pathlib import Path

CHINESE_DIR = "sample_data/chinese_text"
os.makedirs(CHINESE_DIR, exist_ok=True)

def write(path, content):
    Path(os.path.join(CHINESE_DIR, path)).write_text(content.strip() + "\n", encoding="utf-8")
    size = len(content)
    print(f"  [OK] {path} ({size} chars)")


# =============================================================
# Tier 1: Chinese conversational data - everyday topics
# =============================================================

CN_CONVERSATIONS = """
你好，请问你能帮我写代码吗？
当然可以，你想写什么样的代码？
我想写一个计算器程序。
好的，我可以用Python帮你写一个简单的计算器。
太感谢了！
不客气，这是代码：def calculator():
    pass

今天天气真好，适合出去走走。
是啊，不过我要在家写代码。
写什么代码？
在学Python，写一个爬虫程序。
加油！Python很有趣的。

能帮我看看这个bug吗？
什么bug？把代码发给我。
我的函数总是返回None，不知道为什么。
你忘了写return语句了吧？
啊！真的是，谢谢！
不客气，这是最常见的错误之一。

我想学习机器学习，从哪里开始好？
建议先学Python基础，然后学NumPy和Pandas。
之后呢？
然后学Scikit-learn，做几个项目就有感觉了。
好的，谢谢建议！

这个程序运行太慢了，怎么优化？
先找出瓶颈在哪里，用profile工具。
然后呢？
然后用更高效的数据结构，或者用numpy向量化。
明白了，我试试看。

什么是深度学习？
深度学习是机器学习的一个分支，使用多层神经网络。
它和普通机器学习有什么区别？
它可以自动提取特征，不需要手动做特征工程。
听起来很厉害。

pip install总是失败，怎么办？
试试用国内镜像源。
什么镜像源？
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 包名
真的可以了！谢谢！

Python2和Python3有什么区别？
Python3是未来，Python2已经停止维护了。
主要区别呢？
print变成函数、整数除法行为不同、字符串默认Unicode。
好的，那我直接学Python3。

什么是Git？怎么用？
Git是版本控制工具，用来管理代码历史。
基本命令有哪些？
git init, git add, git commit, git push, git pull
听起来有点复杂，但很强大。

你有什么推荐的Python书吗？
《流畅的Python》很不错，适合进阶。
有适合初学者的吗？
《Python编程从入门到实践》很适合新手。
好的，我去看看。

Jupyter Notebook是什么？
它是一个交互式编程环境，很适合数据分析和教学。
怎么安装？
pip install jupyter，然后运行jupyter notebook。
好的，我试试。
""".strip()

write("conversations_cn.txt", CN_CONVERSATIONS)


# =============================================================
# Tier 2: Chinese code-task pairs - Chinese description + code
# =============================================================

CN_CODE_PAIRS = """
# 任务：用Python写一个函数，判断一个数字是不是质数
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

# 任务：写一个函数，把列表里的所有偶数找出来
def find_even_numbers(numbers):
    return [n for n in numbers if n % 2 == 0]

# 任务：写一个函数，统计字符串里每个字符出现的次数
def count_chars(s):
    result = {}
    for c in s:
        result[c] = result.get(c, 0) + 1
    return result

# 任务：写一个函数，反转链表
def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    return prev

# 任务：写一个二分查找算法
def binary_search(arr, target):
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

# 任务：读取CSV文件并打印每一行
import csv
def read_csv_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)

# 任务：用Python发送HTTP请求获取网页内容
import requests
def fetch_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

# 任务：计算两个日期之间相差的天数
from datetime import datetime
def days_between(date1, date2):
    d1 = datetime.strptime(date1, '%Y-%m-%d')
    d2 = datetime.strptime(date2, '%Y-%m-%d')
    return abs((d2 - d1).days)

# 任务：把字典按值排序
def sort_dict_by_value(d, reverse=False):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))

# 任务：生成斐波那契数列的前N项
def fibonacci_sequence(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib

# 任务：用类来实现一个简单的银行账户
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        return f"存款成功，余额: {self.balance}"

    def withdraw(self, amount):
        if amount > self.balance:
            return "余额不足"
        self.balance -= amount
        return f"取款成功，余额: {self.balance}"

# 任务：写一个装饰器，计算函数的执行时间
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__}执行了{end - start:.4f}秒")
        return result
    return wrapper
""".strip()

write("code_pairs_cn.txt", CN_CODE_PAIRS)


# =============================================================
# Tier 3: Chinese technical explanations
# =============================================================

CN_TECH = """
什么是面向对象编程？

面向对象编程是一种编程范式，把程序看作对象的集合。每个对象包含数据和操作数据的方法。

三大特性：
1. 封装：把数据和操作数据的方法包装在一起
2. 继承：子类可以继承父类的属性和方法
3. 多态：同一个方法在不同类中有不同的实现

什么是API？

API是应用程序编程接口的缩写。它是一组定义好的规则，让不同的软件组件之间可以互相通信。

比如，你想知道今天的天气，可以调用天气API。API会返回一个JSON格式的数据，包含温度、湿度、风速等信息。

什么是数据库索引？

数据库索引是一种数据结构，用于加快数据的检索速度。就像书的目录一样，通过索引可以快速找到需要的数据，而不需要一页一页翻。

常见的索引类型有B树索引和哈希索引。

什么是递归？

递归是一种编程技巧，函数调用自己来解决更小规模的相同问题。递归必须有终止条件，否则会无限循环。

经典的例子是计算阶乘：
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

什么是JSON？

JSON是一种轻量级的数据交换格式。它基于JavaScript语法，但被多种编程语言支持。

JSON格式示例：
{
    "name": "张三",
    "age": 25,
    "city": "北京",
    "hobbies": ["编程", "读书"]
}

什么是HTTP协议？

HTTP是超文本传输协议，是互联网上应用最广泛的协议。客户端发送HTTP请求到服务器，服务器返回HTTP响应。

常见的HTTP方法：
- GET：获取资源
- POST：创建资源
- PUT：更新资源
- DELETE：删除资源

常见的HTTP状态码：
- 200：请求成功
- 404：资源未找到
- 500：服务器内部错误

什么是Docker？

Docker是一个容器化平台，可以把应用程序及其依赖打包到一个容器中。容器之间相互隔离，但比虚拟机更轻量。

Docker的核心概念：
- 镜像：只读模板，用于创建容器
- 容器：镜像的运行实例
- Dockerfile：用于构建镜像的配置文件
- 仓库：存储和分享镜像的地方

什么是机器学习？

机器学习是人工智能的一个分支，让计算机从数据中学习规律，而不需要显式编程。

主要类型：
1. 监督学习：使用标注数据训练模型
2. 无监督学习：使用未标注数据发现模式
3. 强化学习：通过奖励机制学习策略

常见的算法：
- 线性回归：预测连续值
- 决策树：分类和回归
- SVM：分类
- K-means：聚类
- 神经网络：深度学习的基础
""".strip()

write("tech_explanations_cn.txt", CN_TECH)


# =============================================================
# Tier 4: Chinese daily chat
# =============================================================

CN_DAILY = """
你好，今天过得怎么样？
还不错，刚写完一个Python程序。
什么程序？
一个爬虫，自动抓取新闻标题。
挺厉害的，学了多久Python？
大概三个月。

你吃饭了吗？
吃了，一边吃一边在想代码的问题。
什么问题？
怎么优化数据库查询，太慢了。
试试加索引，应该会快很多。

周末有什么计划？
打算学一下FastAPI。
FastAPI是什么？
一个Python的Web框架，性能很好，写API很方便。
听起来不错，我也感兴趣。

最近在看什么书？
在看《深度学习入门》，写的很好。
是那本用PyTorch的吗？
对，很适合初学者。
好的，我也去买一本。

你一般用什么IDE？
我用VS Code，插件很丰富。
有什么推荐的插件吗？
Python插件、GitLens、还有GitHub Copilot。
好的，我试试。

今天晚上准备做什么？
准备复习一下数据结构。
主要复习什么？
二叉树、图、还有动态规划。
这些都是面试常考的。

你觉得AI会取代程序员吗？
不会，AI是工具，不是替代品。
但是ChatGPT已经能写代码了。
它能写简单代码，但复杂系统的设计还是需要人。
说得对。

你最喜欢Python的哪个特性？
列表推导式，写起来很优雅。
举个例子？
[x**2 for x in range(10)] 一行代码生成平方列表。
确实很简洁。

想学新的编程语言，推荐哪个？
看你想做什么。Web开发推荐JavaScript或TypeScript。
数据科学和AI呢？
那还是Python最合适。
那我就继续学Python吧。

写代码遇到bug怎么办？
先看错误信息，Python的错误信息很友好。
如果看不懂呢？
复制到搜索引擎搜一下，StackOverflow上基本都有答案。
好的，我试试这个方法。

什么是技术债务？
技术债务是指为了快速交付而写的糟糕代码，以后需要花时间重构。
怎么避免？
写代码时多想一步，做好设计和测试。
明白了。
""".strip()

write("daily_chat_cn.txt", CN_DAILY)


# =============================================================
# Summary
# =============================================================
all_files = list(Path(CHINESE_DIR).rglob("*"))
total_chars = sum(f.read_text(encoding="utf-8").count("\n") for f in all_files if f.is_file())
print(f"\nTotal Chinese files: {len(all_files)}")
print(f"Total lines: {total_chars}")
print(f"Saved to: {CHINESE_DIR}")

# Copy to source too
import shutil
for f in Path(CHINESE_DIR).glob("*"):
    dst = Path("sample_data/source") / ("_zh_" + f.name)
    shutil.copy2(f, dst)
    print(f"  Copied to source/: _zh_{f.name}")
