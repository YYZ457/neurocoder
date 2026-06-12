"""
Download high-quality Python training data.
Includes English and Chinese Python repos.
"""
import os, subprocess, sys, shutil, re, time
from pathlib import Path

DATA_DIR = "D:/NeuroCoder/sample_data/source"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Tier 1: Core Python (English) — high quality, learn good coding patterns
# ---------------------------------------------------------------------------
ENGLISH_REPOS = [
    # Python core
    ("python/cpython", "main", 1),
    # Web frameworks
    ("pallets/flask", "main", 2),
    ("tiangolo/fastapi", "master", 2),
    ("django/django", "main", 3),
    # Data science
    ("numpy/numpy", "main", 3),
    ("pandas-dev/pandas", "main", 3),
    # Tools
    ("psf/black", "main", 2),
    ("astral-sh/ruff", "main", 2),
    ("python/mypy", "master", 2),
    # AI/ML
    ("pytorch/pytorch", "main", 4),
    # Async
    ("python-trio/trio", "main", 2),
    # Testing
    ("pytest-dev/pytest", "main", 2),
    # Typing
    ("python/typing", "main", 1),
]

# ---------------------------------------------------------------------------
# Tier 2: Chinese Python projects (Chinese comments, docs, variable names)
# ---------------------------------------------------------------------------
CHINESE_REPOS = [
    # Chinese NLP tools (many Chinese comments)
    ("HankMcGee/handright", "master", None),
    ("yangjianxin1/GPT级中文聊天模型", "main", None),
    # Chinese open source Python libs
    ("mozillazg/python-pinyin", "master", None),
    ("fxsjy/jieba", "master", None),
    # Chinese docs/tutorials
    ("Prodesire/faceai", "master", None),
    ("liuhuanyong/QASystemOnMedicalKG", "master", None),
    # More Chinese NLP/ML
    ("shibing624/pycorrector", "master", None),
    ("shibing624/text2vec", "master", None),
    ("wainshine/Chinese-Names-Corpus", "master", None),
    # Chinese AI
    ("THUDM/ChatGLM-6B", "main", None),
    ("ymcui/Chinese-LLaMA-Alpaca", "main", None),
]

# ---------------------------------------------------------------------------
# Tier 3: Chinese Python tutorials (generate from known tutorials)
# ---------------------------------------------------------------------------
CHINESE_TUTORIAL_FILES = [
    # We'll write a few manually constructed Chinese-Python files
    # to bootstrap Chinese understanding
]


def clone_repo(repo, branch, depth=None, target_dir=None):
    """Clone a GitHub repo, shallow if possible."""
    name = repo.split("/")[1]
    dest = Path(target_dir or DATA_DIR) / name
    if dest.exists():
        print(f"  [Skip] {name} already exists")
        return True

    url = f"https://github.com/{repo}.git"
    print(f"  Cloning {repo} ({branch})...")
    cmd = ["git", "clone", "--branch", branch, "--single-branch"]
    if depth:
        cmd += ["--depth", str(depth)]
    cmd += [url, str(dest)]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                      timeout=300, check=True)
        print(f"    [OK] {name}")
        return True
    except Exception as e:
        print(f"    [FAIL] {name}: {e}")
        # Try shallow clone as fallback
        try:
            print(f"    Retrying with --depth 1...")
            subprocess.run(["git", "clone", "--depth", "1", url, str(dest)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         timeout=300, check=True)
            print(f"    [OK] {name} (shallow)")
            return True
        except Exception:
            return False


def generate_chinese_tutorials():
    """Generate Python files with Chinese comments for training."""
    tutorial_dir = Path(DATA_DIR) / "_chinese_tutorials"
    tutorial_dir.mkdir(exist_ok=True)

    files_written = 0

    # Basic Python concepts with Chinese explanations
    tutorials = {
        "01_variables.py": '''
# Python 变量和数据类型
# 在Python中，你不需要声明变量类型

# 整数类型 (integer)
age = 25  # 年龄
count = 100  # 计数

# 浮点数 (float)
price = 19.99  # 价格
pi = 3.14159  # 圆周率

# 字符串 (string)
name = "张三"  # 姓名
message = '你好，世界！'  # 消息

# 布尔值 (boolean)
is_valid = True  # 是否有效
is_empty = False  # 是否为空

# 列表 (list) — 有序可变集合
fruits = ["苹果", "香蕉", "橘子"]  # 水果列表
numbers = [1, 2, 3, 4, 5]  # 数字列表

# 字典 (dict) — 键值对映射
person = {"name": "李四", "age": 30, "city": "北京"}  # 个人信息

# 打印输出
print(f"姓名: {name}, 年龄: {age}")
''',

        "02_functions.py": '''
# Python 函数 — 封装可复用的代码
# def 关键字用于定义函数

def greet(name):
    """向用户打招呼的函数。参数 name 是用户的名字。"""
    return f"你好，{name}！欢迎使用Python。"

def calculate_sum(numbers):
    """计算列表中所有数字的总和。"""
    total = 0
    for num in numbers:
        total += num
    return total

def find_max(values):
    """找出列表中的最大值。"""
    if not values:  # 如果列表为空
        return None
    max_val = values[0]
    for v in values:
        if v > max_val:
            max_val = v
    return max_val

def is_palindrome(s):
    """判断字符串是否为回文（正读反读都一样）。"""
    # 去除空格并转为小写
    s = s.replace(" ", "").lower()
    # 比较字符串和它的反转
    return s == s[::-1]

# 测试函数
print(greet("小明"))
print(f"总和: {calculate_sum([1, 2, 3, 4])}")
print(f"最大值: {find_max([3, 7, 2, 9, 1])}")
print(f"回文: {is_palindrome('上海自来水来自海上')}")
''',

        "03_classes.py": '''
# Python 类和面向对象编程
# class 关键字用于定义类

class Student:
    """学生类 — 表示一个学生对象。"""

    def __init__(self, name, grade):
        """构造函数：初始化学生对象。
        name: 学生姓名
        grade: 年级
        """
        self.name = name
        self.grade = grade
        self.scores = {}  # 科目 -> 分数

    def add_score(self, subject, score):
        """添加一门课的成绩。"""
        self.scores[subject] = score

    def get_average(self):
        """计算所有科目的平均分。"""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def __str__(self):
        """返回学生的字符串表示。"""
        return f"学生: {self.name}, 年级: {self.grade}"


class BankAccount:
    """银行账户类 — 演示封装和属性。"""

    def __init__(self, owner, balance=0):
        self.owner = owner  # 账户持有人
        self._balance = balance  # 余额（私有属性）

    @property
    def balance(self):
        """获取当前余额。"""
        return self._balance

    def deposit(self, amount):
        """存款操作。amount: 存款金额"""
        if amount <= 0:
            raise ValueError("存款金额必须大于0")
        self._balance += amount
        print(f"存款 {amount} 元成功，当前余额: {self._balance} 元")

    def withdraw(self, amount):
        """取款操作。amount: 取款金额"""
        if amount <= 0:
            raise ValueError("取款金额必须大于0")
        if amount > self._balance:
            raise ValueError("余额不足！")
        self._balance -= amount
        print(f"取款 {amount} 元成功，当前余额: {self._balance} 元")


# 使用示例
student = Student("王五", "高三")
student.add_score("数学", 95)
student.add_score("语文", 88)
student.add_score("英语", 92)
print(student)
print(f"平均分: {student.get_average():.1f}")

account = BankAccount("赵六", 1000)
account.deposit(500)
account.withdraw(200)
print(f"最终余额: {account.balance} 元")
''',

        "04_algorithms.py": '''
# Python 常用算法实现
# 包含排序、搜索、数据结构等

def bubble_sort(arr):
    """冒泡排序 — 每次将最大的元素冒泡到最后。"""
    n = len(arr)
    for i in range(n):
        # 优化：如果一轮没有交换，说明已经有序
        swapped = False
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break  # 提前退出
    return arr

def quick_sort(arr):
    """快速排序 — 分治策略，选取基准值分区。"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]  # 选择中间元素作为基准
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def binary_search(arr, target):
    """二分查找 — 在有序数组中查找目标值。
    返回目标值的索引，如果不存在则返回 -1。"""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2  # 计算中间位置
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1  # 目标在右半部分
        else:
            right = mid - 1  # 目标在左半部分
    return -1  # 未找到

def fibonacci(n):
    """生成斐波那契数列的前 n 项。"""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    result = [0, 1]
    for i in range(2, n):
        result.append(result[i-1] + result[i-2])
    return result

# 测试
print(f"冒泡排序: {bubble_sort([64, 34, 25, 12, 22, 11, 90])}")
print(f"快速排序: {quick_sort([3, 6, 8, 10, 1, 2, 1])}")
print(f"二分查找: {binary_search([1, 3, 5, 7, 9, 11], 7)}")
print(f"斐波那契: {fibonacci(10)}")
''',

        "05_file_io.py": '''
# Python 文件操作
# 读写文件是编程中最常见的操作之一

import os
import json
import csv

def read_text_file(filepath):
    """读取文本文件的所有内容。"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()  # 一次性读取全部内容
    return content

def read_lines(filepath):
    """按行读取文件，适用于大文件。"""
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # 去除首尾空白字符
            if line:  # 跳过空行
                lines.append(line)
    return lines

def write_text_file(filepath, content):
    """将内容写入文本文件。"""
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"文件已保存: {filepath}")

def read_csv(filepath):
    """读取 CSV 文件并返回数据列表。"""
    data = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)  # 使用字典读取器
        for row in reader:
            data.append(row)
    return data

def write_json(filepath, data):
    """将数据保存为 JSON 文件。"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"JSON 文件已保存: {filepath}")

# 测试示例
test_content = "这是第一行\\n这是第二行\\n这是第三行"
# 实际使用时取消注释以下代码
# write_text_file("test_output.txt", test_content)
# lines = read_lines("test_output.txt")
# print(f"读取到 {len(lines)} 行")
''',

        "06_data_structures.py": '''
# Python 数据结构详解
# 列表、字典、集合、元组的常用操作

from collections import defaultdict, Counter, deque
from typing import List, Dict, Set, Tuple, Optional

def count_frequency(items: List[str]) -> Dict[str, int]:
    """统计列表中每个元素出现的频率。"""
    counter = {}
    for item in items:
        if item in counter:
            counter[item] += 1
        else:
            counter[item] = 1
    return counter

    # 更简洁的写法：
    # return dict(Counter(items))

def find_duplicates(items: List) -> List:
    """找出列表中的重复元素。"""
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)

def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """合并两个字典，如果键重复，dict2 的值覆盖 dict1。"""
    result = dict1.copy()  # 浅拷贝
    result.update(dict2)
    return result

    # Python 3.9+ 可以直接用：
    # return dict1 | dict2

def top_k_items(items: List, k: int = 3) -> List:
    """找出列表中出现次数最多的前 k 个元素。"""
    # 使用 Counter 统计频率
    freq = Counter(items)
    # most_common 返回 [(元素, 频率), ...]
    return freq.most_common(k)

def group_by_key(items: List[Dict], key: str) -> Dict[str, List[Dict]]:
    """根据字典中的某个键对列表进行分组。"""
    groups = defaultdict(list)
    for item in items:
        groups[item[key]].append(item)
    return dict(groups)

# 测试
words = ["apple", "banana", "apple", "orange", "banana", "apple"]
print(f"频率统计: {count_frequency(words)}")
print(f"重复元素: {find_duplicates([1, 2, 3, 2, 4, 1, 5, 1])}")
print(f"前3个: {top_k_items(words, 3)}")

students = [
    {"name": "张三", "class": "一班", "score": 85},
    {"name": "李四", "class": "二班", "score": 92},
    {"name": "王五", "class": "一班", "score": 78},
]
print(f"按班级分组: {group_by_key(students, 'class')}")
''',

        "07_error_handling.py": '''
# Python 错误处理和异常
# try-except-finally 用于优雅地处理错误

import traceback

def safe_divide(a, b):
    """安全除法 — 处理除零错误。"""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("错误：除数不能为零！")
        return None
    except TypeError:
        print("错误：请输入数字类型！")
        return None

def read_config_file(filepath):
    """读取配置文件，处理文件不存在的错误。"""
    import json
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"警告：配置文件 {filepath} 不存在，使用默认配置。")
        return {"version": "1.0", "debug": False}
    except json.JSONDecodeError as e:
        print(f"错误：配置文件格式错误 - {e}")
        return None
    except Exception as e:
        print(f"未知错误：{e}")
        traceback.print_exc()  # 打印完整错误堆栈
        return None

def retry_on_failure(func, max_retries=3):
    """装饰器模式：失败自动重试。"""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"第 {attempt + 1} 次尝试失败: {e}")
                if attempt == max_retries - 1:
                    raise  # 最后一次重试仍失败，抛出异常
    return wrapper

# 自定义异常
class ValidationError(Exception):
    """输入验证失败的异常。"""
    def __init__(self, message, field=None):
        super().__init__(message)
        self.field = field  # 哪个字段验证失败

def validate_age(age):
    """验证年龄的合法性。"""
    if not isinstance(age, (int, float)):
        raise ValidationError("年龄必须是数字", field="age")
    if age < 0 or age > 150:
        raise ValidationError(f"年龄 {age} 不合法，应在0-150之间", field="age")
    return True

# 测试
print(f"10 / 2 = {safe_divide(10, 2)}")
print(f"10 / 0 = {safe_divide(10, 0)}")

try:
    validate_age(200)
except ValidationError as e:
    print(f"验证错误: {e} (字段: {e.field})")
''',
    }

    for filename, code in tutorials.items():
        filepath = tutorial_dir / filename
        # Strip leading newline
        code = code.strip() + "\n"
        filepath.write_text(code, encoding="utf-8")
        files_written += 1
        print(f"  Generated: {filename} ({len(code)} chars)")

    print(f"  Total: {files_written} Chinese tutorial files")
    return str(tutorial_dir)


def main():
    print("=" * 60)
    print("Downloading Python Training Data")
    print("=" * 60)

    total_files = 0

    # Phase 1: Clone English repos
    print("\n--- Tier 1: English Python Repos ---")
    for repo, branch, depth in ENGLISH_REPOS:
        success = clone_repo(repo, branch, depth=depth)
        if success:
            name = repo.split("/")[1]
            repo_dir = Path(DATA_DIR) / name
            py_count = len(list(repo_dir.rglob("*.py")))
            total_files += py_count
            print(f"    {py_count} .py files")

    # Phase 2: Clone Chinese Python repos
    print("\n--- Tier 2: Chinese Python Repos ---")
    for repo, branch, depth in CHINESE_REPOS:
        success = clone_repo(repo, branch, depth=1)
        if success:
            name = repo.split("/")[1]
            repo_dir = Path(DATA_DIR) / name
            py_count = len(list(repo_dir.rglob("*.py")))
            total_files += py_count
            if py_count > 0:
                print(f"    {py_count} .py files")

    # Phase 3: Generate Chinese tutorial files
    print("\n--- Tier 3: Chinese Python Tutorials ---")
    generate_chinese_tutorials()

    # Phase 4: Also download some known Chinese coding interview repos
    print("\n--- Additional Chinese Resources ---")
    chinese_code_repos = [
        "doocs/leetcode",  # LeetCode solutions with Chinese
        "halfrost/LeetCode-Go",  # Algorithm solutions (has Python too)
    ]
    for repo in chinese_code_repos:
        clone_repo(repo, "main", depth=1)

    # Count final
    all_py = list(Path(DATA_DIR).rglob("*.py"))
    print(f"\n{'=' * 60}")
    print(f"Total: {len(all_py)} Python files")
    total_size_mb = sum(f.stat().st_size for f in all_py) / (1024 * 1024)
    print(f"Total size: {total_size_mb:.1f} MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
