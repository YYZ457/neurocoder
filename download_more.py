"""
Phase 2: Download more data + Chinese text corpus.
"""
import os, subprocess, sys, json
from pathlib import Path

DATA_DIR = "sample_data/source"
os.makedirs(DATA_DIR, exist_ok=True)


def clone(repo, branch="main", depth=1):
    name = repo.split("/")[1]
    dest = Path(DATA_DIR) / name
    if dest.exists():
        return
    url = f"https://github.com/{repo}.git"
    try:
        subprocess.run(["git", "clone", "--depth", str(depth), "--branch", branch, url, str(dest)],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180, check=True)
        py = len(list(dest.rglob("*.py")))
        print(f"  [OK] {name} ({py} .py)")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")


# ===========================================================================
# More high-quality Python repos
# ===========================================================================
MORE_REPOS = [
    # Web
    ("encode/httpx", "master"),
    ("tiangolo/typer", "master"),
    ("samuelcolvin/pydantic", "main"),
    # Data
    ("apache/arrow", "main"),
    ("dask/dask", "main"),
    # AI/ML
    ("huggingface/transformers", "main"),
    ("scikit-learn/scikit-learn", "main"),
    # Tools
    ("python-poetry/poetry", "master"),
    ("pypa/pip", "main"),
    ("microsoft/pyright", "main"),
    # Async
    ("aio-libs/aiohttp", "master"),
    # Stdlib-like
    ("python-attrs/attrs", "main"),
    ("grantjenks/python-sortedcontainers", "master"),
    # Chinese-friendly repos
    ("shibing624/pytextclassifier", "master"),
    ("shibing624/text2vec", "master"),
    ("chineseocr/chineseocr", "master"),
    ("6mb/Microsoft-Activation-Scripts", "master"),
]

print("--- More Python Repos ---")
for repo, branch in MORE_REPOS:
    clone(repo, branch)


# ===========================================================================
# Generate Chinese + Python paired data
# This teaches the model Chinese natural language → Python code
# ===========================================================================
CHINESE_PAIRS_DIR = Path(DATA_DIR) / "_chinese_code_pairs"
CHINESE_PAIRS_DIR.mkdir(exist_ok=True)

chinese_pairs = {
    "01_basic_operations.py": '''# 任务：创建一个列表，包含数字1到10，然后打印所有偶数
# Task: Create a list of numbers 1 to 10, then print all even numbers

# 首先，创建一个包含数字1到10的列表
numbers = list(range(1, 11))

# 然后，使用列表推导式找出所有偶数
even_numbers = [n for n in numbers if n % 2 == 0]

# 打印结果
print("偶数有:", even_numbers)

# 检查答案是否正确
assert even_numbers == [2, 4, 6, 8, 10]
print("验证通过！")
''',

    "02_string_processing.py": '''# 需求：统计一段中文文本中每个字出现的次数
# 用户输入的文本
text = "我爱北京天安门，天安门上太阳升"

# 创建一个字典来存储统计结果
char_count = {}

# 遍历文本中的每个字
for char in text:
    # 跳过标点符号
    if char in "，。！？、；：""''（）":
        continue
    # 如果这个字已经在字典中，加1；否则设为1
    if char in char_count:
        char_count[char] += 1
    else:
        char_count[char] = 1

# 按出现次数从多到少排序
sorted_chars = sorted(char_count.items(), key=lambda x: x[1], reverse=True)

# 打印统计结果
print("字频统计结果：")
for char, count in sorted_chars:
    print(f"  '{char}' 出现了 {count} 次")

# 预期结果
expected = {'我': 1, '爱': 1, '北': 1, '京': 1, '天': 2, '安': 2, '门': 2, '上': 2, '太': 1, '阳': 1, '升': 1}
print(f"\\n验证: {'通过' if char_count == expected else '需要检查'}")
''',

    "03_api_design.py": '''# 任务：设计一个简单的用户管理系统
# 要求：能够添加用户、删除用户、查找用户、列出所有用户

from typing import Optional, List, Dict
from datetime import datetime

class UserManager:
    """用户管理系统。
    支持添加、删除、查找、列出用户。
    每个用户有：用户名、邮箱、注册时间。
    """

    def __init__(self):
        """初始化用户管理器。"""
        self._users: Dict[str, Dict] = {}  # 用户名字 → 用户信息

    def add_user(self, username: str, email: str) -> bool:
        """添加新用户。如果用户名已存在则返回False。"""
        if username in self._users:
            print(f"错误：用户名 '{username}' 已存在")
            return False

        # 验证邮箱格式（简单检查）
        if "@" not in email or "." not in email:
            print(f"错误：邮箱格式不正确: {email}")
            return False

        self._users[username] = {
            "username": username,
            "email": email,
            "created_at": datetime.now().isoformat(),
        }
        print(f"用户 '{username}' 添加成功")
        return True

    def remove_user(self, username: str) -> bool:
        """删除用户。如果用户不存在则返回False。"""
        if username not in self._users:
            print(f"错误：用户 '{username}' 不存在")
            return False
        del self._users[username]
        print(f"用户 '{username}' 已删除")
        return True

    def find_user(self, username: str) -> Optional[Dict]:
        """查找用户，返回用户信息或None。"""
        return self._users.get(username)

    def list_users(self) -> List[Dict]:
        """列出所有用户的信息。"""
        return list(self._users.values())

# 使用示例
if __name__ == "__main__":
    manager = UserManager()

    # 添加几个用户
    manager.add_user("张三", "zhangsan@example.com")
    manager.add_user("李四", "lisi@example.com")
    manager.add_user("王五", "wangwu@example.com")

    # 查找用户
    user = manager.find_user("李四")
    if user:
        print(f"找到用户: {user['username']}, 邮箱: {user['email']}")

    # 列出所有用户
    print(f"\\n当前共有 {len(manager.list_users())} 个用户：")
    for u in manager.list_users():
        print(f"  - {u['username']} ({u['email']})")

    # 删除用户
    manager.remove_user("王五")
    print(f"删除后还剩 {len(manager.list_users())} 个用户")
''',

    "04_data_analysis.py": '''# 需求：分析学生成绩数据，找出最高分、最低分、平均分、及格率

# 学生成绩数据
students_scores = [
    {"name": "张三", "math": 85, "english": 92, "chinese": 78},
    {"name": "李四", "math": 45, "english": 60, "chinese": 55},
    {"name": "王五", "math": 95, "english": 88, "chinese": 91},
    {"name": "赵六", "math": 72, "english": 65, "chinese": 80},
    {"name": "孙七", "math": 58, "english": 71, "chinese": 62},
]

def analyze_scores(data):
    """分析学生成绩数据，返回统计结果。"""
    subjects = ["math", "english", "chinese"]
    subject_names = {"math": "数学", "english": "英语", "chinese": "语文"}

    report = {}

    for subject in subjects:
        scores = [s[subject] for s in data]

        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        pass_rate = sum(1 for s in scores if s >= 60) / len(scores) * 100

        report[subject_names[subject]] = {
            "平均分": round(avg_score, 1),
            "最高分": max_score,
            "最低分": min_score,
            "及格率": f"{pass_rate:.1f}%",
        }

    return report

# 执行分析
result = analyze_scores(students_scores)

# 打印分析报告
print("=" * 40)
print("        学生成绩分析报告")
print("=" * 40)

for subject, stats in result.items():
    print(f"\\n{subject}:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

# 找出总分最高的学生
best_student = max(students_scores,
    key=lambda s: s["math"] + s["english"] + s["chinese"])
print(f"\\n总分最高的学生: {best_student['name']}")
total = best_student["math"] + best_student["english"] + best_student["chinese"]
print(f"总分: {total} / 300")
''',

    "05_web_scraper.py": '''# 需求：写一个简单的网页爬虫，获取网页标题和所有链接
# 使用 requests 和 BeautifulSoup

import re
from typing import List, Tuple

# 模拟网页爬虫（实际使用时需要 requests 和 bs4）
# import requests
# from bs4 import BeautifulSoup

def extract_title(html: str) -> str:
    """从HTML内容中提取网页标题。"""
    match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "无标题"

def extract_links(html: str) -> List[Tuple[str, str]]:
    """从HTML内容中提取所有超链接。
    返回 (链接文本, URL) 的列表。
    """
    # 匹配 <a href="URL">文本</a>
    pattern = r'<a\\s+(?:[^>]*?\\s+)?href="([^"]*)"[^>]*>(.*?)</a>'
    matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)

    links = []
    for url, text in matches:
        # 清理文本中的HTML标签
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text:
            links.append((text, url))

    return links

def is_valid_url(url: str) -> bool:
    """检查URL是否有效。"""
    pattern = r"^https?://[^\\s/$.?#].[^\\s]*$"
    return bool(re.match(pattern, url))

# 测试数据（模拟一个简单网页）
test_html = """
<html>
<head>
    <title>Python学习资源</title>
</head>
<body>
    <h1>欢迎来到Python学习网站</h1>
    <p>以下是一些有用的链接：</p>
    <ul>
        <li><a href="https://python.org">Python官方网站</a></li>
        <li><a href="https://github.com">GitHub代码托管</a></li>
        <li><a href="https://stackoverflow.com">Stack Overflow问答</a></li>
    </ul>
    <p>联系方式: <a href="mailto:admin@example.com">发送邮件</a></p>
</body>
</html>
"""

# 测试爬虫功能
print("网页标题:", extract_title(test_html))
print("\\n提取到的链接:")
for text, url in extract_links(test_html):
    valid = "有效" if is_valid_url(url) else "无效"
    print(f"  [{valid}] {text} -> {url}")
''',

    "06_async_example.py": '''# 需求：异步下载多个网页的内容
# 使用 asyncio 和 aiohttp 实现并发请求

import asyncio
from typing import List

# 注意：这是示例代码，实际运行需要 aiohttp
# import aiohttp

async def fetch_url(url: str, delay: float = 0.5) -> str:
    """模拟异步获取URL的内容。
    实际使用时替换为真实的HTTP请求。
    """
    # 模拟网络延迟
    await asyncio.sleep(delay)
    return f"<content from {url}>"

    # 实际实现（需要 aiohttp）：
    # async with aiohttp.ClientSession() as session:
    #     async with session.get(url) as response:
    #         return await response.text()

async def download_all(urls: List[str]) -> List[str]:
    """并发下载所有URL的内容。
    使用 asyncio.gather 同时发起多个请求，
    而不是一个一个地等待。
    """
    tasks = []
    for i, url in enumerate(urls):
        # 创建异步任务，每个任务有不同的延迟
        task = fetch_url(url, delay=0.1 * (i + 1))
        tasks.append(task)

    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    return results

async def main():
    """主函数：演示异步下载。"""
    urls = [
        "https://api.example.com/data/1",
        "https://api.example.com/data/2",
        "https://api.example.com/data/3",
        "https://api.example.com/data/4",
        "https://api.example.com/data/5",
    ]

    print("开始并发下载...")
    print(f"共 {len(urls)} 个请求\\n")

    # 记录开始时间
    import time
    start = time.time()

    # 并发下载
    contents = await download_all(urls)

    elapsed = time.time() - start

    # 打印结果
    for url, content in zip(urls, contents):
        print(f"  {url} -> {len(content)} 字节")

    print(f"\\n全部完成！耗时: {elapsed:.2f} 秒")
    print(f"如果串行下载需要约 {sum(0.1*i for i in range(1,len(urls)+1)):.1f} 秒")
    print(f"异步提速约 {sum(0.1*i for i in range(1,len(urls)+1))/elapsed:.1f} 倍")

# 运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())
''',
}

for filename, code in chinese_pairs.items():
    filepath = CHINESE_PAIRS_DIR / filename
    code = code.strip() + "\n"
    filepath.write_text(code, encoding="utf-8")
    print(f"  Generated: {filename} ({len(code)} chars)")


# ===========================================================================
# Generate pure Chinese text for language understanding
# ===========================================================================
CHINESE_TEXT_DIR = Path(DATA_DIR) / "_chinese_text"
CHINESE_TEXT_DIR.mkdir(exist_ok=True)

# Chinese technical text — teaches the model Chinese tech vocabulary
chinese_text = """
# Python编程入门指南

Python是一种解释型、面向对象的高级编程语言。它具有简洁的语法和强大的功能，
是目前最受欢迎的编程语言之一。

## 为什么选择Python？

Python的设计哲学强调代码的可读性和简洁的语法。相比其他编程语言，
Python的代码更像英语自然语言，这使得初学者更容易上手。

Python广泛应用于以下领域：
- 数据科学和机器学习
- Web开发（Django、Flask、FastAPI）
- 自动化脚本和运维
- 科学计算和数据分析
- 人工智能和深度学习

## Python的核心特性

第一，Python是动态类型语言，变量不需要声明类型。
第二，Python使用缩进来表示代码块，而不是花括号。
第三，Python有丰富的标准库和第三方库生态。

## 如何开始学习Python

建议从基础语法开始，然后学习常用的数据结构和算法。
接下来可以选择一个方向深入学习，比如Web开发或数据科学。
最重要的是多写代码，多做项目，在实践中学习。

# 常见编程模式和最佳实践

## 命名规范

在Python中，我们通常遵循PEP 8命名规范：
- 变量和函数使用小写字母加下划线（snake_case）
- 类名使用大驼峰命名法（PascalCase）
- 常量使用全大写字母加下划线
- 私有属性以单下划线开头

## 代码组织

一个好的Python项目通常包含以下结构：
- src/ 或 app/ 目录存放源代码
- tests/ 目录存放测试文件
- docs/ 目录存放文档
- requirements.txt 列出依赖包

## 错误处理的哲学

Python社区有一个著名的设计原则：宁可通过异常来处理错误，
而不是通过返回特殊值。这样可以让代码更加清晰和健壮。

# Python生态系统

Python的成功很大程度上归功于其丰富的生态系统。
无论是Web开发、数据分析、机器学习还是自动化运维，
都能找到成熟可靠的第三方库。

最受欢迎的Python库包括：
NumPy用于科学计算，Pandas用于数据分析，
Matplotlib用于数据可视化，Scikit-learn用于机器学习，
Django和Flask用于Web开发，Requests用于HTTP请求。
""".strip()

(chinese_text_dir := CHINESE_TEXT_DIR / "python_guide_cn.txt").write_text(chinese_text, encoding="utf-8")
print(f"\n  Generated Chinese text: python_guide_cn.txt ({len(chinese_text)} chars)")


# ===========================================================================
# Final stats
# ===========================================================================
all_py = list(Path(DATA_DIR).rglob("*.py"))
all_files = list(Path(DATA_DIR).rglob("*"))
print(f"\n{'=' * 60}")
print(f"Total .py files: {len(all_py)}")
print(f"Total files: {len(all_files)}")
total_mb = sum(f.stat().st_size for f in all_py) / (1024 * 1024)
print(f"Total .py size: {total_mb:.1f} MB")
print(f"{'=' * 60}")
