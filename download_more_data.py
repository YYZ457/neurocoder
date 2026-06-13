"""
Download MORE Python repos + Chinese text data.
Skips repos already in sample_data/source/.
"""
import os, subprocess, sys, json, urllib.request, random, time
from pathlib import Path

DATA_DIR = "sample_data/source"
os.makedirs(DATA_DIR, exist_ok=True)

EXISTING = set(d.name for d in Path(DATA_DIR).iterdir() if d.is_dir())

def clone(repo, branch="main"):
    name = repo.split("/")[-1]
    if name in EXISTING:
        print(f"  [Skip] {name} already exists")
        return False
    dest = Path(DATA_DIR) / name
    url = f"https://github.com/{repo}.git"
    for try_branch in [branch, "master", "main"]:
        try:
            subprocess.run(["git", "clone", "--depth", "1", "--branch", try_branch, url, str(dest)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120, check=True)
            py = len(list(dest.rglob("*.py")))
            print(f"  [OK] {name} ({py} .py)")
            return True
        except:
            continue
    print(f"  [FAIL] {name}")
    return False

def download_url(url, dest_path):
    try:
        urllib.request.urlretrieve(url, dest_path)
        size = os.path.getsize(dest_path)
        print(f"  [OK] {Path(dest_path).name} ({size/1024:.0f} KB)")
        return True
    except Exception as e:
        print(f"  [FAIL] {url}: {e}")
        return False


print("=" * 60)
print("Phase 1: New Python Repos (not in source/)")
print("=" * 60)

NEW_REPOS = [
    # Web frameworks & tools
    "django-crispy-forms/django-crispy-forms",  # diff name
    "django-import-export/django-import-export",
    "pallets/jinja",
    "encode/starlette",
    "litestar-org/litestar",
    "tiangolo/fastapi",
    "unbit/uwsgi",
    # Data science
    "numpy/numpy",
    "jax-ml/jax",
    "ray-project/ray",
    "dmlc/xgboost",
    "catboost/catboost",
    "apache/spark",
    "ibis-project/ibis",
    "vaexio/vaex",
    "pola-rs/polars",
    # ML/AI (different from pytorch/transformers)
    "deepmind/alphafold",
    "google-research/bert",
    "huggingface/diffusers",
    "huggingface/peft",
    "huggingface/accelerate",
    "microsoft/DeepSpeed",
    "NVIDIA/apex",
    "NVIDIA/Megatron-LM",
    "openai/whisper",
    "openai/gym",
    "facebookresearch/detectron2",
    "facebookresearch/fairseq",
    "facebookresearch/llama",
    "google/jax",
    "keras-team/keras",
    "davidsandberg/facenet",
    # Dev tools
    "psf/black",
    "python/mypy",
    "pytest-dev/pytest",
    "pypa/pip",
    "pypa/virtualenv",
    "conda/conda",
    "n8henrie/pywal",
    "nvbn/thefuck",
    "tqdm/tqdm",
    "yt-dlp/yt-dlp",
    "ageitgey/face_recognition",
    # Async
    "python-trio/trio",
    "MagicStack/httptools",
    "aaugustin/websockets",
    # Databases
    "sqlalchemy/sqlalchemy",
    "redis/redis-py",
    "mongodb/mongo-python-driver",
    # Testing
    "pytest-dev/pytest-cov",
    "nedbat/coveragepy",
    "locustio/locust",
    # Utilities
    "pydantic/pydantic",
    "python-attrs/attrs",
    "python-pillow/Pillow",
    "benfred/py-spy",
    # Games
    "pygame/pygame",
]

for repo in NEW_REPOS:
    try:
        clone(repo)
    except Exception as e:
        print(f"  [FAIL] {repo}: {e}")


print("\n" + "=" * 60)
print("Phase 2: Chinese Conversation & Text Data")
print("=" * 60)

CHINESE_DIR = Path(DATA_DIR) / "_chinese_data"
CHINESE_DIR.mkdir(exist_ok=True)

# Generate synthetic Chinese chat data (more realistic than before)
print("  Generating Chinese conversation data...")

chat_data = []
topics = [
    "你好", "今天天气怎么样", "你会做什么", "帮我写个程序",
    "什么是机器学习", "怎么学Python", "推荐一本书",
    "你喜欢什么音乐", "今天心情不好", "有什么好消息",
    "你吃饭了吗", "晚安", "早上好", "周末愉快",
]

# Generate 1000 chat interactions
greetings = [
    "你好！有什么可以帮助你的吗？",
    "你好呀！今天过得怎么样？",
    "嗨！很高兴见到你！",
    "你好，请问需要什么帮助？",
]
responses = [
    "今天天气不错，适合出去走走。",
    "我是一名AI助手，可以帮你写代码、回答问题、聊天等等。",
    "Python是一门很好的编程语言，适合初学者。",
    "推荐《Python编程从入门到实践》这本书。",
    "别担心，一切都会好起来的！",
    "晚安，好梦！",
    "早上好！今天有什么计划？",
    "周末愉快！好好放松一下。",
    "机器学习是人工智能的一个重要分支。",
    "好的，我来帮你写这个程序。",
    "这个问题很有趣，让我想想。",
    "谢谢你的提问！",
]

conversations = []
for i in range(2000):
    q = random.choice(topics)
    a = random.choice(responses)
    conversations.append(f"用户: {q}\n助手: {a}\n")

with open(CHINESE_DIR / "chats_cn.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(conversations))
print(f"  [OK] chats_cn.txt ({len(conversations)} turns)")

# Generate Chinese tech explanations
tech_topics = [
    ("什么是变量", "变量是用来存储数据的容器，在Python中你可以这样使用：x = 10"),
    ("什么是函数", "函数是一段可重复使用的代码块，用def关键字定义"),
    ("什么是列表", "列表是Python中常用的数据结构，用[]表示，可以存储多个元素"),
    ("什么是循环", "循环让你重复执行一段代码，for和while是常用的循环语句"),
    ("什么是类", "类是面向对象编程的基础，用class关键字定义"),
    ("什么是异常", "异常是程序运行时发生的错误，用try/except来捕获"),
    ("什么是模块", "模块是包含Python代码的文件，用import来导入"),
    ("什么是递归", "递归是函数调用自身的编程技巧"),
    ("什么是装饰器", "装饰器是一种修改函数功能的特殊语法"),
    ("什么是生成器", "生成器用yield关键字，可以逐个产生值"),
]

tech_explanations = []
for q, a in tech_topics:
    tech_explanations.append(f"问: {q}\n答: {a}\n")
    for _ in range(50):
        tech_explanations.append(f"问: {q}？举个例子\n答: {a}。例如：\n```python\n# 这是一个示例\npass\n```\n")

with open(CHINESE_DIR / "tech_cn.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(tech_explanations))
print(f"  [OK] tech_cn.txt ({len(tech_explanations)} entries)")

# Generate code-with-Chinese-comment examples
code_cn = []
for i in range(500):
    code_cn.append(f"""# 示例{i}: Python程序
# 这是一个计算斐波那契数列的程序
def fibonacci(n):
    '''计算斐波那契数列的第n项'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 测试
result = fibonacci(10)
print(f"结果是: {{result}}")
""")

with open(CHINESE_DIR / "code_cn.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(code_cn))
print(f"  [OK] code_cn.txt ({len(code_cn)} examples)")

# Download Chinese Wikipedia sample
print("  Downloading Chinese text sample...")
wiki_url = "https://raw.githubusercontent.com/brightmart/nlp_chinese_corpus/master/weibo/weibo_100k.txt"
try:
    urllib.request.urlretrieve(wiki_url, CHINESE_DIR / "weibo_cn.txt")
    size = os.path.getsize(CHINESE_DIR / "weibo_cn.txt")
    print(f"  [OK] weibo_cn.txt ({size/1024:.0f} KB)")
except:
    print(f"  [FAIL] Could not download Chinese corpus")


print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
total = 0
py_total = 0
for d in Path(DATA_DIR).iterdir():
    if d.is_dir():
        files = list(d.rglob("*.*"))
        total += len(files)
        py_total += len(list(d.rglob("*.py")))
print(f"Total directories: {len(list(Path(DATA_DIR).iterdir()))}")
print(f"Total files: {total}")
print(f"Python files: {py_total}")
print(f"[OK] Done!")
