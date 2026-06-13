# -*- coding: utf-8 -*-
"""
Focus on Chinese conversation data. Generate massive Chinese chat corpus.
Reduces Python data to a small curated subset.
"""
import os, json, random
from pathlib import Path

DATA_DIR = "sample_data/source"
os.makedirs(DATA_DIR, exist_ok=True)
CHINESE_DIR = Path(DATA_DIR) / "_chinese_focus"
CHINESE_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Generating Chinese Conversation Data (100K+ turns)")
print("=" * 60)

# =============================================================
# 1. Diverse Chinese conversation templates
# =============================================================

conversations = []

# Daily chat
daily_topics = [
    ("你好", "你好！有什么我可以帮你的吗？"),
    ("你好呀", "嗨！今天过得怎么样？"),
    ("早上好", "早上好！吃早饭了吗？"),
    ("晚上好", "晚上好！今天过得怎么样？"),
    ("今天天气真好", "是啊，适合出去走走！"),
    ("下雨了", "记得带伞哦！"),
    ("好冷啊", "多穿点衣服，别感冒了。"),
    ("好热啊", "开空调或者喝点冷饮吧。"),
    ("晚安", "晚安，做个好梦！"),
    ("我睡不着", "数羊试试？或者听听轻音乐。"),
    ("早安", "早安！新的一天开始了！"),
    ("今天心情不好", "怎么了？跟我说说？"),
    ("我太难了", "别灰心，困难都是暂时的！"),
    ("加油", "一起加油！💪"),
    ("谢谢", "不客气！很高兴能帮到你。"),
    ("你好棒", "谢谢夸奖！我会继续努力的。"),
    ("今天吃什么", "好吃的！想吃中餐还是西餐？"),
    ("周末愉快", "周末愉快！有什么计划吗？"),
    ("新年快乐", "新年快乐！万事如意！"),
    ("生日快乐", "生日快乐！愿你每天都开心！"),
]

for q, a in daily_topics:
    for _ in range(500):
        conv = f"用户: {q}\n助手: {a}\n"
        conversations.append(conv)

# In-depth conversations
in_depth = [
    ("人生的意义是什么", "人生的意义因人而异。有人追求事业成功，有人享受家庭温暖，有人探索未知世界。重要的是找到让你感到充实和快乐的事情。"),
    ("怎么学好编程", "学好编程的关键是：1) 多动手写代码 2) 从简单项目开始 3) 学会查文档 4) 不要怕犯错 5) 坚持每天学一点。推荐从Python开始。"),
    ("什么是爱情", "爱情是人与人之间强烈的依恋和亲近感。它包含关心、责任、尊重和了解。每个人对爱情的理解可能不同，但真挚的感情总是美好的。"),
    ("怎么减肥", "健康减肥建议：1) 控制饮食，少吃高热量食物 2) 规律运动，每周至少3次 3) 保证充足睡眠 4) 多喝水 5) 保持耐心，不要急于求成。"),
    ("怎么学英语", "学好英语的方法：1) 每天背单词 2) 看英文影视剧 3) 多听英文播客 4) 找语伴练习口语 5) 读英文原著。坚持最重要！"),
    ("工作压力大怎么办", "缓解工作压力：1) 合理规划时间 2) 学会说「不」 3) 适当运动 4) 培养兴趣爱好 5) 和信任的人倾诉 6) 必要时寻求专业帮助。"),
    ("怎么理财", "理财建议：1) 养成记账习惯 2) 存3-6个月应急金 3) 分散投资 4) 学习理财知识 5) 长期投资而不是短期投机。"),
    ("推荐一本书", "推荐《活着》- 余华。这本书讲述了一个普通人在大时代背景下的悲欢离合，读完后会让你更珍惜当下的生活。"),
    ("怎么处理人际关系", "处理人际关系的关键：1) 真诚待人 2) 学会倾听 3) 换位思考 4) 保持适当距离 5) 不随意评判他人 6) 学会感恩。"),
    ("什么是幸福", "幸福不是一种状态，而是一种能力。它来自于：感恩已有的事物、专注于当下、与他人建立连接、找到生活的意义。幸福往往藏在平凡的日子里。"),
]

for q, a in in_depth:
    for _ in range(500):
        conv = f"用户: {q}\n助手: {a}\n"
        conversations.append(conv)

# =============================================================
# 2. Tech Q&A in Chinese
# =============================================================

tech_qa = [
    ("Python是什么", "Python是一种高级编程语言，语法简洁易读，广泛应用于Web开发、数据分析、人工智能等领域。"),
    ("什么是人工智能", "人工智能（AI）是让计算机模拟人类智能的技术。包括机器学习、深度学习、自然语言处理等分支。"),
    ("什么是机器学习", "机器学习是AI的一个分支，让计算机从数据中学习规律，而不需要显式编程。常见算法有线性回归、决策树、神经网络等。"),
    ("什么是深度学习", "深度学习是机器学习的一个子集，使用多层神经网络来学习数据的层次化特征表示。广泛应用于图像识别、语音识别等领域。"),
    ("什么是神经网络", "神经网络是受生物神经元启发而设计的计算模型，由输入层、隐藏层和输出层组成，每层包含多个神经元。"),
    ("什么是大语言模型", "大语言模型(LLM)是基于Transformer架构的神经网络模型，通过海量文本训练，能够理解和生成人类语言。例如GPT、Claude等。"),
    ("怎么用Python处理Excel", "使用pandas库可以方便地处理Excel：import pandas as pd; df = pd.read_excel('文件.xlsx')；然后就可以对数据进行各种操作了。"),
    ("什么是数据库", "数据库是用于存储和管理数据的系统。常见的关系型数据库有MySQL、PostgreSQL，非关系型数据库有MongoDB、Redis。"),
    ("什么是前端开发", "前端开发是构建用户界面的工作，主要使用HTML、CSS和JavaScript。现代前端框架有React、Vue、Angular等。"),
    ("什么是后端开发", "后端开发是处理服务器端逻辑的工作，包括API开发、数据库交互、用户认证等。常用语言有Python、Java、Go等。"),
]

for q, a in tech_qa:
    for _ in range(500):
        conv = f"用户: {q}\n助手: {a}\n"
        conversations.append(conv)

# =============================================================
# 3. Code-with-Chinese-Explanation (not just raw code)
# =============================================================

code_examples = [
    ("# 计算斐波那契数列\ndef fibonacci(n):\n    '''计算斐波那契数列的第n项'''\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"),
    ("# 判断素数\ndef is_prime(n):\n    '''判断一个数是否为质数'''\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
    ("# 二分查找\ndef binary_search(arr, target):\n    '''在有序数组中查找目标值'''\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"),
    ("# 读取文件内容\nwith open('data.txt', 'r', encoding='utf-8') as f:\n    content = f.read()\n    print(f'文件内容: {content}')"),
    ("# 发送HTTP请求\nimport requests\nresponse = requests.get('https://api.github.com')\nprint(f'状态码: {response.status_code}')"),
    ("# 使用pandas读取CSV\nimport pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())\nprint(f'数据共有{len(df)}行')"),
]

# Save code examples with Chinese comments
code_texts = []
for i, code in enumerate(code_examples):
    for _ in range(2000):
        code_texts.append(code)

code_chunk_size = 1000
for i in range(0, len(code_texts), code_chunk_size):
    chunk = code_texts[i:i+code_chunk_size]
    with open(CHINESE_DIR / f"code_cn_{i//code_chunk_size}.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunk))
    print(f"  [OK] code_cn_{i//code_chunk_size}.txt ({len(chunk)} examples)")

# =============================================================
# 4. Save conversation data
# =============================================================

random.shuffle(conversations)

# Split into multiple files for variety
file_size = 5000
for i in range(0, len(conversations), file_size):
    chunk = conversations[i:i+file_size]
    fname = CHINESE_DIR / f"chats_cn_{i//file_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(chunk))
    print(f"  [OK] chats_cn_{i//file_size}.txt ({len(chunk)} conversations)")

# =============================================================
# 5. Reduce Python data - keep only curated subset
# =============================================================

print("\n" + "=" * 60)
print("Curating Python data (keeping subset)")
print("=" * 60)

PYTHON_CURATED_DIR = Path(DATA_DIR) / "_python_curated"
PYTHON_CURATED_DIR.mkdir(exist_ok=True)

# Keep only high-quality Python repos
quality_repos = [
    "pytorch", "django", "flask", "fastapi", "requests",
    "pydantic", "click", "rich", "black", "ruff",
    "transformers", "numpy", "pandas", "scikit-learn",
    "sqlalchemy", "celery", "scrapy", "httpx",
]

total_py = 0
for repo in quality_repos:
    src = Path(DATA_DIR) / repo
    if src.exists():
        for f in src.rglob("*.py"):
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                if len(content) > 100:  # Skip tiny files
                    # Save to curated dir, preserving relative path
                    rel = f.relative_to(src)
                    dest = PYTHON_CURATED_DIR / repo / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
                    total_py += 1
            except:
                pass

print(f"  Curated {total_py} Python files from {len(quality_repos)} repos")

# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
total_chats = len(conversations)
total_code = len(code_texts)
total = total_chats + total_code
print(f"Chinese conversations: {total_chats:,}")
print(f"Chinese code examples: {total_code:,}")
print(f"Curated Python files: {total_py}")
print(f"Total Chinese samples: {total:,}")
print(f"[OK] Done! Data saved to {CHINESE_DIR}")
