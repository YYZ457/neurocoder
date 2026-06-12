"""
Download another batch of Python repos to enrich training data.
Clones to the same source/ directory as existing data.
"""
import os, subprocess, time
from pathlib import Path

DATA_DIR = "sample_data/source"
os.makedirs(DATA_DIR, exist_ok=True)

def clone(repo, branch="main"):
    name = repo.split("/")[1]
    dest = Path(DATA_DIR) / name
    if dest.exists():
        print(f"  [Skip] {name} already exists")
        return True
    url = f"https://github.com/{repo}.git"
    print(f"  Cloning {repo}...", end=" ", flush=True)
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--branch", branch, url, str(dest)],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180, check=True)
        py = len(list(dest.rglob("*.py")))
        print(f"[OK] {py} .py files")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

# More high-quality Python repos - breadth over depth
REPOS = [
    # Standard library clones
    ("python/cpython", "main"),

    # Web & API
    ("bottlepy/bottle", "master"),
    ("tornadoweb/tornado", "master"),
    ("aio-libs/aiohttp", "master"),
    ("sanic-org/sanic", "main"),
    ("falconry/falcon", "master"),
    ("hugapi/hug", "master"),
    ("Kinto/kinto", "master"),
    ("taoufik07/responder", "master"),

    # ORM & Databases
    ("sqlalchemy/sqlalchemy", "main"),
    ("encode/databases", "master"),
    ("MongoEngine/mongoengine", "main"),

    # Data Science - more
    ("matplotlib/matplotlib", "main"),
    ("seaborn/seaborn", "master"),
    ("plotly/plotly.py", "master"),
    ("sympy/sympy", "master"),
    ("networkx/networkx", "main"),
    ("statsmodels/statsmodels", "main"),
    ("keras-team/keras", "master"),

    # Dev tools
    ("pypa/setuptools", "main"),
    ("pypa/virtualenv", "main"),
    ("pypa/wheel", "main"),
    ("tox-dev/tox", "master"),
    ("pre-commit/pre-commit", "main"),
    ("pytest-dev/pytest", "main"),
    ("nedbat/coveragepy", "master"),
    ("python/mypy", "master"),
    ("google/yapf", "main"),
    ("PyCQA/isort", "main"),
    ("PyCQA/flake8", "main"),
    ("PyCQA/pylint", "main"),
    ("coala/coala", "master"),

    # Config/CLI
    ("pallets/click", "main"),
    ("google/python-fire", "master"),
    ("docopt/docopt", "master"),
    ("clibs/clib", "master"),

    # Async
    ("python-trio/trio", "master"),
    ("MagicStack/uvloop", "master"),
    ("urllib3/urllib3", "main"),

    # Networking
    ("paramiko/paramiko", "main"),
    ("jonathanslenders/python-prompt-toolkit", "master"),
    ("sqlmapproject/sqlmap", "master"),

    # Media
    ("pallets/jinja", "main"),
    ("mitsuhiko/markupsafe", "main"),
    ("pallets/werkzeug", "main"),

    # Chinese/ML
    ("PaddlePaddle/PaddleOCR", "main"),
    ("PaddlePaddle/PaddleNLP", "develop"),
    ("d2l-ai/d2l-zh", "main"),  # 中文深度学习教材
    ("datawhalechina/leedl-tutorial", "main"),  # 中文ML教程

    # More Chinese
    ("jhao104/proxy_pool", "master"),
    ("WongKinYiu/yolov7", "main"),
    ("ultralytics/ultralytics", "main"),

    # Security
    ("andrew-d/pty_shell", "master"),
    ("pyca/cryptography", "main"),

    # GUI
    ("ChrisKnott/Eel", "master"),
    ("beeware/toga", "main"),

    # Interesting
    ("Textualize/rich", "master"),
    ("Textualize/textual", "main"),
    ("ManimCommunity/manim", "main"),
    ("yt-dlp/yt-dlp", "master"),
    ("nicolargo/glances", "develop"),

    # More quality
    ("celery/celery", "main"),
    ("redis/redis-py", "master"),
    ("boto/boto3", "develop"),
    ("googleapis/google-api-python-client", "main"),
    ("elastic/elasticsearch-py", "main"),
    ("scrapy/scrapy", "master"),
    ("psf/black", "main"),
    ("ipython/ipython", "main"),
    ("jupyter/jupyter_client", "main"),
    ("jupyter/notebook", "main"),
]

print(f"=" * 60)
print(f"Downloading {len(REPOS)} more Python repos...")
print(f"=" * 60)

count = 0
for repo, branch in REPOS:
    if clone(repo, branch):
        count += 1

# Count final
all_py = list(Path(DATA_DIR).rglob("*.py"))
all_txt = list(Path(DATA_DIR).rglob("*.txt"))
print(f"\n{'='*60}")
print(f"Downloaded {count} new repos")
print(f"Total now: {len(all_py)} .py + {len(all_txt)} .txt files")
total_mb = sum(f.stat().st_size for f in all_py) / (1024*1024)
print(f"Total .py size: {total_mb:.1f} MB")
print(f"{'='*60}")
