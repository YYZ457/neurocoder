"""
Upload checkpoint as GitHub Release.
Usage: python upload_ckpt.py GITHUB_TOKEN
"""
import os, sys, json, time, mimetypes
import urllib.request
import urllib.error

TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_TOKEN", "")
if not TOKEN:
    print("Need GITHUB_TOKEN as arg or env var")
    sys.exit(1)

REPO = "YYZ457/neurocoder"
CHECKPOINT = "D:/NeuroCoder/checkpoints/latest.pt"

def gh_api(method, path, data=None, binary=False):
    url = f"https://api.github.com/repos/{REPO}/{path}"
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    if not binary:
        headers["Content-Type"] = "application/json"
        body = json.dumps(data).encode() if data else None
    else:
        headers["Content-Type"] = "application/octet-stream"
        body = data

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"HTTP {e.code}: {body}")
        return None

# 1. Create release
print("Creating Release...")
tag = f"checkpoint-2500"
release = gh_api("POST", "releases", {
    "tag_name": tag,
    "name": f"NeuroCoder Checkpoint 2500",
    "body": "Training checkpoint at 2500 steps, 197M params, trained on 47K Python files",
})
if not release:
    # Maybe already exists, try to get it
    releases = gh_api("GET", "releases")
    if releases:
        for r in releases:
            if r["tag_name"] == tag:
                release = r
                break
if not release:
    print("Failed to create release")
    sys.exit(1)

print(f"Release ID: {release['id']}")

# 2. Compress checkpoint (it's 2.3GB, upload as-is)
print(f"Uploading {CHECKPOINT}... ({os.path.getsize(CHECKPOINT) / 1024/1024:.0f} MB)")

upload_url = release["upload_url"].replace("{?name,label}", "")
import urllib.parse
upload_url += f"?name=latest.pt"

with open(CHECKPOINT, "rb") as f:
    data = f.read()

req = urllib.request.Request(
    upload_url, data=data,
    headers={
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/octet-stream",
        "Content-Length": str(len(data)),
    },
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read())
        print(f"Upload complete! Download URL: {result['browser_download_url']}")
except Exception as e:
    print(f"Upload failed: {e}")
