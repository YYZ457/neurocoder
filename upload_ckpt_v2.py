"""Upload checkpoint to transfer.sh for cloud download."""
import urllib.request, os

CKPT = "D:/NeuroCoder/checkpoints/checkpoint-2500.pt"
SIZE_MB = os.path.getsize(CKPT) / (1024*1024)
print(f"Uploading {SIZE_MB:.0f} MB checkpoint to transfer.sh...")
print(f"File: {CKPT}")

# Transfer.sh has a 2GB limit for free
import subprocess
result = subprocess.run(
    ["curl", "--upload-file", CKPT, "https://transfer.sh/neurocoder-checkpoint-2500.pt"],
    capture_output=True, text=True, timeout=600
)
if result.returncode == 0:
    url = result.stdout.strip()
    print(f"\nDownload URL: {url}")
    print(f"\nOn the cloud, run:")
    print(f"wget {url} -O checkpoint-2500.pt")
else:
    print(f"Upload failed: {result.stderr}")
    print("\nTrying alternative method...")
