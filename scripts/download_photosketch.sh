#!/usr/bin/env bash
set -euo pipefail

mkdir -p third_party data/raw
if [[ ! -d third_party/PhotoSketch/.git ]]; then
    git clone https://github.com/mtli/PhotoSketch.git third_party/PhotoSketch
fi

output="data/raw/photosketch_latest_net_G.pth"
if [[ -s "$output" ]]; then
    echo "[skip] $output already exists"
    exit 0
fi

# The original project distributes this checkpoint through Google Drive.  The
# API blob below is a public mirror of latest_net_G.pth and is used because
# drive.google.com is not reachable on the training server.
blob_sha="20f46fe1c2fba348ce2e3776ddd774039be86128"
temporary=$(mktemp --suffix=.json)
trap 'rm -f "$temporary"' EXIT
wget -q "https://api.github.com/repos/HersonLi/PhotoSketch/git/blobs/$blob_sha" -O "$temporary"
python -c "import base64,json,pathlib; d=json.load(open('$temporary')); pathlib.Path('$output').write_bytes(base64.b64decode(d['content']))"

actual_sha1=$(sha1sum "$output" | cut -d' ' -f1)
expected_sha1="5968e8f007c650008a265c11f2d2a3887e5840d4"
if [[ "$actual_sha1" != "$expected_sha1" ]]; then
    echo "PhotoSketch checkpoint checksum mismatch: $actual_sha1" >&2
    exit 1
fi
echo "[done] PhotoSketch generator: $output"
