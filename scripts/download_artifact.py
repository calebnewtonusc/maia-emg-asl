#!/usr/bin/env python3
"""
Download model artifacts from Cloudflare R2.

Usage:
    python scripts/download_artifact.py --model asl_emg_classifier.onnx
    python scripts/download_artifact.py --all-latest
    python scripts/download_artifact.py --list
    python scripts/download_artifact.py --url-only asl_emg_classifier.onnx
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


def _get_client():
    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3")
        sys.exit(1)

    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket = os.environ.get("R2_BUCKET_NAME", "maia-emg-asl")

    if not all([account_id, access_key, secret_key]):
        print("ERROR: Missing R2 credentials.")
        sys.exit(1)

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )
    return client, bucket


def _verify_sha256(path: str, expected: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def download_model(file_name: str, tag: str = "latest", dest_dir: str = "models"):
    """Download a model file from R2."""
    client, bucket = _get_client()
    key = f"models/{tag}/{file_name}"
    sha_key = f"models/{tag}/{file_name}.sha256"

    dest = Path(dest_dir) / file_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {bucket}/{key} → {dest}")
    client.download_file(bucket, key, str(dest))

    # Verify SHA256
    try:
        sha_resp = client.get_object(Bucket=bucket, Key=sha_key)
        expected_sha = sha_resp["Body"].read().decode().strip()
        if _verify_sha256(str(dest), expected_sha):
            print(f"SHA256 verified: {expected_sha[:16]}...")
        else:
            print("WARNING: SHA256 mismatch! File may be corrupted.")
    except Exception:
        print("Note: SHA256 sidecar not found, skipping verification")

    print(f"Saved: {dest}")
    return str(dest)


def download_all_latest(dest_dir: str = "models"):
    """Download all files from the latest/ prefix."""
    client, bucket = _get_client()
    resp = client.list_objects_v2(Bucket=bucket, Prefix="latest/")
    objects = [o for o in resp.get("Contents", []) if not o["Key"].endswith(".sha256")]

    if not objects:
        print("No artifacts in latest/")
        return

    for obj in objects:
        file_name = Path(obj["Key"]).name
        if file_name == "manifest.json":
            continue
        dest = Path(dest_dir) / file_name
        print(f"Downloading {obj['Key']} → {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, obj["Key"], str(dest))
        print(f"  Saved: {dest}")


def list_artifacts():
    client, bucket = _get_client()
    resp = client.list_objects_v2(Bucket=bucket)
    objects = [o for o in resp.get("Contents", []) if not o["Key"].endswith(".sha256")]
    if not objects:
        print("No artifacts found.")
        return
    print(f"\n{'Key':<60} {'Size':>10}")
    print("-" * 72)
    for obj in sorted(objects, key=lambda x: x["Key"]):
        print(f"{obj['Key']:<60} {obj['Size']:>10,}")


def get_public_url(file_name: str) -> str:
    r2_public_url = os.environ.get("R2_PUBLIC_URL", "")
    if not r2_public_url:
        return f"(Set R2_PUBLIC_URL env var to get public URL for {file_name})"
    return f"{r2_public_url.rstrip('/')}/latest/{file_name}"


def main():
    parser = argparse.ArgumentParser(description="Download model artifacts from Cloudflare R2")
    parser.add_argument("--model", help="Model filename to download (e.g. asl_emg_classifier.onnx)")
    parser.add_argument("--tag", default="latest", help="Version tag (default: latest)")
    parser.add_argument("--dest", default="models", help="Destination directory")
    parser.add_argument("--all-latest", action="store_true", help="Download all latest/ artifacts")
    parser.add_argument("--list", action="store_true", help="List all artifacts")
    parser.add_argument("--url-only", metavar="FILE", help="Print public URL without downloading")
    args = parser.parse_args()

    if args.list:
        list_artifacts()
    elif args.all_latest:
        download_all_latest(args.dest)
    elif args.url_only:
        print(get_public_url(args.url_only))
    elif args.model:
        download_model(args.model, tag=args.tag, dest_dir=args.dest)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
