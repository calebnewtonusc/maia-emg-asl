#!/usr/bin/env python3
"""
Upload trained model artifacts to Cloudflare R2.

Usage:
    python scripts/upload_artifact.py --file models/asl_emg_classifier.onnx --set-latest
    python scripts/upload_artifact.py --file models/asl_emg_classifier.onnx --tag v1.2.0
    python scripts/upload_artifact.py --list
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def _get_client():
    """Create boto3 S3 client configured for Cloudflare R2."""
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
        print("ERROR: Missing R2 credentials. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY")
        sys.exit(1)

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )
    return client, bucket


def _sha256(path: str) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_file(file_path: str, tag: str = "latest", set_latest: bool = False):
    """Upload a model file to R2."""
    client, bucket = _get_client()
    path = Path(file_path)
    if not path.exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    # Compute SHA256
    sha = _sha256(file_path)
    print(f"SHA256: {sha}")

    # Upload to tagged path
    key = f"models/{tag}/{path.name}"
    print(f"Uploading {path.name} ({path.stat().st_size / 1e6:.1f} MB) → {bucket}/{key}")
    client.upload_file(
        file_path,
        bucket,
        key,
        ExtraArgs={"Metadata": {"sha256": sha, "tag": tag}},
    )
    print(f"Uploaded: {key}")

    # Upload SHA256 sidecar
    sha_key = f"models/{tag}/{path.name}.sha256"
    client.put_object(Bucket=bucket, Key=sha_key, Body=sha.encode())
    print(f"Sidecar:  {sha_key}")

    if set_latest or tag == "latest":
        # Also update latest/ pointer
        latest_key = f"latest/{path.name}"
        print(f"Updating latest: {latest_key}")
        client.upload_file(
            file_path,
            bucket,
            latest_key,
            ExtraArgs={"Metadata": {"sha256": sha, "source_tag": tag}},
        )
        client.put_object(Bucket=bucket, Key=f"latest/{path.name}.sha256", Body=sha.encode())
        # Update manifest
        manifest = {"tag": tag, "file": path.name, "sha256": sha}
        client.put_object(
            Bucket=bucket, Key="latest/manifest.json",
            Body=json.dumps(manifest, indent=2).encode(),
            ContentType="application/json",
        )
        print("Latest manifest updated")

    print("Done!")
    return key


def list_artifacts():
    """List all uploaded artifacts."""
    client, bucket = _get_client()
    resp = client.list_objects_v2(Bucket=bucket, Prefix="models/")
    objects = resp.get("Contents", [])
    if not objects:
        print("No artifacts found.")
        return
    print(f"\n{'Key':<60} {'Size':>10}")
    print("-" * 72)
    for obj in sorted(objects, key=lambda x: x["Key"]):
        if not obj["Key"].endswith(".sha256"):
            print(f"{obj['Key']:<60} {obj['Size']:>10,}")


def generate_public_url(file_name: str, tag: str = "latest") -> str:
    """Generate public R2 URL (requires public bucket or custom domain)."""
    bucket = os.environ.get("R2_BUCKET_NAME", "maia-emg-asl")
    r2_public_url = os.environ.get("R2_PUBLIC_URL", "")
    if not r2_public_url:
        print("WARN: R2_PUBLIC_URL not set. Set it to your R2 custom domain.")
        return ""
    return f"{r2_public_url.rstrip('/')}/latest/{file_name}"


def main():
    parser = argparse.ArgumentParser(description="Upload model artifacts to Cloudflare R2")
    parser.add_argument("--file", help="Path to model file to upload")
    parser.add_argument("--tag", default="latest", help="Version tag (default: latest)")
    parser.add_argument("--set-latest", action="store_true", help="Also update latest/ pointer")
    parser.add_argument("--list", action="store_true", help="List all artifacts")
    parser.add_argument("--url", help="Generate public URL for a file name")
    args = parser.parse_args()

    if args.list:
        list_artifacts()
    elif args.url:
        url = generate_public_url(args.url, args.tag)
        print(url)
    elif args.file:
        upload_file(args.file, tag=args.tag, set_latest=args.set_latest)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
