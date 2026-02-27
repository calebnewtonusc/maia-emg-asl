#!/usr/bin/env python3
"""
Download WLASL (World-Level American Sign Language) dataset.

WLASL contains 21,083 videos of 2,000 ASL word signs from 119 signers.
We use it for cross-modal pre-training: MediaPipe landmarks -> EMG embedding.

Usage:
    # Download first 100 words
    python scripts/download_wlasl.py --n-words 100

    # Download all 2000 words
    python scripts/download_wlasl.py --n-words 2000

    # Extract landmarks only (if videos already downloaded)
    python scripts/download_wlasl.py --extract-landmarks-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/code/WLASL_v0.3.json"


def download_wlasl_json(dest: str = "data/wlasl/WLASL_v0.3.json") -> dict:
    """Download the WLASL metadata JSON."""
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    if not Path(dest).exists():
        print(f"Downloading WLASL metadata: {WLASL_JSON_URL}")
        urllib.request.urlretrieve(WLASL_JSON_URL, dest)
    with open(dest) as f:
        return json.load(f)


def _try_download_video(entry: dict, video_dir: Path) -> bool:
    """Try to download a single video entry. Returns True on success."""
    video_id = entry.get("video_id", "")
    url = entry.get("url", "")
    if not url or not video_id:
        return False

    dest = video_dir / f"{video_id}.mp4"
    if dest.exists():
        return True

    try:
        # Try yt-dlp first (handles YouTube, Vimeo, etc.)
        import subprocess
        result = subprocess.run(
            ["yt-dlp", "-q", "--no-warnings", "-o", str(dest), url],
            capture_output=True, timeout=60,
        )
        return dest.exists()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: direct HTTP download
    try:
        urllib.request.urlretrieve(url, str(dest))
        return dest.exists()
    except Exception:
        return False


def download_videos(
    wlasl_data: list,
    n_words: int = 100,
    video_dir: str = "data/wlasl/videos",
    n_workers: int = 4,
):
    """Download videos for the first n_words word classes."""
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    # Collect entries for first n_words classes
    entries = []
    for item in wlasl_data[:n_words]:
        word = item["gloss"]
        word_dir = video_dir / word
        word_dir.mkdir(exist_ok=True)
        for inst in item.get("instances", []):
            entries.append((inst, word_dir))

    print(f"Downloading {len(entries)} videos for {n_words} word classes...")
    success = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_try_download_video, e, d): (e, d) for e, d in entries}
        for i, fut in enumerate(as_completed(futures)):
            if fut.result():
                success += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(entries)} processed, {success} downloaded")

    print(f"Done: {success}/{len(entries)} videos downloaded")


def extract_landmarks(
    video_dir: str = "data/wlasl/videos",
    output_path: str = "data/wlasl/landmarks.npz",
):
    """Extract MediaPipe landmarks from downloaded videos."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        print("ERROR: opencv-python and mediapipe required. Run: pip install opencv-python mediapipe")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from src.data.vision_teacher import HandLandmarkExtractor

    extractor = HandLandmarkExtractor()
    video_dir = Path(video_dir)
    all_landmarks = {}  # word -> list of (N_frames, 63) arrays

    video_files = list(video_dir.rglob("*.mp4"))
    print(f"Extracting landmarks from {len(video_files)} videos...")

    for i, vf in enumerate(video_files):
        word = vf.parent.name
        cap = cv2.VideoCapture(str(vf))
        frame_lms = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            lm = extractor.extract(frame)
            if lm is not None:
                frame_lms.append(lm)
        cap.release()
        if frame_lms:
            if word not in all_landmarks:
                all_landmarks[word] = []
            all_landmarks[word].extend(frame_lms)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(video_files)} processed")

    # Save per-word prototype embeddings
    import numpy as np
    save_dict = {word: np.array(lms) for word, lms in all_landmarks.items()}
    np.savez(output_path, **save_dict)
    print(f"Saved landmarks: {output_path} ({len(save_dict)} words)")


def main():
    parser = argparse.ArgumentParser(description="Download WLASL dataset for cross-modal pre-training")
    parser.add_argument("--n-words", type=int, default=100, help="Number of word classes to download")
    parser.add_argument("--video-dir", default="data/wlasl/videos")
    parser.add_argument("--output", default="data/wlasl/landmarks.npz")
    parser.add_argument("--extract-landmarks-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    import numpy as np

    if not args.extract_landmarks_only:
        wlasl_data = download_wlasl_json()
        download_videos(wlasl_data, n_words=args.n_words, video_dir=args.video_dir, n_workers=args.workers)

    if args.extract_landmarks_only or True:
        extract_landmarks(video_dir=args.video_dir, output_path=args.output)


if __name__ == "__main__":
    main()
