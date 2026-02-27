#!/usr/bin/env python3
"""
Auto-label a recorded EMG session using the vision teacher.

Requires a trained pose classifier at models/pose_classifier.pkl.
If none exists, it generates a synthetic one for testing.

Usage:
    # Live mode (webcam + synthetic EMG)
    python scripts/auto_label_session.py --live

    # Offline mode (video file + EMG npz)
    python scripts/auto_label_session.py --video path/to/video.mp4 --emg path/to/emg.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.constants import ASL_CLASSES, SAMPLE_RATE
from src.data.vision_teacher import (
    HandLandmarkExtractor, SimpleASLPoseClassifier, VisionTeacher, sync_labels_to_emg
)


def _build_synthetic_classifier() -> SimpleASLPoseClassifier:
    """Build a simple synthetic pose classifier for testing."""
    rng = np.random.default_rng(42)
    clf = SimpleASLPoseClassifier()
    # Synthetic landmark prototypes (one per class)
    X = rng.normal(0, 0.3, (26 * 20, 63)).astype(np.float32)
    y = np.repeat(np.arange(26), 20)
    clf.fit(X, y)
    return clf


def main():
    parser = argparse.ArgumentParser(description="Auto-label EMG session using vision teacher")
    parser.add_argument("--live", action="store_true", help="Live webcam mode")
    parser.add_argument("--video", help="Path to video file (offline mode)")
    parser.add_argument("--emg", help="Path to .npz EMG file (offline mode)")
    parser.add_argument("--classifier", default="models/pose_classifier.pkl")
    parser.add_argument("--output", help="Output .npz path (default: input_labeled.npz)")
    parser.add_argument("--confidence", type=float, default=0.7)
    args = parser.parse_args()

    # Load or create classifier
    clf_path = Path(args.classifier)
    if clf_path.exists():
        print(f"Loading pose classifier: {clf_path}")
        clf = SimpleASLPoseClassifier.load(str(clf_path))
    else:
        print(f"No classifier at {clf_path} -- using synthetic classifier")
        clf = _build_synthetic_classifier()

    teacher = VisionTeacher(clf, confidence_threshold=args.confidence)

    if args.live:
        print("Live mode: webcam (press Q to stop)")
        try:
            import cv2
        except ImportError:
            print("ERROR: opencv-python required. Run: pip install opencv-python")
            sys.exit(1)
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        results = []
        for ts, cls_idx, conf in teacher.stream_labels(cap, fps=fps):
            if cls_idx is not None:
                print(f"t={ts:.1f}s  {ASL_CLASSES[cls_idx]}  conf={conf:.3f}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif args.video and args.emg:
        print(f"Labeling {args.emg} using {args.video}")
        vision_labels = teacher.label_video_file(args.video)

        data = np.load(args.emg)
        emg = data["emg"]
        n_samples = emg.shape[0]
        emg_ts = np.arange(n_samples) / SAMPLE_RATE
        label_array = sync_labels_to_emg(vision_labels, emg_ts)

        n_labeled = (label_array >= 0).sum()
        print(f"Labeled {n_labeled}/{n_samples} samples ({n_labeled/n_samples*100:.1f}%)")

        out_path = args.output or args.emg.replace(".npz", "_labeled.npz")
        np.savez(out_path, emg=emg, labels=label_array)
        print(f"Saved: {out_path}")
    else:
        print("Provide --live or --video + --emg")
        sys.exit(1)


if __name__ == "__main__":
    main()
