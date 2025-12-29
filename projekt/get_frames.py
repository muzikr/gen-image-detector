from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from detect_face import FaceExtractor

def _select_frame_indices(total_frames: int, num_frames: int) -> list[int]:
    """Return `num_frames` evenly spaced indices within [0, total_frames)."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if total_frames <= 0:
        raise ValueError("Video has no frames")
    if num_frames == 1:
        return [0]

    last = total_frames - 1
    return [int(round(i * last / (num_frames - 1))) for i in range(num_frames)]


class FrameExtractor:
    """Sample frames from a video and optionally save them."""

    def __init__(self, save_dir: str | Path | None = None, prefix: str = "frame") -> None:
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.prefix = prefix
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.face_extractor = FaceExtractor()

    def extract(self, video_path: str | Path, num_frames: int) -> list[np.ndarray]:
        """Return `num_frames` evenly spaced frames as a list of BGR numpy arrays.

        If `save_dir` was provided at construction, frames are also saved to disk
        using the given prefix. Raises if the video cannot be opened or frames
        cannot be read.
        """
        video_path = Path(video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        targets = set(_select_frame_indices(total_frames, num_frames))

        frames: list[np.ndarray] = []
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in targets:
                frames.append(frame)
                if self.save_dir is not None:
                    out_path = self.save_dir / f"{self.prefix}_{frame_idx:06d}.jpg"
                    face = self.face_extractor.extract_face(frame)
                    frame = cv2.resize(face, (224, 224))
                    cv2.imwrite(str(out_path), frame)
            frame_idx += 1

        cap.release()

        if len(frames) != len(targets):
            raise RuntimeError(
                f"Extracted {len(frames)} of {len(targets)} frames; video may be short or unreadable.")

        return frames
    

if __name__ == "__main__":
    video_path = 'C:\\Users\\marti\\Documents\\projekt_muj_a_risankovo_DL\\data\\archive (3)\\Celeb-real\\id0_0000.mp4'  # Replace with your video path
    extractor = FrameExtractor(save_dir='extracted_frames', prefix='video_frame')
    frames = extractor.extract(video_path, num_frames=5)
    for i,frame in enumerate(frames):
        cv2.imwrite(f'test_out\\frame_output_{i}.jpg', frame)  # Example of saving a frame
    print(f"Extracted {len(frames)} frames.")