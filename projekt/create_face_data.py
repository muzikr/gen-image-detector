import torch
import os
from tqdm import tqdm
import cv2
from get_frames import FrameExtractor
import pandas as pd
from dataset import FrameDataset

def get_list_of_videos() -> list[tuple[str, int]]:
    """Save face data from videos into a list of (video_path, label) tuples."""
    data = []
    for folder in [
        ("data\\archive (3)\\Celeb-real",0,"celebreal"),
        ("data\\archive (3)\\Celeb-synthesis",1,"celebsyn"),
        ("data\\archive (3)\\Youtube-real",0,"ytreal"),]:
        for file in tqdm(os.listdir(folder[0])):
            if file.endswith('.mp4'):
                video_path = os.path.join(folder[0], file)
                data.append((video_path, folder[1], folder[2]))
    print(data[0])
    return data

def get_face_data(data: list[tuple[str, int, str]], output_dir: str) -> None:
    """Save the face data list to a file using torch."""
    extractor = FrameExtractor(save_dir=None)

    os.makedirs(output_dir, exist_ok=True)

    face_data = []
    for video_path, label, folder_name in tqdm(data):
        try:
            frames = extractor.extract(video_path, num_frames=5)
            for i, frame in enumerate(frames):
                face = extractor.face_extractor.extract_face(frame)
                if face is None:
                    continue
                face = cv2.resize(face, (256, 256))
                # build a filename per extracted face
                fname = f"{folder_name}_{os.path.splitext(os.path.basename(video_path))[0]}_{i}.jpg"
                output_path = os.path.join(output_dir, fname)
                cv2.imwrite(output_path, face)
                face_data.append((output_path, label, folder_name, i, video_path))
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(face_data, output_path)
    print(f"Saved face data to {output_path}")  
    pd.DataFrame(face_data, columns=[
        'image_path',
        'label', 
        'folder_name', 
        'frame_index', 
        'video_path']
    ).to_csv(os.path.join(output_dir, 'face_data.csv'), index=False)  

if __name__ == "__main__":
    data = get_list_of_videos()
    output_dir = "faces_data"
    get_face_data(data, output_dir)
    dataset = FrameDataset()
    print(f"Dataset size: {len(dataset)}")
    image, label = dataset[0]
    print(f"First image shape: {image.shape}, label: {label}")
    print(image)
    cv2.imwrite('test_face.jpg', image)
