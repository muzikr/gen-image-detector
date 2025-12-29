import cv2
import mediapipe as mp

class FaceExtractor:

    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a face detector instance with the image mode:
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path='blaze_face_short_range.tflite'),
            running_mode=VisionRunningMode.IMAGE)
        
        self.detector = FaceDetector.create_from_options(options)

    def get_face_bbox(self, image) -> tuple[int, int, int, int] | None:
        # Load the input image:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Prepare the input image for the detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        # Perform face detection:
        detection_result = self.detector.detect(mp_image)
        # Draw bounding boxes around detected faces:
        best_face_bbox = None
        best_face_confidence = 0
        for face in detection_result.detections:
            if face.categories[0].score > best_face_confidence:
                best_face_confidence = face.categories[0].score
                bbox = face.bounding_box
            
                x1 = int(bbox.origin_x)
                y1 = int(bbox.origin_y)
                x2 = int(bbox.origin_x + bbox.width)
                y2 = int(bbox.origin_y + bbox.height)
                best_face_bbox = (x1, y1, x2, y2)

        return best_face_bbox

    def extract_face(self, image, output_path = None) -> None:
        bbox = self.get_face_bbox(image)

        if bbox:
            x1, y1, x2, y2 = bbox
            face_image = image[y1:y2, x1:x2]
            if output_path:
                cv2.imwrite(output_path, face_image)
            return face_image
        else:
            print("No face detected.")
            return image
        

if __name__ == "__main__":
    face_extractor = FaceExtractor()
    input_image_path = 'arnold.jpg'  # Replace with your input image path
    output_image_path = 'extracted_face.jpg'  # Replace with your desired output path
    input_image = cv2.imread(input_image_path)
    face_extractor.extract_face(input_image, output_image_path)