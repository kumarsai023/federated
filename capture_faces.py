import cv2
import os
import numpy as np
from tqdm import tqdm

class FaceCapture:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.required_images = 250
        # Increased minimum face size
        self.min_face_width = 200
        self.min_face_height = 200
        # Adjusted quality thresholds
        self.blur_threshold = 80  # Reduced blur threshold
        self.brightness_threshold = 30  # Reduced brightness threshold
        
    def create_directory(self, student_id):
        """Create directory for storing student images"""
        self.image_dir = f'dataset/student_{student_id}'
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        return self.image_dir

    def check_face_quality(self, face_img):
        """Check if face image meets quality standards"""
        # Check face size
        height, width = face_img.shape[:2]
        if width < self.min_face_width or height < self.min_face_height:
            return False, "Move closer"

        # Check for blur
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.blur_threshold:
            return False, "Hold still"

        # Check brightness
        brightness = np.mean(gray)
        if brightness < self.brightness_threshold:
            return False, "Need more light"

        return True, "Good"

    def preprocess_face(self, face_img):
        """Preprocess the facial image"""
        # Simple resize to 224x224
        resized = cv2.resize(face_img, (224, 224))
        # Convert to RGB
        final_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return final_img

    def capture_faces(self, student_id):
        """Capture and save facial images"""
        self.create_directory(student_id)
        cap = cv2.VideoCapture(0)
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        count = 0
        pbar = tqdm(total=self.required_images, desc='Capturing Faces')
        
        frame_count = 0
        frames_between_captures = 8
        
        while count < self.required_images:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Show original frame with guidelines
            display_frame = frame.copy()
            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            
            # Draw guide square (larger than before)
            square_size = 300  # Increased size
            cv2.rectangle(display_frame, 
                         (center_x - square_size//2, center_y - square_size//2),
                         (center_x + square_size//2, center_y + square_size//2),
                         (0, 255, 0), 2)
            
            if frame_count % frames_between_captures != 0:
                cv2.imshow('Face Capture - Align face within square', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,  # Reduced for better detection
                minSize=(self.min_face_width, self.min_face_height)
            )
            
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                quality_ok, message = self.check_face_quality(face_img)
                
                if quality_ok:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, message, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    processed_face = self.preprocess_face(face_img)
                    image_path = os.path.join(self.image_dir, f'face_{count}.jpg')
                    cv2.imwrite(image_path, cv2.cvtColor(processed_face, cv2.COLOR_RGB2BGR))
                    
                    count += 1
                    pbar.update(1)
                else:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(display_frame, message, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                if count >= self.required_images:
                    break
            
            # Display guide text
            cv2.putText(display_frame, "Position face within square", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Captured: {count}/{self.required_images}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Face Capture - Align face within square', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        pbar.close()
        cap.release()
        cv2.destroyAllWindows()
        return count

    def verify_dataset(self):
        """Verify the captured images"""
        image_files = os.listdir(self.image_dir)
        print(f"Total images captured: {len(image_files)}")
        return len(image_files)

if __name__ == "__main__":
    face_capture = FaceCapture()
    student_id = input("Enter student ID: ")
    
    print("\nStarting face capture system...")
    print("\nGuidelines:")
    print("1. Position your face within the green square")
    print("2. Maintain good lighting")
    print("3. Move slowly to capture different angles")
    print("4. Keep eyes open and maintain neutral expression")
    print("5. Press 'q' to quit\n")
    
    captured = face_capture.capture_faces(student_id)
    
    if captured == 250:
        print("\nSuccessfully captured all required images!")
    else:
        print(f"\nCaptured {captured} images out of 250")
    
    face_capture.verify_dataset() 