import cv2
import face_recognition
import numpy as np
from PIL import Image
import os

class RealtimeFaceRecognition:
    def __init__(self):
        print("Initializing Face Recognition System...")
        
        # Load the known face image and encode it
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Load images from dataset
        dataset_path = "dataset/student_1"
        print("Loading known faces from dataset...")
        
        # Take first 5 images for encoding (for speed)
        image_files = os.listdir(dataset_path)[:5]
        
        for image_file in image_files:
            image_path = os.path.join(dataset_path, image_file)
            image = face_recognition.load_image_file(image_path)
            
            # Get face encodings from the image
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append("Student 1")
                
        print(f"Loaded {len(self.known_face_encodings)} face encodings")
        
        # Initialize variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        
        # Set confidence threshold
        self.confidence_threshold = 0.5
        print(f"Initial confidence threshold: {self.confidence_threshold}")

    def run_detection(self):
        print("Starting real-time detection...")
        print("Controls:")
        print("- Press '+' to increase confidence threshold")
        print("- Press '-' to decrease confidence threshold")
        print("- Press 'q' to quit")
        
        # Get a reference to webcam
        video_capture = cv2.VideoCapture(0)

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Only process every other frame of video to save time
            if self.process_this_frame:
                # Resize frame for faster face recognition
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color to RGB color
                rgb_small_frame = small_frame[:, :, ::-1]

                # Find all the faces and face encodings in the current frame
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known faces
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, 
                        face_encoding,
                        tolerance=self.confidence_threshold
                    )
                    name = "Unknown"

                    if True in matches:
                        # Calculate face distance (lower is better)
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        confidence = 1 - face_distances[best_match_index]
                        
                        if matches[best_match_index] and confidence > self.confidence_threshold:
                            name = f"{self.known_face_names[best_match_index]} ({confidence:.2f})"

                    self.face_names.append(name)

            self.process_this_frame = not self.process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw box and label
                if "Unknown" in name:
                    color = (0, 0, 255)  # Red for unknown
                else:
                    color = (0, 255, 0)  # Green for recognized

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # Display threshold
            cv2.putText(frame, f"Threshold: {self.confidence_threshold:.2f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display the resulting image
            cv2.imshow('Face Recognition', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):
                self.confidence_threshold = min(1.0, self.confidence_threshold + 0.05)
                print(f"Threshold increased to: {self.confidence_threshold:.2f}")
            elif key == ord('-'):
                self.confidence_threshold = max(0.0, self.confidence_threshold - 0.05)
                print(f"Threshold decreased to: {self.confidence_threshold:.2f}")

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_recognizer = RealtimeFaceRecognition()
    face_recognizer.run_detection() 