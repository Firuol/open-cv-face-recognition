import cv2
import os
import face_recognition
# Path to the directory containing the reference images
reference_images_path = "./images"
# Create a list of reference image filenames
reference_images = [f for f in os.listdir(reference_images_path) if f.endswith('.jpg')]
# Load the reference images and create face encodings
known_face_encodings = []
known_face_names = []
for image_name in reference_images:
    reference_image = face_recognition.load_image_file(os.path.join(reference_images_path, image_name))
    face_encoding = face_recognition.face_encodings(reference_image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(image_name)
# Initialize the webcam feed. Change the camera index if needed (0 is typically the default for the built-in webcam).
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    for (top, right, bottom, left) in face_locations:
        # Extract the face encoding for the detected face
        face_encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])[0]
        # Compare the detected face with the reference faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown Person"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        # Draw a rectangle and label on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # Display the frame
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
