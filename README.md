Facial Recognition and Security Alert System

This project implements a facial recognition system using the face_recognition library in Python. The system is capable of detecting faces from images, comparing them with a database of known faces, and sending alerts via email if a "dangerous" subject is detected. When a dangerous subject is recognized, the system triggers an alarm, displays an alert message on the screen, and sends an email with the Google Maps link to the device's geographic location.



Features

Facial Recognition: The system can recognize faces in real-time through the webcam.
Security Alert: If a dangerous face is recognized, an alert is sent via email, an alarm is triggered, and a warning message is displayed on the screen.
Geolocation: The device's geographic location is retrieved via IP and sent via email with a Google Maps link.
Multi-Screen Support: The screen resolution is automatically adjusted based on the monitor in use.
Easy Setup: Uses a .env file to securely store email credentials.




Methods Used for Facial Recognition Model

The facial recognition system uses the following methods and techniques:
Face Detection: The face_recognition library utilizes deep learning models to perform face detection. Specifically, it uses the Histogram of Oriented Gradients (HOG) method for detecting faces in images. This method is fast and accurate for most real-time face detection applications.
Face Encoding: Once faces are detected, the system extracts unique face embeddings (or face encodings) using a pre-trained ResNet model. The face encodings are vectors that represent the facial features of the subject. These vectors are then compared to known face encodings for identification.
Face Comparison: The system compares the extracted face encoding with known face encodings using the Euclidean distance metric. If the distance between the two encodings is below a certain threshold (e.g., 0.55), the face is considered a match.
Threshold for Recognition: The system uses a precision threshold to decide when a match is valid. If the distance between the current face encoding and the known encoding is smaller than the threshold, the match is considered accurate.



Prerequisites

To run this project, you need Python 3.6+ and the following libraries:
opencv-python – For video processing and face detection.
face_recognition – For facial recognition.
screeninfo – To get screen resolution.
geocoder – To get geographic location via IP.
smtplib – To send emails via SMTP.
python-dotenv – To load environment variables from the .env file.





Usage

The system will automatically start detecting faces via the webcam.
When a face is recognized, it will be compared to the known faces stored in the database.
If the face is "dangerous" (marked in the text file as such), an alarm will sound, a warning message will be displayed on the screen, and an email will be sent with the recognized face's information.
The email will include a Google Maps link with the device's geographic location.
Additional Features
Recognition Precision: The system uses a precision threshold (precision_threshold = 0.55), which can be adjusted to increase or decrease the sensitivity of facial recognition.
Multi-Screen Support: The webcam window resolution will automatically adapt to the monitor's resolution.
Contributing
If you'd like to contribute to this project, feel free to fork it and submit a pull request. If you encounter any bugs or have suggestions, please open an issue in the repository.

