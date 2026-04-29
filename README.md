# AI Face Detection & Recognition System

## Live Demo
Try out the deployed application here:

🚀 **Streamlit App** → https://ai-face-detection-and-recognition.streamlit.app/

## Overview
This project is an AI-based Face Detection and Recognition system that identifies whether a given face image belongs to a known person or an unknown individual. The application uses deep learning–based face embeddings to convert faces into numerical representations and compares them against stored known faces to perform recognition accurately.
The system is designed with a modular architecture and provides an efficient pipeline for detecting faces, generating embeddings, and recognizing identities.

## Features
- Detect faces from images
- Generate numerical face embeddings (face fingerprints)
- Compare detected faces with known faces
- Identify known vs unknown individuals using a similarity threshold
- Efficient embedding reuse for better performance
- Clean and modular project structure

## Tech Stack
- Python  
- Streamlit  
- OpenCV
- NumPy
- Face Recognition / Deep Learning Models

## Project Structure
```text
FACE DETECTION AND RECOGNITION/
│
├── app.py                  # Main application entry point
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── gitignore               # Ignore files
├── LICENSE
│
├── ai_engine/
│   ├── detector.py         # Face detection logic
│   ├── recognizer.py       # Face recognition & embedding comparison logic
│   └── __pycache__/
│
├── assets/                 # Dataset of known persons
│   ├── friend/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── lohith/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│
└── screenshots/             # Application output screenshots
```

## Screenshots

### Application Interface
![Application Interface](<screenshots/Screenshot 2025-12-24 165433.png>)

### Image Input
![Image Input](<screenshots/Screenshot 2025-12-24 165452.png>)

### Detection & Recognition Output
For Known:
![Known Output](<screenshots/Screenshot 2025-12-24 165534.png>)
For Unknown:
![Unknown Output](<screenshots/Screenshot 2025-12-24 170204.png>)

## How It Works
First, the system loads images of known people and converts each face into a numerical representation called an embedding. When a new image is provided, the face is detected and converted into an embedding in the same way. These numerical values are then compared with the stored embeddings. If the similarity distance is within a defined threshold, the face is recognized as known; otherwise, it is marked as unknown.

## Installation & Setup
1. Open your IDE (VS Code is recommended).
2. Clone or download the project files to your local system.
3. Create a virtual environment to avoid dependency conflicts:
   ```bash
   python -m venv venv
   venv\Scripts\activate
4. Install the required dependencies:
    pip install -r requirements.txt
5. Run the application:
    streamlit run app.py

## Usage
This application can be used to identify whether a person is already known to the system. Users simply add images of known individuals to the dataset and provide a new image for recognition. The system processes the image and displays whether the detected face belongs to a known person or an unknown one.

## Future Improvements
In the future, this project can be enhanced by adding real-time face recognition using a webcam. Support for detecting and recognizing multiple faces in a single image can also be implemented. Accuracy can be further improved by using advanced face recognition models and integrating a database to manage large numbers of face records efficiently.

## Learning Outcomes
Working on this project helped me understand how face detection and face recognition systems work in real-world applications. I learned how images are converted into numerical embeddings, how similarity comparison is performed and how thresholds affect recognition results. This project improved my understanding of computer vision concepts and strengthened my ability to build practical AI-based applications.
