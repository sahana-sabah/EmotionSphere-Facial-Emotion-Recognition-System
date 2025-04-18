# EmotionSphere-Facial-Emotion-Recognition-System
**EmotionSphere** is a Facial Emotion Recognition (FER) system designed to detect human emotions from static image uploads. It uses a Convolutional Neural Network (CNN) trained on facial expression data to classify emotions such as *Happy*, *Sad*, *Angry*, *Fearful*, *Surprised*, and *Neutral*.

## ✨ Features
- **Image-Based Emotion Detection**: Upload an image with a human face and receive the predicted emotion.
- **Deep Learning Model**: Built using CNN architecture and trained on the FER-2013 dataset.
- **Web-Based Interface**: User-friendly interface built with Flask for image upload and result display.
- **Smooth UI**: Responsive and animated frontend using HTML, CSS, and JavaScript.

## 🛠️ Technologies Used
- **Python**
- **TensorFlow / Keras** – Deep learning framework
- **OpenCV** – Image preprocessing
- **Flask** – Backend web framework
- **HTML/CSS/JS** – Frontend
- **NumPy & Pandas** – Data handling

## 📂 Uploaded Image Storage
All uploaded images are stored temporarily in the `uploads/` folder (excluded via `.gitignore`) for detection and display purposes.


## 🔍 How It Works

- The uploaded image is converted to grayscale, resized to 48x48 pixels, and normalized.
- It is then passed through a trained CNN model to classify the emotion.
- The result and uploaded image are displayed back on the web interface for the user.


## 🙌 Contribution & Learning
This project showcases the integration of deep learning and web development. Feel free to explore, modify, or enhance the project further!


Let me know if you want this written into a downloadable `README.md` file!
