# 🌿 Plant Disease Detection using Deep Learning

A full-stack AI-powered web application that detects plant diseases from leaf images using a Convolutional Neural Network (CNN). Built with Flask and TensorFlow, the app allows users to upload an image and get real-time predictions.

---

## 🚀 Features

- Upload plant leaf images via a web interface  
- Deep learning model detects disease type with high accuracy  
- Covers multiple plant types: Apple, Corn, Potato, Tomato  
- Clean UI with animated background and responsive design  
- Lightweight and easy to deploy locally

---

## 🧠 Tech Stack

| Area           | Tools / Libraries         |
|----------------|---------------------------|
| Language       | Python 3.10+              |
| Framework      | Flask                     |
| Deep Learning  | TensorFlow, Keras         |
| Frontend       | HTML5 + Inline CSS        |
| Image Handling | Pillow, Keras Preprocessing |

---

## 🖼️ Classes Supported

- Apple - Black Rot  
- Apple - Healthy  
- Corn (Maize) - Common Rust  
- Corn (Maize) - Healthy  
- Potato - Late Blight  
- Potato - Healthy  
- Tomato - Early Blight  
- Tomato - Healthy  

---

## 📁 Project Structure

```
plant-disease-detection/
├── backend/
│   ├── app.py                  # Flask backend
│   ├── predict.py              # Model loading + prediction logic
│   ├── model/
│   │   └── plant_disease_model.h5
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css (optional if not using inline CSS)
├── README.md
```

---

## 🛠️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection/backend
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📌 If requirements.txt is missing, install manually:</summary>

```bash
pip install flask tensorflow pillow
```

</details>

### 3. Run the App

```bash
python app.py
```

Then open your browser and go to:  
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🌐 Frontend Details

- `index.html` uses inline CSS and has custom background animations  
- Upload box is styled and large for user friendliness  
- Uses a subtle leaf-themed UI for visual appeal

---

## 📦 Model Details

- Trained on a subset of the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Input size: 224x224
- Output: Softmax probabilities over 8 classes
- Format: `plant_disease_model.h5`

---

## ✅ To-Do / Future Improvements

- Add more plant types and diseases  
- Add drag-and-drop image support  
- Deploy on Render / Railway / Hugging Face Spaces  
- Replace inline CSS with Tailwind or Bootstrap for better theming

---

## 🙋‍♂️ Author

**Tanmay Bangar**  
GitHub: [@t0nyb17](https://github.com/t0nyb17)  

---

## 📸 Screenshots

<img width="1919" height="1024" alt="Screenshot 2025-07-27 194714" src="https://github.com/user-attachments/assets/460e3483-3eb2-48af-b6a8-3aabcfcd2c79" />
<img width="1919" height="1079" alt="Screenshot 2025-07-27 194726" src="https://github.com/user-attachments/assets/5d488c0e-de02-487c-b131-1f3202ec3f1f" />
<img width="1920" height="1080" alt="Screenshot (1753)" src="https://github.com/user-attachments/assets/0ff35767-e052-4e89-b743-5bceb7e39122" />
<img width="1920" height="1080" alt="Screenshot (1754)" src="https://github.com/user-attachments/assets/74763410-9c25-4efe-b85f-113305e41f9a" />
<img width="1920" height="1080" alt="Screenshot (1755)" src="https://github.com/user-attachments/assets/3d1d5ba0-82cd-4e78-a6e7-83448acc1d9c" />
<img width="1590" height="788" alt="Screenshot 2025-07-27 212019" src="https://github.com/user-attachments/assets/a4dcd1c7-81dc-41a1-835d-a30cf38c86ee" />
<img width="553" height="414" alt="Screenshot 2025-07-27 184955" src="https://github.com/user-attachments/assets/01999cd6-7643-42bf-a5ef-a9d5c85879dc" />

---
