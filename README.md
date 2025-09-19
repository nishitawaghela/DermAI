# DermAI
An AI-powered Skin Disease Classifier

DermAI is a machine learning project designed to classify different types of skin diseases from dermatoscopic images.  
It aims to assist dermatologists, medical students, and researchers by providing a quick, accessible, and scalable diagnostic aid.

---

## Overview
- **Problem**: Skin cancer and related conditions are often misdiagnosed due to visual similarities. Early detection is critical.  
- **Solution**: DermAI uses deep learning (CNNs) to classify skin lesions into seven categories.  
- **Goal**: Build an accessible web application using Streamlit where users can upload images and receive predictions.

---

## Dataset
The project uses the **HAM10000 Dataset** (Human Against Machine with 10000 training images), which contains dermatoscopic images of pigmented skin lesions.  

- **Classes:**
  1. `akiec` – Actinic keratoses
  2. `bcc` – Basal cell carcinoma  
  3. `bkl` – Benign keratosis-like lesions  
  4. `df` – Dermatofibroma  
  5. `mel` – Melanoma  
  6. `nv` – Melanocytic nevi  
  7. `vasc` – Vascular lesions  

Dataset source: [Kaggle – HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)

---

## Tech Stack
- **Languages**: Python  
- **Frameworks/Libraries**:
  - TensorFlow / Keras (deep learning models)
  - OpenCV and PIL (image preprocessing)
  - NumPy, Pandas (data handling)
  - Matplotlib, Seaborn (visualization)  


---

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/nishitawaghela/DermAI.git
   cd DermAI
