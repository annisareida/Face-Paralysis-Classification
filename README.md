# Face Paralysis Classification

This repository contains the implementation of the **Face Paralysis Classification System**, an experimental final project developed by Informatics Engineering students at **Sriwijaya University**.  
The project focuses on **early detection and severity classification of facial paralysis** using deep learning and transfer learning techniques.

---

## 🎯 Project Objective
The objective of this system is to provide a **fast, consistent, and accessible AI-based screening tool** for facial paralysis assessment.  
The model performs **multi-class classification into six severity levels based on the eFACE scale**, enabling a more detailed and standardized evaluation to support clinical decision-making.

---

## 📊 Dataset Description
- **Dataset source**: Private dataset from *Massachusetts Eye and Ear Infirmary (MEEI)*  
- **Total samples**: **60 facial images**  
- **Classification scale**: **eFACE (6-class classification)**  

### Class Distribution:
- **Complete Paralysis**: 10 samples  
- **Severe Paralysis**: 10 samples  
- **Moderate Paralysis**: 10 samples  
- **Mild Paralysis**: 10 samples  
- **Near Normal**: 10 samples  
- **Normal**: 10 samples  

> The dataset represents varying degrees of facial paralysis severity and was preprocessed and augmented to improve model generalization.

---

## 🔄 Project Workflow
1. **Dataset Preparation**  
   - Data sourced from the MEEI dataset  
   - Image cleaning, resizing, normalization, and augmentation  
   - Labeling based on the **eFACE severity scale**

2. **Model Training**  
   - Transfer learning using **VGGFace architecture**  
   - Implemented with **TensorFlow**  
   - Achieved **99% classification accuracy**, demonstrating strong performance despite limited data availability  

3. **Deployment**  
   - Model deployed as an interactive **Streamlit web application**  
   - Enables real-time facial paralysis severity classification  

---

## 🌐 Live Demo
🔗 **Streamlit App**:  
https://face-paralysis-classification.streamlit.app/

---

## 🛠️ Tech Stack
- **Deep Learning Framework**: TensorFlow  
- **Model Architecture**: VGGFace (Transfer Learning)  
- **Computer Vision**: CNN-based image classification  
- **Deployment**: Streamlit  

---

## 👨‍🎓 About
This project was developed by **Informatics Engineering students at Sriwijaya University** as a **Final Project**, focusing on applying **AI and computer vision techniques in the healthcare domain** to deliver an accessible and reliable facial paralysis classification system.
