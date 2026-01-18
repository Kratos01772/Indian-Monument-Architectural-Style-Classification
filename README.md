# Indian-Monument-Architectural-Style-Classification
A deep learning pipeline for classifying Indian monuments by architectural style using Inception-based CNNs, saliency-driven preprocessing, and memory-optimized TensorFlow workflows.
# Indian Monument Architectural Style Classification

A **deep learning–based computer vision project** that classifies Indian monuments into **architectural styles** using **Inception-based CNNs**, **saliency-driven region extraction**, and a **memory-optimized TensorFlow pipeline**.

This project is designed to be **research-grade**, scalable, and suitable for both **academic use and real-world deployment**.

---

##  Architectural Styles Classified

The model predicts the following **8 Indian architectural styles**:

- Ancient_Caves  
- Buddhist  
- Colonial  
- Delhi_Sultanate  
- Dravidian  
- Mughal  
- Nagara  
- Rajput  

Each image is labeled based on **architectural style**, not individual monuments.

---

##  Key Features

###  End-to-End Pipeline
- Dataset cleaning & validation  
- EXIF correction and RGB normalization  
- Memory-optimized resizing for large images  
- Saliency-based region extraction (OpenCV)  
- TensorFlow `tf.data` pipeline  
- Transfer learning + fine-tuning  

###  Saliency-Aware Learning
- Uses **Spectral Residual Saliency** (OpenCV)
- Automatically extracts visually important regions
- Fallback cropping for failure cases
- Improves robustness on cluttered images

###  Deep Learning Architecture
- **InceptionV3**
- **InceptionResNetV2**
- ImageNet pretrained weights
- Two-phase training:
  - Feature extraction
  - Fine-tuning (partial unfreeze)

###  Performance & Optimization
- Mixed precision training (`float16`)
- Aggressive memory optimization (16GB RAM friendly)
- Multi-threaded preprocessing
- Efficient dataset caching & prefetching

---

## 📂 Dataset Structure

