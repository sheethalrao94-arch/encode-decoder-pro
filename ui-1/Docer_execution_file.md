# Encode Decoder Pro - Docker Execution Guide

Author: Sheethal  
Roll No: 251100670032  
Course: BDA  
Project: Encode Decoder Pro  

Docker Hub Repository:
https://hub.docker.com/repositories/sheethalrao94

---

# 📌 Project Overview

This project implements:

- Static Huffman Coding
- Dynamic Huffman (FGK Algorithm)
- LZW Compression
- Arithmetic Coding
- Step-by-step tree visualization
- Jupyter Notebook academic demonstration

The entire system is containerized using Docker.

---

# 🐳 1️⃣ Build Docker Image (Local Machine)

Open terminal inside project directory:

```bash
cd UI-1
docker build -t encode-decoder-pro .
```
▶️ 2️⃣ Run Docker Container
```bash
docker run -p 5000:5000 -p 8888:8888 encode-decoder-pro

```
🌐 3️⃣ Access the Application

Flask Web Application:
http://localhost:5000
Jupyter Notebook (Direct File):
http://localhost:8888/notebooks/Sheethal_BDA_251100670032_ED.ipynb

📤 4️⃣ Push Image to Docker Hub
```bash
docker login
docker tag encode-decoder-pro sheethalrao94/encode-decoder-pro:latest
docker push sheethalrao94/encode-decoder-pro:latest

```
🌍 5️⃣ Run From Docker Hub (Anywhere)
```bash
docker pull sheethalrao94/encode-decoder-pro:latest

docker run -p 5000:5000 -p 8888:8888 sheethalrao94/encode-decoder-pro

```
🛑 6️⃣ Stop Container
```bash
docker ps
docker stop <container_id>
```