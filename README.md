# RAG System with Video File Support

## Overview
This project is a **Retrieval-Augmented Generation (RAG)** system that processes video files to answer user queries based on their content. By combining audio transcription, image captioning, and multimodal embedding-based retrieval, the system builds a knowledge base from video content and generates intelligent responses to user questions.

---

## Key Features
- **Audio Transcription**: Extracts audio from video and transcribes it using OpenAI Whisper.
- **Keyframe Extraction**: Captures keyframes (I-frames) from video using FFmpeg.
- **Visual Embeddings**: Uses OpenAI CLIP to generate embeddings for keyframes.
- **Image Captioning**: Leverages BLIP for generating textual descriptions of keyframes.
- **Knowledge Base**: Combines audio transcription and visual embeddings using FAISS for similarity-based retrieval.
- **Response Generation**: Generates responses to user queries using OpenAI's GPT model.

---

## Installation

### Prerequisites
Ensure the following are installed on your system:
- Python 3.8 or higher
- FFmpeg (for keyframe extraction)
- CUDA-enabled GPU (for model acceleration)

### Python Dependencies
Install the required Python libraries:
```bash
pip install streamlit openai moviepy pillow sentence-transformers clip-by-openai faiss-cpu torch torchvision torchaudio whisper transformers
```
### How to Run
Clone the repository and navigate to the project directory.
Start the Streamlit application:
```bash
streamlit run Video_RAG.py
```
Open the provided URL in your browser.
Upload a video file (MP4 format) and enter a query based on its content.

---

### File Structure
Video_RAG.py: Main application file.
requirements.txt: Python dependencies.
assets/: Directory for storing intermediate outputs (e.g., keyframes, audio).

---

### How to Use
Upload a video of a nature documentary.
Enter a query like:
"What animals are mentioned in the video?"
"Describe the environment shown in the video."
View the response, which combines insights from the audio transcription and keyframe captions.

---

### Limits
Requires GPU for efficient processing.
Depends on the quality of the video and audio for accurate transcription and captioning.
Limited to English transcription and text-based queries.

---

### Author
Edward Wu for INFO 5940 Final Project, Group 1


