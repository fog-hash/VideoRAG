import streamlit as st
import openai
import os
import subprocess
import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from sentence_transformers import SentenceTransformer
import clip
import torch
import faiss
import numpy as np
from torch import nn
import whisper
from transformers import BlipProcessor, BlipForConditionalGeneration


# Load SentenceTransformer for text embeddings
text_model = SentenceTransformer('all-MiniLM-L6-v2')
# Load CLIP model for generating visual embeddings
clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
# Load Whisper model for audio transcription
whisper_model = whisper.load_model("base")
# Load BLIP model and processor for generating captions from images
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

# Streamlit UI Setup
st.title("RAG System with Video File Support")
st.write("INFO 5940 Final Project Group 1. Upload a video file and ask a question based on its content!")

# UI for video file upload and user query
uploaded_video = st.file_uploader("Upload a video file (MP4 format)", type=["mp4"])
user_query = st.text_input("Enter your query here:")
submit_button = st.button("Submit")

# Utility Functions

def extract_audio(video_file, audio_filename):
    """
    Extract audio from the given video file and save it as an audio file.
    :param video_file: Path to the video file
    :param audio_filename: Path to save the extracted audio file
    :return: Path to the saved audio file
    """
    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(audio_filename, codec='aac')
    return audio_filename

def transcribe_audio(audio_path):
    """
    Transcribe the audio file using the Whisper model.
    :param audio_path: Path to the audio file
    :return: Transcribed text from the audio
    """
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def extract_keyframes(video_path, output_dir):
    """
    Extract keyframes (I-frames) from the video using FFmpeg and save them as images.
    :param video_path: Path to the input video file
    :param output_dir: Directory to save the extracted keyframes
    :return: Directory containing extracted keyframes
    """
    os.makedirs(output_dir, exist_ok=True)
    keyframe_pattern = os.path.join(output_dir, "keyframe_%03d.jpg")
    command = ["ffmpeg", "-i", video_path, 
               "-vf", "select=eq(pict_type\\,I)", 
               "-vsync", "vfr", keyframe_pattern]
    subprocess.run(command, check=True)
    return output_dir

def generate_visual_embeddings(keyframes_dir):
    """
    Generate visual embeddings for all keyframe images using the CLIP model.
    :param keyframes_dir: Directory containing keyframe images
    :return: Numpy array of visual embeddings
    """
    embeddings = []
    keyframe_files = sorted(os.listdir(keyframes_dir))
    for keyframe_file in keyframe_files:
        keyframe_path = os.path.join(keyframes_dir, keyframe_file)
        image = preprocess(Image.open(keyframe_path)).unsqueeze(0).to("cuda")
        with torch.no_grad():
            embedding = clip_model.encode_image(image).cpu().numpy()
        embeddings.append(embedding)
    return np.squeeze(np.array(embeddings), axis=1).astype('float32')

def generate_image_captions(keyframes_dir):
    """
    Generate captions for keyframe images using the BLIP model.
    :param keyframes_dir: Directory containing keyframe images
    :return: List of textual captions for each keyframe
    """
    captions = []
    for keyframe_file in sorted(os.listdir(keyframes_dir)):
        keyframe_path = os.path.join(keyframes_dir, keyframe_file)
        image = Image.open(keyframe_path).convert("RGB")
        inputs = blip_processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = blip_model.generate(**inputs)
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
        captions.append(caption)
    return captions

class EmbeddingProjector(nn.Module):
    """
    A simple neural network to project embeddings into a consistent dimension.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)

def build_knowledge_base(transcription, visual_embeddings, captions, projector):
    """
    Build a combined knowledge base of text and visual embeddings using FAISS for retrieval.
    :param transcription: Transcribed text from the video
    :param visual_embeddings: Visual embeddings generated from keyframes
    :param captions: Textual captions for the keyframes
    :param projector: Projection model to align embedding dimensions
    :return: FAISS index and captions
    """
    # Generate text embedding for the transcription
    text_embedding = text_model.encode(transcription).astype('float32').reshape(1, -1)
    text_embedding = torch.tensor(text_embedding).to("cuda")
    with torch.no_grad():
        text_embedding = projector(text_embedding).cpu().numpy()

    # Combine text and visual embeddings
    combined_embeddings = np.vstack([text_embedding, visual_embeddings])
    index = faiss.IndexFlatL2(combined_embeddings.shape[1])
    index.add(combined_embeddings)
    return index, captions

def retrieve_context(index, query, projector, captions, transcription, top_k=3):
    """
    Retrieve the most relevant text or visual contexts from the knowledge base based on the user query.
    :param index: FAISS index for similarity search
    :param query: User query string
    :param projector: Projection model to align query embedding dimensions
    :param captions: Captions for visual embeddings
    :param transcription: Original transcription text
    :param top_k: Number of top contexts to retrieve
    :return: List of retrieved contexts as strings
    """
    query_embedding = text_model.encode(query).astype('float32').reshape(1, -1)
    query_embedding = torch.tensor(query_embedding).to("cuda")
    with torch.no_grad():
        query_embedding = projector(query_embedding).cpu().numpy()
    distances, indices = index.search(query_embedding, top_k)

    retrieved_contexts = []
    for idx in indices[0]:
        if idx == 0:  # First embedding is text
            retrieved_contexts.append(f"Text: {transcription}.\n")
        else:  # Visual embeddings
            retrieved_contexts.append(f"Visual context: {captions[idx-1]}")
    return retrieved_contexts

def generate_response(context, query):
    """
    Generate a response using OpenAI GPT model based on the retrieved context and user query.
    :param context: Combined text and visual context retrieved from the knowledge base
    :param query: User query string
    :return: Generated response string
    """
    response = openai.chat.completions.create(
        model="openai.gpt-4o",
        messages=[
            {"role": "system", 
             "content": "You are a helpful assistant that understands the transcribed text and description of keyframes of visual inputs."},
            {"role": "user", 
             "content": f"Here is the retrieved context: {context}\nQuestion: {query}"}
        ],
    )
    st.write("Here is the retrieved context: " + context)
    return response.choices[0].message.content

# Main Workflow
if uploaded_video and submit_button:
    st.write("Processing video...")
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "uploaded_video.mp4")

    # Save uploaded video
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    # Extract audio and transcribe
    audio_path = os.path.join(temp_dir, "uploaded_audio.m4a")
    st.write("Extracting audio...")
    extract_audio(video_path, audio_path)

    st.write("Transcribing audio...")
    transcription = transcribe_audio(audio_path)
    st.write("Transcription:", transcription)

    # Extract keyframes and generate embeddings
    st.write("Extracting keyframes...")
    keyframes_dir = os.path.join(temp_dir, "keyframes")
    extract_keyframes(video_path, keyframes_dir)

    st.write("Generating visual embeddings...")
    visual_embeddings = generate_visual_embeddings(keyframes_dir)

    st.write("Generating captions for keyframes...")
    captions = generate_image_captions(keyframes_dir)

    # Initialize projector for consistent embedding dimensions
    projector = EmbeddingProjector(384, 512).to("cuda")
    projector.eval()

    # Build the knowledge base
    st.write("Building knowledge base...")
    index, captions = build_knowledge_base(transcription, visual_embeddings, captions, projector)

    # Query handling
    if user_query:
        st.write("Retrieving relevant context...")
        retrieved_contexts = retrieve_context(index, user_query, projector, captions, transcription)

        context_str = "\n".join(retrieved_contexts)

        # Generate response
        st.write("Generating response...")
        response = generate_response(context_str, user_query)
        st.write("### Response:")
        st.write(response)
