import torch
from TTS.api import TTS
from transformers import pipeline
import pandas as pd
import gradio as gr

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load TTS model
try:
    tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True).to(device)
    print("TTS model loaded successfully!")
except Exception as e:
    print(f"Error loading TTS model: {e}")
    exit()

# Load emotion classifier and metadata
try:
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    metadata = pd.read_csv("/home/rayen/coqui/vctk_metadata_upd_cleaned_wsl.csv")  # Updated path
    print("Emotion classifier and metadata loaded!")
except Exception as e:
    print(f"Error loading classifier or metadata: {e}")
    exit()

def classify_category_and_gender(text):
    text = text.lower()
    female_fashion = ["makeup", "dress", "heels", "purse", "necklace", "earrings", "bracelet", "handbag"]
    male_fashion = ["tie", "cufflinks", "watch", "belt", "wallet"]
    female_clothes = ["skirt", "blouse", "top", "leggings", "gown"]
    male_clothes = ["suit", "jacket", "trousers", "vest"]
    neutral_clothes = ["shirt", "pants", "jeans", "sweater"]
    food_keywords = ["food", "meal", "recipe", "delicious", "pizza"]
    tech_keywords = ["tech", "gadget", "computer", "phone", "laptop"]
    
    if any(word in text for word in female_fashion):
        return "fashion", "F"
    elif any(word in text for word in male_fashion):
        return "fashion", "M"
    elif any(word in text for word in female_clothes):
        return "clothes", "F"
    elif any(word in text for word in male_clothes):
        return "clothes", "M"
    elif any(word in text for word in neutral_clothes):
        return "clothes", "F" if "style" in text or "trendy" in text else "M"
    elif any(word in text for word in food_keywords):
        return "food", None
    elif any(word in text for word in tech_keywords):
        return "technology", "M"
    return "food", None

def get_speaker(category, gender, emotion):
    emotion_map = {"joy": "hap", "sadness": "sad", "anger": "ang", "neutral": "neu"}
    dataset_emotion = emotion_map.get(emotion, "hap")
    if gender is None:
        gender = metadata['gender'].sample(1).values[0]
    candidates = metadata[(metadata['gender'] == gender) & (metadata['emotion'] == dataset_emotion)]
    if candidates.empty:
        candidates = metadata[metadata['gender'] == gender]
    return candidates.sample(1)['speaker'].values[0]

def generate_audio(text, category, gender, emotion):
    speaker_id = get_speaker(category, gender, emotion)
    output_path = f"/home/rayen/coqui/outputs/{speaker_id}_output.wav"
    tts.tts_to_file(text=text, speaker=speaker_id, file_path=output_path)
    return output_path

def marketing_tts(text):
    emotions = emotion_classifier(text, top_k=None)
    if not emotions:
        print("No emotions detected, defaulting to 'joy'")
        emotion = "joy"
    else:
        emotion = max(emotions, key=lambda x: x['score'])['label']
    print(f"Detected emotion: {emotion}")
    
    category, gender = classify_category_and_gender(text)
    print(f"Category: {category}, Gender: {gender}")
    
    audio_path = generate_audio(text, category, gender, emotion)
    return audio_path

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Marketing TTS Generator")
    text_input = gr.Textbox(label="Enter marketing text")
    output_audio = gr.Audio(label="Generated Audio")
    generate_btn = gr.Button("Generate")
    generate_btn.click(fn=marketing_tts, inputs=text_input, outputs=output_audio)

demo.launch()