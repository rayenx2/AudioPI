import torch
from TTS.api import TTS
from transformers import pipeline
import pandas as pd
import gradio as gr
import librosa
import numpy as np
import soundfile as sf
from scipy import signal

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load TTS model - use YourTTS for better emotion control
try:
    # YourTTS has better prosody control
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=True).to(device)
    print("TTS model loaded successfully!")
except Exception as e:
    # Fallback to VITS if YourTTS isn't available
    try:
        print(f"Falling back to VITS model: {e}")
        tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True).to(device)
        print("Fallback TTS model loaded successfully!")
    except Exception as e:
        print(f"Error loading TTS models: {e}")
        exit()

# Load emotion classifier and metadata
try:
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    metadata = pd.read_csv("/home/rayen/coqui/vctk_metadata_upd_cleaned_wsl.csv")
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

def apply_audio_emotion_effects(audio_path, emotion):
    """Apply post-processing effects to enhance emotional tone"""
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Apply emotion-specific effects
    if emotion == "joy":
        # Brighten the sound, slightly faster, more dynamic range
        y = signal.lfilter([1, -0.97], [1, -0.8], y)  # Boost higher frequencies
        y = librosa.effects.time_stretch(y, rate=1.05)  # Slightly faster
        # Enhance dynamics for excitement
        y = np.sign(y) * (np.abs(y) ** 0.8)
        
    elif emotion == "sadness":
        # Darker tone, slightly slower, compressed
        y = signal.lfilter([1, -0.8], [1, -0.97], y)  # Boost lower frequencies
        y = librosa.effects.time_stretch(y, rate=0.95)  # Slightly slower
        # Compress dynamic range
        y = np.sign(y) * (np.abs(y) ** 1.2)
        
    elif emotion == "anger":
        # More aggressive tone, slight distortion
        y = signal.lfilter([1, -0.9], [1, -0.85], y)  # Mid-range focus
        # Add slight distortion for aggression
        distortion_gain = 1.2
        y = np.clip(y * distortion_gain, -0.95, 0.95)
        
    # Save the processed audio
    processed_path = audio_path.replace(".wav", "_processed.wav")
    sf.write(processed_path, y, sr)
    return processed_path

def generate_audio(text, category, gender, emotion):
    speaker_id = get_speaker(category, gender, emotion)
    base_output_path = f"/home/rayen/coqui/outputs/{speaker_id}_output.wav"
    
    # Apply emotion-specific parameters if using YourTTS
    if "your_tts" in tts.model_name:
        # YourTTS has better emotion control via style weights
        if emotion == "joy":
            tts.tts_to_file(text=text, speaker=speaker_id, file_path=base_output_path, 
                          speed=1.1, style_wav="path/to/happy_reference.wav", style_weight=0.7)
        elif emotion == "sadness":
            tts.tts_to_file(text=text, speaker=speaker_id, file_path=base_output_path, 
                          speed=0.9, style_wav="path/to/sad_reference.wav", style_weight=0.7)
        elif emotion == "anger":
            tts.tts_to_file(text=text, speaker=speaker_id, file_path=base_output_path, 
                          speed=1.05, style_wav="path/to/angry_reference.wav", style_weight=0.7)
        else:  # neutral
            tts.tts_to_file(text=text, speaker=speaker_id, file_path=base_output_path)
    else:
        # Basic TTS without direct emotion control
        tts.tts_to_file(text=text, speaker=speaker_id, file_path=base_output_path)
    
    # Apply post-processing for emotional effects
    final_output_path = apply_audio_emotion_effects(base_output_path, emotion)
    return final_output_path

def add_emotional_prosody(text, emotion):
    """Add emotion-specific markers to text to guide prosody"""
    if emotion == "joy":
        # Add excitement markers
        text = text.replace("!", "!!") 
        text = text.replace(".", "!")
        # Add emphasis markers for key words
        words = text.split()
        for i, word in enumerate(words):
            if len(word) > 4 and i % 3 == 0:  # Emphasize longer words periodically
                words[i] = f"<emphasis>{word}</emphasis>"
        return " ".join(words)
        
    elif emotion == "sadness":
        # Add pauses for contemplative tone
        text = text.replace(".", "...")
        text = text.replace(",", "...")
        return text
        
    elif emotion == "anger":
        # Add stronger emphasis
        words = text.split()
        for i, word in enumerate(words):
            if len(word) > 3 and i % 2 == 0:  # Emphasize more words
                words[i] = f"<emphasis level='strong'>{word}</emphasis>"
        return " ".join(words)
        
    return text  # Return original for neutral

def marketing_tts(text):
    # Analyze emotion
    emotions = emotion_classifier(text, top_k=None)
    if not emotions:
        print("No emotions detected, defaulting to 'joy'")
        emotion = "joy"
    else:
        emotion = max(emotions[0], key=lambda x: x['score'])['label']
    print(f"Detected emotion: {emotion}")
    
    # Analyze category and gender
    category, gender = classify_category_and_gender(text)
    print(f"Category: {category}, Gender: {gender}")
    
    # Add emotional prosody markers to text
    enhanced_text = add_emotional_prosody(text, emotion)
    
    # Generate audio with emotional enhancements
    audio_path = generate_audio(enhanced_text, category, gender, emotion)
    return audio_path

# Gradio interface with expanded options
with gr.Blocks() as demo:
    gr.Markdown("## Marketing TTS Generator with Emotion Control")
    
    with gr.Row():
        text_input = gr.Textbox(label="Enter marketing text", lines=3)
    
    with gr.Row():
        emotion_override = gr.Radio(
            label="Override detected emotion (optional)", 
            choices=["Auto-detect", "joy", "sadness", "anger", "neutral"],
            value="Auto-detect"
        )
    
    with gr.Row():
        generate_btn = gr.Button("Generate")
    
    with gr.Row():
        output_audio = gr.Audio(label="Generated Audio")
        emotion_display = gr.Label(label="Detected Emotion")
    
    def process_with_override(text, emotion_choice):
        emotions = emotion_classifier(text, top_k=None)
        if not emotions or emotion_choice != "Auto-detect":
            detected_emotion = "joy" if emotion_choice == "Auto-detect" else emotion_choice
        else:
            detected_emotion = max(emotions, key=lambda x: x['score'])['label']
        
        # Rest of processing
        category, gender = classify_category_and_gender(text)
        enhanced_text = add_emotional_prosody(text, detected_emotion)
        audio_path = generate_audio(enhanced_text, category, gender, detected_emotion)
        
        return audio_path, detected_emotion
    
    generate_btn.click(
        fn=process_with_override, 
        inputs=[text_input, emotion_override], 
        outputs=[output_audio, emotion_display]
    )

demo.launch()
