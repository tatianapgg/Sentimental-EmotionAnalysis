import streamlit as st
import os
import whisper
import numpy as np
import pandas as pd
import tempfile
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
import pickle
import random
import ast

# --- Setup ---
st.set_page_config(page_title="🎧 Sentiment & Emotion Analyzer + Music Finder", layout="centered")
os.environ["PATH"] += os.pathsep + r"C:\Users\filip\OneDrive\Desktop\bin"

# --- CSS Styling ---
st.markdown("""
    <style>
    .block-container {
        max-width: 900px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .main {
        background-color: #fff6fa;
        padding: 2rem;
    }
    h1 {
        color: #3b3b3b;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
        font-size: 2.5rem;
    }
    .stTextInput > div > div > input, .stTextArea textarea {
        background-color: #fff0f5;
        border-radius: 10px;
        padding: 10px;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton button,
    .stButton button:hover,
    .stButton button:focus,
    .stButton button:active {
        background-color: #ff69b4 !important;
        color: white !important;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        outline: none;
        box-shadow: none;
    }
    </style>
""", unsafe_allow_html=True)

# --- Custom Attention Layer ---
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        et = tf.nn.tanh(tf.matmul(x, self.W) + self.b)  # shape: (batch_size, seq_len, 1)
        at = tf.nn.softmax(et, axis=1)  # shape: (batch_size, seq_len, 1)
        output = x * at  # Apply attention weights
        return output

# --- Focal loss for Emotion Model ---
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(alpha_factor * modulating_factor * cross_entropy)
    return loss

# --- Load music_labels dataset ---
@st.cache_data
def load_music_labels():
    return pd.read_csv(r"C:\Users\filip\OneDrive\Desktop\sentiment_demo\music_labels.csv")

music_labels = load_music_labels()

# --- Load models separately ---
@st.cache_resource
def load_sentiment_model():
    model = load_model("best_model.h5", compile=False)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

@st.cache_resource
def load_emotion_model():
    model = load_model(
        "best_model_emo.h5",
        custom_objects={'AttentionLayer': AttentionLayer, 'loss': focal_loss()},
        compile=False
    )
    model.compile(
        loss=focal_loss(),
        optimizer='adam',
        metrics=[tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall()]
    )
    return model

model_sent = load_sentiment_model()
model_emo = load_emotion_model()

# --- Load Whisper ---
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# --- Load tokenizers ---
@st.cache_resource
def load_tokenizer_sentiment():
    with open("tokenizer_sentimento.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_tokenizer_emotion():
    with open("tokenizer_emocoes.pkl", "rb") as f:
        return pickle.load(f)

tokenizer_sent = load_tokenizer_sentiment()
tokenizer_emo = load_tokenizer_emotion()

# --- Load maxlen ---
with open("maxlen_emocoes.pkl", "rb") as f:
    maxlen_emo = pickle.load(f)
maxlen_sent = 400

emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# --- Preprocessing ---
def preprocess_sentiment(texts):
    if isinstance(texts, str):
        texts = [texts]
    sequences = tokenizer_sent.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen_sent, padding="post", truncating="post")
    return padded

def preprocess_emotion(texts):
    if isinstance(texts, str):
        texts = [texts]
    sequences = tokenizer_emo.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen_emo, padding="post", truncating="post")
    return padded

# --- Prediction ---
def predict(text):
    X_sent = preprocess_sentiment(text)
    X_emo = preprocess_emotion(text)

    sentiment_pred = model_sent.predict(X_sent)
    emotion_pred = model_emo.predict(X_emo)

    sentiment_label = "positive" if sentiment_pred[0][0] >= 0.5 else "negative"
    top_emotion_idx = np.argmax(emotion_pred[0])
    top_emotion = emotion_labels[top_emotion_idx]

    return sentiment_label, float(sentiment_pred[0][0]), top_emotion

# --- UI helpers ---
def sentiment_box(sentiment, score):
    bg_color = "#d1fae5" if sentiment == "positive" else "#fee2e2"
    text_color = "#065f46" if sentiment == "positive" else "#991b1b"
    border_color = "#10b981" if sentiment == "positive" else "#ef4444"
    return f"""
    <div style="
        background-color: {bg_color};
        color: {text_color};
        font-weight: 600;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-left: 6px solid {border_color};
        font-family: 'Segoe UI', sans-serif;
        font-size: 1rem;
        box-shadow: rgba(0, 0, 0, 0.05) 1px 1px 4px 0px;
        display: block;
    ">Sentiment: {sentiment} ({score:.2f})</div>
    """

def floating_emojis(sentiment):
    if sentiment == "positive":
        emojis = "🎈"
        animation_css = """
        @keyframes floatUp {
            0% { transform: translateY(0vh) scale(0.8); opacity: 1; }
            100% { transform: translateY(-110vh) scale(1.2); opacity: 0; }
        }
        .float-emoji {
            position: fixed;
            bottom: 0;
            font-size: 4rem;
            animation-name: floatUp;
            animation-duration: 6s;
            animation-iteration-count: infinite;
            animation-timing-function: ease-in-out;
            pointer-events: none;
            user-select: none;
            will-change: transform, opacity;
            filter: drop-shadow(0 0 6px rgba(255,105,180,0.7));
        }
        """
    else:
        emojis = "😢"
        animation_css = """
        @keyframes floatUp {
            0% { transform: translateY(0vh) scale(0.8); opacity: 1; }
            100% { transform: translateY(-110vh) scale(1.2); opacity: 0; }
        }
        .float-emoji {
            position: fixed;
            bottom: 0;
            font-size: 3.5rem;
            animation-name: floatUp;
            animation-duration: 6s;
            animation-iteration-count: infinite;
            animation-timing-function: ease-in-out;
            pointer-events: none;
            user-select: none;
            will-change: transform, opacity;
            filter: drop-shadow(0 0 5px rgba(0,0,0,0.4));
        }
        """
    floats_html = ""
    for i in range(12):
        left_pos = random.randint(5, 95)
        delay = round(random.uniform(0, 6), 2)
        floats_html += f'<div class="float-emoji" style="left:{left_pos}%; animation-delay:{delay}s">{emojis}</div>'
    return f"<style>{animation_css}</style>{floats_html}"

# --- Helper to check if emotion is in the predicted_emotion list column ---
def emotion_in_predicted_list(row_emotion_str, target_emotion):
    try:
        emotions_list = ast.literal_eval(row_emotion_str) if isinstance(row_emotion_str, str) else []
        emotions_list = [e.strip().lower() for e in emotions_list]
        return target_emotion.lower() in emotions_list
    except Exception:
        return False

# --- Music Finder ---
def music_finder():
    st.header("🎵 Music Finder")
    text_input = st.text_area("Describe how you feel or what you want to listen to:")

    if st.button("Find Playlist"):
        if not text_input.strip():
            st.warning("Please enter some text.")
            return

        sentiment, score, emotion = predict(text_input)
        st.markdown(sentiment_box(sentiment, score), unsafe_allow_html=True)
        st.info(f"**Detected Emotion:** {emotion}")
        st.markdown(floating_emojis(sentiment), unsafe_allow_html=True)

        filtered_songs = music_labels[music_labels['predicted_emotion'].apply(
            lambda x: emotion_in_predicted_list(x, emotion)
        )]
        if not filtered_songs.empty:
            st.markdown("### Here's a playlist for you:")
            for _, row in filtered_songs.iterrows():
                artist = row['artist_name']
                track = row['track_name']
                genre = row.get('genre', 'Unknown Genre')
                st.markdown(f"- **{track}** by *{artist}* ({genre})")
        else:
            st.info("Sorry, no songs found for this emotion yet.")

# --- Sentiment & Emotion Analyzer ---
def sentiment_emotion_analyzer():
    st.header("🎧 Sentiment + Emotion Analyzer")
    input_type = st.radio("Choose input type:", ["📝 Text", "🎙️ Audio"])

    if input_type == "📝 Text":
        user_input = st.text_area("Enter your text:")

        if st.button("Analyze Text"):
            if user_input.strip():
                sentiment, score, emotion = predict(user_input)
                st.markdown(sentiment_box(sentiment, score), unsafe_allow_html=True)
                st.info(f"**Emotion:** {emotion}")
                st.markdown(floating_emojis(sentiment), unsafe_allow_html=True)
            else:
                st.warning("Please enter some text.")

    else:
        audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "mp4"])
        if audio_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_file.read())
                audio_path = tmp.name

            st.audio(audio_path)
            st.info("Transcribing with Whisper...")

            try:
                result = whisper_model.transcribe(audio_path)
                transcription = result["text"]
                st.markdown(f"**Transcription:** {transcription}")

                sentiment, score, emotion = predict(transcription)
                st.markdown(sentiment_box(sentiment, score), unsafe_allow_html=True)
                st.info(f"**Emotion:** {emotion}")
                st.markdown(floating_emojis(sentiment), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Transcription failed: {e}")

# --- Main app ---
st.title("Welcome to the Multi-Tool App 🎵🎧")
app_mode = st.selectbox("Choose an app:", ["Music Finder", "Sentiment & Emotion Analyzer"])

if app_mode == "Music Finder":
    music_finder()
else:
    sentiment_emotion_analyzer()
