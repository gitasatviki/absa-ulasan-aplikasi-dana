import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import emoji
import json

st.set_page_config(
    page_title="Analisis Sentimen Berbasis Aspek",
    layout="centered"
)

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_factory = StopWordRemoverFactory()
stop_words = set(stop_factory.get_stop_words())
negation_words = {'tidak', 'tak', 'tanpa', 'bukan', 'belum'}
sentiment_words = {'puas', 'kecewa', 'buruk', 'jelek', 'parah', 'bagus', 'baik', 'mantap',
                   'hilang', 'gagal', 'error', 'bermasalah', 'kehilangan'}
stop_words = stop_words - negation_words - sentiment_words

def load_slang_dict(filepath='slangword.txt'):
    slang_dict = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    slang, formal = line.strip().split(':', 1)
                    slang_dict[slang.strip()] = formal.strip()
    except:
        pass
    return slang_dict

slang_dict = load_slang_dict()
stemmer = StemmerFactory().create_stemmer()

def clean_text(text):
    text = str(text).lower()
    text = ' '.join([slang_dict.get(w, w) for w in text.split()])
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub('[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

@st.cache_resource
def load_resources():
    try:
        with open('word_index_sentiment.json', 'r', encoding='utf-8') as f:
            word_index_sent = json.load(f)
        with open('word_index_topic.json', 'r', encoding='utf-8') as f:
            word_index_topic = json.load(f)
        word_index_sent = {str(k): int(v) for k, v in word_index_sent.items()}
        word_index_topic = {str(k): int(v) for k, v in word_index_topic.items()}

        sentiment_model = tf.keras.models.load_model("model_sentiment_overall.h5")
        topic_model = tf.keras.models.load_model("model_topic_classification.h5")

        topic_names = [
            "Keluhan Saldo Hilang, Transaksi Hilang, dan Keamanan Akun",
            "Masalah Login Akun Baru, Premium, dan Akses Aplikasi",
            "Pujian Aplikasi Mudah, Cepat, Bagus, dan Sangat Membantu"
        ]

        return word_index_sent, word_index_topic, sentiment_model, topic_model, topic_names

    except Exception as e:
        st.error("❌ Gagal memuat model atau word_index. Pastikan file berikut ada:")
        st.error("• model_sentiment_overall.h5")
        st.error("• model_topic_classification.h5") 
        st.error("• word_index_sentiment.json")
        st.error("• word_index_topic.json")
        st.stop()

word_index_sent, word_index_topic, sentiment_model, topic_model, topic_names = load_resources()

st.title("Analisis Sentimen Berbasis Aspek Aplikasi DANA")

user_input = st.text_area(
    "Masukkan ulasan pengguna:",
    height=150,
    placeholder="Contoh: Aplikasi DANA standar saja, transfer cepat tapi promo kurang..."
)

if st.button("🔍 Analisis Ulasan", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("📝 Silakan masukkan ulasan terlebih dahulu.")
    else:
        with st.spinner("Sedang menganalisis ulasan..."):
            tokens = preprocess_text(user_input)
            seq_sent = pad_sequences(
                [[word_index_sent.get(t, 0) for t in tokens]], 
                maxlen=100, 
                padding='post'
            )[0]

            seq_topic = pad_sequences(
                [[word_index_topic.get(t, 0) for t in tokens]], 
                maxlen=100, 
                padding='post'
            )[0]
            pred_sent = sentiment_model.predict(np.expand_dims(seq_sent, axis=0), verbose=0)[0]
            pred_topic = topic_model.predict(np.expand_dims(seq_topic, axis=0), verbose=0)[0]

            sent_idx = np.argmax(pred_sent)
            topic_idx = np.argmax(pred_topic)

            sent_labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
            sentiment = sent_labels[sent_idx]
            sent_conf = float(pred_sent[sent_idx])

            topic_name = topic_names[topic_idx]
            topic_conf = float(pred_topic[topic_idx])

            negative_keywords = [
                'hilang', 'kehilangan', 'blokir', 'diblokir', 'gagal', 'error', 'parah', 
                'kecewa', 'mengecewakan', 'complaint', 'salah', 'hilang', 'bajingan',
                'penipuan', 'scam', 'bodoh', 'jelek', 'buruk', 'lamabat', 'macet'
            ]

            text_lower = user_input.lower()
            has_negative_keyword = any(keyword in text_lower for keyword in negative_keywords)

            negative_topic_indices = [0, 1] 

            fallback_applied = False
            if (topic_idx in negative_topic_indices 
                and topic_conf > 0.95 
                and has_negative_keyword 
                and sentiment != "NEGATIVE"):
                sentiment = "NEGATIVE"
                sent_conf = max(sent_conf, 0.90)
                fallback_applied = True

            st.success("✅ Analisis Selesai!")

            col1, col2 = st.columns(2)

            with col1:
                if sentiment == "POSITIVE":
                    st.success("😊 **Sentimen: POSITIF**")
                elif sentiment == "NEGATIVE":
                    st.error("😞 **Sentimen: NEGATIF**")
                else:
                    st.info("😐 **Sentimen: NETRAL**")

                st.progress(sent_conf)
                st.caption(f"Confidence: **{sent_conf:.1%}**")

            with col2:
                st.warning(f"📌 **Topik Utama: {topic_name.split(',')[0]}**")
                st.progress(topic_conf)
                st.caption(f"Confidence: **{topic_conf:.1%}**")

            st.info(f"**Topik Lengkap:** {topic_name}")

            if fallback_applied:
                st.caption("🔧 *Sentimen disesuaikan otomatis berdasarkan keluhan serius*")