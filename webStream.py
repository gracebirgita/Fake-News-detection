import streamlit as st
import spacy
import requests
from bs4 import BeautifulSoup
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import time # Untuk simulasi loading
import random # Untuk model dummy jika Anda belum punya
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import subprocess
import sys

st.set_page_config(page_title="Deteksi Berita Palsu", layout="wide")


lemma = WordNetLemmatizer()
# nlp = spacy.load('en_core_web_sm')


try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')


#stopword
list1 = nlp.Defaults.stop_words
list2 = stopwords.words('english')
#combinig
Stopwords = set((set(list1) | set(list2)))
# --- KONFIGURASI MODEL ANDA (GANTI ATAU SESUAIKAN) ---
# Pastikan file model dan vectorizer (jika ada) berada di direktori yang sama
# dengan script Python ini, atau berikan path lengkapnya.
MODEL_PATH = 'model_dataset1.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

# --- FUNGSI UNTUK MENGAMBIL TEKS DARI URL ---
@st.cache_data(show_spinner=False) # Cache hasil agar tidak fetch ulang jika URL sama
def get_text_from_url(url):
    """Mengambil teks utama dari sebuah URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Akan error jika status code bukan 2xx
        soup = BeautifulSoup(response.content, 'html.parser')

        # Coba ekstrak teks dari tag <article> atau <p>
        # Ini adalah heuristik sederhana, mungkin perlu disesuaikan
        article_tag = soup.find('article')
        if article_tag:
            paragraphs = article_tag.find_all('p')
        else:
            # Jika tidak ada tag article, coba cari semua paragraf
            paragraphs = soup.find_all('p')

        if not paragraphs: # Jika tidak ada <p>, ambil semua teks terlihat
            return soup.get_text(separator=' ', strip=True)

        text_content = ' '.join([p.get_text(strip=True) for p in paragraphs])
        return text_content
    except requests.exceptions.RequestException as e:
        st.error(f"Gagal mengambil konten dari URL: {e}")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses URL: {e}")
        return None

# --- FUNGSI UNTUK MEMUAT MODEL DAN PREDIKTOR (GANTI INI) ---
@st.cache_resource # Gunakan ini untuk memuat model sekali saja
def load_model_and_dependencies():
    """Memuat model dan dependensi (misal: vectorizer)."""
    # GANTI INI DENGAN LOGIKA MODEL ANDA
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        # Inisialisasi lain jika ada (misal: nltk.download jika perlu)
        # nltk.download('punkt', quiet=True)
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"File model atau vectorizer tidak ditemukan. Pastikan '{MODEL_PATH}' dan 'Vectorizer' ada.")
        return None, None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

# Panggil fungsi load model di luar fungsi prediksi agar di-cache
loaded_model, loaded_vectorizer = load_model_and_dependencies()

def preprocess_text(text):
    
    string = ""
    
    #lower casing
    text=text.lower()
    
    #simplifying text
    text=re.sub(r"i'm","i am",text)
    text=re.sub(r"he's","he is",text)
    text=re.sub(r"she's","she is",text)
    text=re.sub(r"that's","that is",text)
    text=re.sub(r"what's","what is",text)
    text=re.sub(r"where's","where is",text)
    text=re.sub(r"\'ll"," will",text)
    text=re.sub(r"\'ve"," have",text)
    text=re.sub(r"\'re"," are",text)
    text=re.sub(r"\'d"," would",text)
    text=re.sub(r"won't","will not",text)
    text=re.sub(r"can't","cannot",text)
    
    #removing any special character
    text=re.sub(r"[-()\"#!@$%^&*{}?.,:]"," ",text)
    text=re.sub(r"\s+"," ",text)
    text=re.sub('[^A-Za-z0-9]+',' ', text)
    
    for word in text.split():
        if word not in Stopwords:
            string+=lemma.lemmatize(word)+" "
    
    return string



# --- FUNGSI PREDIKSI (GANTI INI DENGAN LOGIKA MODEL ANDA) ---
def predict_news(text_input):
    """
    Fungsi untuk melakukan prediksi berita palsu.
    Ganti fungsi ini dengan logika model Anda.
    """
    if loaded_model is None or loaded_vectorizer is None:
        return "Error: Model tidak berhasil dimuat.", 0.0

    # 1. Preprocessing teks (sesuaikan dengan model Anda)
    #    Contoh: tokenisasi, stemming, lemmatization, tf-idf, dll.
    text_processed = preprocess_text(str(text_input)) # Fungsi preprocessing Anda
    # text_vectorized = loaded_vectorizer.transform([text_processed])

    # 2. Lakukan prediksi dengan model Anda
    prediction_proba = loaded_model.predict_proba([text_processed])
    predicted_class_index = np.argmax(prediction_proba)
    confidence = np.max(prediction_proba)

    # 3. Tentukan label berdasarkan hasil prediksi
    #    Asumsikan model Anda mengeluarkan kelas: 0 untuk Asli, 1 untuk Palsu
    if predicted_class_index == 0: # Indeks untuk 'Palsu'
        label = "Fake"
    else: # Indeks untuk 'Asli'
        label = "True"
    return label, float(confidence)

    # ----- CONTOH DUMMY MODEL (HAPUS ATAU GANTI BAGIAN INI) -----
    # st.warning("PERHATIAN: Ini adalah hasil dari MODEL DUMMY. Ganti fungsi `predict_news` dengan model Anda.")
    # time.sleep(1) # Simulasi proses prediksi
    # if not text_input or len(text_input.split()) < 10: # Jika teks terlalu pendek
    #     return "Tidak Cukup Informasi", 0.0
    
    # # Logika dummy sederhana
    # is_fake = random.choice([True, False, False]) # Lebih cenderung asli untuk dummy
    # confidence = random.uniform(0.65, 0.98)
    # if is_fake:
    #     return "Palsu", confidence
    # else:
    #     return "Asli", confidence
    # ----- AKHIR DARI CONTOH DUMMY MODEL -----

# --- ANTARMUKA STREAMLIT ---


st.title("📰 Fake News Detector")

st.write("")

st.markdown("""
Enter the news URL link or paste the news text below to detect whether the news is potentially fake or real.
""")

st.write("")

# st.subheader("Input format : ")
input_method = st.radio("Input format:", ("Link", "Text"))

news_text = ""
url_input = ""

st.markdown("---")


if input_method == "Link":
    url_input = st.text_input("🔗 Input News URL:", placeholder="Contoh: https://www.contohberita.com/...")
    if url_input:
        with st.spinner("Fetching content from URL..."):
            news_text = get_text_from_url(url_input)
            if news_text:
                st.text_area("News content taken from URL (preview):", news_text, height=200, disabled=True)
            else:
                st.info("Cannot retrieve text content from that URL or the URL is invalid.")
                news_text = "" # Kosongkan news_text jika gagal


elif input_method == "Text":
    news_text_input = st.text_area("📝 Enter or paste news content here:", height=250, placeholder="Type or paste news content...")
    if news_text_input:
        news_text = news_text_input

if st.button("🔎 Predict Now!", type="primary", use_container_width=True, disabled=(not news_text)):
    if news_text:
        with st.spinner("Analyzing the news... Please wait a moment."):
            # Panggil fungsi prediksi Anda
            prediction_label, confidence_score = predict_news(news_text)

        st.subheader("Prediction Result:")
        if prediction_label == "Fake":
            st.error(f"**This news is indicated: {prediction_label}**")
        elif prediction_label == "True":
            st.success(f"**This news is indicated: {prediction_label}**")
        else: # Untuk kasus seperti "Tidak Cukup Informasi"
            st.warning(f"**Status: {prediction_label}**")

        if confidence_score > 0: # Hanya tampilkan skor jika relevan
            st.metric(label="Model Confidence", value=f"{confidence_score*100:.2f}%")

        with st.expander("View the analyzed text"):
            st.caption("The following text was used for analysis:")
            st.markdown(f"> {news_text[:1000]}...") # Tampilkan sebagian teks
    else:
        st.warning("Please enter the news URL or fill in the news text first...")

st.markdown("---")
st.caption("Disclaimer: These predictions are based on machine learning models and may not always be 100% accurate. Use them as a tool and always cross-verify with trusted sources.")