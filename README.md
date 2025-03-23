# News-Sentiment-Analysis
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from gtts import gTTS
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import streamlit as st

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

def fetch_news(company_name):
    search_url = f"https://news.google.com/search?q={company_name}&hl=en"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    articles = []
    for item in soup.find_all("article")[:10]:
        title = item.find("h3").text if item.find("h3") else "No Title"
        link = item.find("a")["href"] if item.find("a") else "#"
        summary = item.find("p").text if item.find("p") else "No Summary"
        articles.append({"title": title, "summary": summary, "link": link})
    
    return articles

def analyze_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    return "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"

def extract_topics(text):
    doc = nlp(text)
    return [token.text for token in doc.ents]

def generate_tts(text, filename="output.mp3"):
    tts = gTTS(text=text, lang="hi")
    tts.save(filename)
    return filename

@app.route("/fetch_news", methods=["GET"])
def get_news():
    company = request.args.get("company")
    articles = fetch_news(company)
    
    for article in articles:
        article["sentiment"] = analyze_sentiment(article["summary"])
        article["topics"] = extract_topics(article["summary"])
    
    return jsonify(articles)

if __name__ == "__main__":
    app.run(debug=True)

# Streamlit Web Interface
st.title("📢 News Summarization & Sentiment Analysis")
company_name = st.text_input("Enter Company Name")

if st.button("Analyze News"):
    response = requests.get(f"http://127.0.0.1:5000/fetch_news?company={company_name}")
    news_data = response.json()

    for article in news_data:
        st.subheader(article["title"])
        st.write(article["summary"])
        st.write(f"🗞 Sentiment: {article['sentiment']}")
        st.write(f"📌 Topics: {', '.join(article['topics'])}")
        
        if st.button(f"🔊 Listen (Hindi) - {article['title']}", key=article["title"]):
            tts_file = generate_tts(article["summary"])
            st.audio(tts_file)
