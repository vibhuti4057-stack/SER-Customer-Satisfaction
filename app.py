import streamlit as st
import sqlite3
from datetime import datetime
import pandas as pd
from analyze_speech import get_satisfaction

conn = sqlite3.connect('calls.db', check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS calls 
    (id INTEGER PRIMARY KEY, filename TEXT, emotion TEXT, 
     sentiment REAL, score REAL, transcript TEXT, timestamp TEXT)""")
conn.commit()

st.title("🎙️ Customer Satisfaction Detector")
st.write("Record your voice OR upload a call recording to detect emotion and satisfaction level")

# NEW: Let user choose between mic recording and file upload
option = st.radio("Choose input method:", ["🎤 Record Voice", "📁 Upload File"])

audio_bytes = None
filename = None

if option == "🎤 Record Voice":
    recorded_audio = st.audio_input("Click to record")
    if recorded_audio:
        audio_bytes = recorded_audio.getbuffer()
        filename = "live_recording.wav"

elif option == "📁 Upload File":
    uploaded_file = st.file_uploader("Upload audio file", type=["wav"])
    if uploaded_file:
        audio_bytes = uploaded_file.getbuffer()
        filename = uploaded_file.name

if audio_bytes:
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)

    with st.spinner("Analyzing..."):
        result = get_satisfaction("temp.wav")

    st.success("Analysis complete!")
    st.write("**Detected Emotion:**", result['emotion'])
    st.write("**Transcribed Text:**", result['text'])
    st.write("**Sentiment Score:**", result['sentiment_score'])
    st.write("**Satisfaction Score:**", result['satisfaction_score'])

    conn.execute("INSERT INTO calls (filename, emotion, sentiment, score, transcript, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                 (filename, result['emotion'], result['sentiment_score'], 
                  result['satisfaction_score'], result['text'], str(datetime.now())))
    conn.commit()

st.subheader("📊 Call History")
history = pd.read_sql("SELECT * FROM calls ORDER BY timestamp DESC", conn)
if not history.empty:
    st.dataframe(history)
    st.line_chart(history.set_index('timestamp')['score'])
else:
    st.write("No calls analyzed yet.")