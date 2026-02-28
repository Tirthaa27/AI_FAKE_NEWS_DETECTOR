import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils.bias_detector import detect_bias
from utils.fake_detector import detect_fake_news
# from utils.bias_detector import detect_bias

st.set_page_config(page_title="AI Fake News Detector", layout="wide")

# ------------------- CSS -------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.header {
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: white;
}

.real { background: linear-gradient(135deg,#11998e,#38ef7d); }
.fake { background: linear-gradient(135deg,#ff416c,#ff4b2b); }

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="header"><h1>📰 AI Fake News & Bias Detector</h1></div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Detect News", "ℹ️ Model Info"])

# ================= TAB 1 =================
with tab1:

    col1, col2 = st.columns([2,1])

    with col1:
        news_text = st.text_area(
            "News Article",
            height=300,
            placeholder="Paste news article here...",
            label_visibility="collapsed"
        )
        analyze = st.button("Analyze")

    with col2:
        if analyze and news_text:

            with st.spinner("Analyzing with AI model..."):

                label, score = detect_fake_news(news_text)
                bias_label, bias_score = detect_bias(news_text)

            # ---------------- RESULT CARD ----------------
            if "REAL" in label:
                st.markdown('<div class="result-card real">REAL NEWS ✅</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-card fake">FAKE NEWS ❌</div>', unsafe_allow_html=True)

            st.markdown("---")

            # ---------------- 📊 Circular Confidence Chart ----------------
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "Confidence %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "cyan"},
                    'steps': [
                        {'range': [0, 50], 'color': "#ff4b2b"},
                        {'range': [50, 75], 'color': "#f9d423"},
                        {'range': [75, 100], 'color': "#38ef7d"},
                    ],
                }
            ))
            fig.update_layout(height=250, margin=dict(t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # ---------------- 🌈 Color Changing Bar ----------------
            if score < 50:
                bar_color = "red"
            elif score < 75:
                bar_color = "orange"
            else:
                bar_color = "green"

            st.markdown(f"### 🌈 Confidence Bar ({score}%)")
            st.progress(score / 100)

            # ---------------- 📌 Credibility Meter ----------------
            credibility = "High" if score > 75 else "Medium" if score > 50 else "Low"
            st.markdown(f"### 📌 News Credibility: **{credibility}**")

            # ---------------- 🎯 Risk Level ----------------
            risk = "Low Risk" if "REAL" in label else "High Risk"
            st.markdown(f"### 🎯 Risk Level: **{risk}**")

            # ---------------- 🧠 AI Score Breakdown ----------------
            st.markdown("### 🧠 AI Score Breakdown")
            st.write(f"• Fake/Real Classification Confidence: {score}%")
            st.write(f"• Bias Confidence: {bias_score}%")

            # ---------------- 📈 Fake Probability Graph ----------------
            st.markdown("### 📈 Fake Probability Distribution")
            probabilities = [score, 100-score]
            labels_graph = ["Predicted Label", "Opposite Label"]

            fig2 = go.Figure([go.Bar(
                x=labels_graph,
                y=probabilities
            )])
            fig2.update_layout(height=250)
            st.plotly_chart(fig2, use_container_width=True)

            # ---------------- 🌍 Source Reliability ----------------
            st.markdown("### 🌍 Source Reliability")
            st.info("Source reliability estimation is based on textual consistency and bias signals. (Advanced source verification module can be integrated here.)")

        else:
            st.info("Results will appear here.")

# ================= TAB 2 =================
with tab2:
    st.markdown("""
    ## 🧠 Model Information

    **Fake News Detection**
    - Model: facebook/bart-large-mnli
    - Framework: Hugging Face Transformers
    - Architecture: BART (Transformer)
    - Method: Zero-Shot Classification
    - Size: ~1.6GB
    - Running: Locally via PyTorch

    **Bias Detection**
    - Multi-class Zero-Shot Classification
    - Labels: Left / Right / Neutral

    **Technology Stack**
    - Frontend: Streamlit
    - Backend: Python
    - NLP Engine: Transformers + PyTorch
    - Visualization: Plotly

    This system performs semantic classification without supervised retraining.
    """)