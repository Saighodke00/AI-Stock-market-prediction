import streamlit as st

st.set_page_config(
    page_title="About Us - Apex AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .member-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    .member-name {
        font-size: 20px;
        font-weight: bold;
        color: #00d2aa;
        margin-bottom: 10px;
    }
    .member-role {
        font-size: 14px;
        color: #888;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title("👥 About the Team")
st.markdown("---")
st.write("Meet the minds behind **Apex AI - Stock Prediction**.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="member-card">
        <div class="member-name">Sai Narendra Ghodke</div>
        <div class="member-role">Lead AI Architect & Developer</div>
        <p>Contribution: Spearheaded the development of the LSTM neural network architecture and orchestrated the end-to-end system design. Focused on model optimization and predictive accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="member-card">
        <div class="member-name">Siddhartha Vijay Bhosale</div>
        <div class="member-role">Data Scientist & Quantitative Analyst</div>
        <p>Contribution: Led the data acquisition and preprocessing pipelines. Implemented technical indicators (RSI, SMA) and the core trading logic algorithms.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="member-card">
        <div class="member-name">Sunraj Shetty</div>
        <div class="member-role">Frontend Engineer & UI/UX Specialist</div>
        <p>Contribution: Designed the glassmorphism-inspired user interface. Ensured responsive layout and seamless user experience across the dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.info("Apex AI Project - 2025")
