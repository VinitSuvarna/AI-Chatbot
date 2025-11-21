# assistant_app.py
import streamlit as st
import pandas as pd
import google.generativeai as genai
import fitz 
import os 

# --- Page Configuration ---
st.set_page_config(
    page_title="Root Cause Analysis Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Enhanced UI ---
st.markdown("""
<style>
    /* Main app styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Status cards */
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .status-card.success {
        border-left-color: #10b981;
    }
    
    .status-card.warning {
        border-left-color: #f59e0b;
    }
    
    .status-card.error {
        border-left-color: #ef4444;
    }
    
    .status-card h4 {
        margin: 0 0 0.5rem 0;
        color: #374151;
    }
    
    .status-card p {
        margin: 0;
        color: #6b7280;
        font-size: 0.9rem;
    }
    
    /* Chat container */
    .chat-container {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        min-height: 400px;
        border: 1px solid #e2e8f0;
    }
    
    /* Input styling */
    .stChatInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
    }
    
    .stChatInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Data source indicators */
    .data-source {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        background: #f1f5f9;
    }
    
    .data-source.loaded {
        background: #dcfce7;
        color: #166534;
    }
    
    .data-source.missing {
        background: #fef2f2;
        color: #991b1b;
    }
    
    /* Metrics styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        margin: 0;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading-animation {
        animation: pulse 1.5s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration for Gemini API ---

api_status = {"configured": False, "model": None, "error": None}

try:
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    else:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('models/gemini-2.5-pro')
    api_status["configured"] = True
    api_status["model"] = model
except Exception as e:
    api_status["error"] = str(e)

# --- Data Loading and Preprocessing (cached) ---
@st.cache_data
def load_all_data():
    # Load CSV
    df_csv = pd.read_csv('customer_interaction.csv')

    # Convert 'Event Timestamp'
    df_csv['Event Timestamp'] = pd.to_datetime(df_csv['Event Timestamp'], errors='coerce')
    df_csv.dropna(subset=['Event Timestamp'], inplace=True)

    # Ensure 'Response Time (s)' is numeric
    df_csv['Response Time (s)'] = pd.to_numeric(df_csv['Response Time (s)'], errors='coerce')
    df_csv.dropna(subset=['Response Time (s)'], inplace=True)

    # Handle missing Industry/Customer Segment
    df_csv['Industry'].fillna('Unknown', inplace=True)
    df_csv['Customer Segment'].fillna('Unknown', inplace=True)

    # Drop missing Sentiment Score
    df_csv.dropna(subset=['Sentiment Score'], inplace=True)

    # Convert ID and Name columns to string
    df_csv['User Name'] = df_csv['User Name'].astype(str)
    df_csv['Record ID'] = df_csv['Record ID'].astype(str)

    # Ensure Department is string
    df_csv['Department'] = df_csv['Department'].astype(str)

    # Extract text from PDF
    pdf_text = ""
    pdf_path = 'div_B_escalation_audit.pdf'
    pdf_loaded = False
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pdf_text += page.get_text()
        pdf_loaded = True
    except FileNotFoundError:
        st.warning(f"PDF file not found: {pdf_path}. Assistant might have limited context.")
    except Exception as e:
        st.warning(f"Error reading PDF: {e}. Assistant might have limited context.")

    # Extract text from TXT
    txt_text = ""
    txt_path = 'div_B_ops_report.txt'
    txt_loaded = False
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            txt_text = f.read()
        txt_loaded = True
    except FileNotFoundError:
        st.warning(f"TXT file not found: {txt_path}. Assistant might have limited context.")
    except Exception as e:
        st.warning(f"Error reading TXT: {e}. Assistant might have limited context.")

    return df_csv, pdf_text, txt_text, pdf_loaded, txt_loaded

# Load data once
df_csv, pdf_text, txt_text, pdf_loaded, txt_loaded = load_all_data()

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### 🔧 System Status")
    
    # API Status
    if api_status["configured"]:
        st.markdown("""
        <div class="status-card success">
            <h4>✅ Gemini AI</h4>
            <p>API configured and ready</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-card error">
            <h4>❌ Gemini AI</h4>
            <p>Configuration error: {api_status.get('error', 'Unknown error')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Data Sources")
    
    # CSV Status
    st.markdown(f"""
    <div class="data-source loaded">
        📈 Customer Interactions: {len(df_csv)} records
    </div>
    """, unsafe_allow_html=True)
    
    # PDF Status
    pdf_status_class = "loaded" if pdf_loaded else "missing"
    pdf_icon = "📄" if pdf_loaded else "❌"
    st.markdown(f"""
    <div class="data-source {pdf_status_class}">
        {pdf_icon} Escalation Audit: {'Loaded' if pdf_loaded else 'Missing'}
    </div>
    """, unsafe_allow_html=True)
    
    # TXT Status
    txt_status_class = "loaded" if txt_loaded else "missing"
    txt_icon = "📋" if txt_loaded else "❌"
    st.markdown(f"""
    <div class="data-source {txt_status_class}">
        {txt_icon} Operations Report: {'Loaded' if txt_loaded else 'Missing'}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📈 Quick Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{len(df_csv)}</p>
            <p class="metric-label">Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_sentiment = df_csv['Sentiment Score'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{avg_sentiment:.2f}</p>
            <p class="metric-label">Avg Sentiment</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 💡 Suggested Questions")
    suggestions = [
        "What are the main escalation patterns?",
        "Which departments have the highest response times?",
        "What causes low customer sentiment?",
        "Show me root causes for technical issues",
        "Analyze escalation trends by industry"
    ]
    
    for suggestion in suggestions:
        if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
            st.session_state.suggested_question = suggestion

# --- Main Content Area ---
# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 Root Cause Analysis Assistant</h1>
    <p>Powered by Gemini AI • Analyze customer interactions, escalations, and operational insights</p>
</div>
""", unsafe_allow_html=True)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle suggested questions
if hasattr(st.session_state, 'suggested_question'):
    st.session_state.messages.append({"role": "user", "content": st.session_state.suggested_question})
    delattr(st.session_state, 'suggested_question')

# Display chat messages from history on app rerun
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #6b7280;">
            <h3>👋 Welcome to your Root Cause Analysis Assistant!</h3>
            <p>Ask me anything about customer interactions, escalations, or operational insights.</p>
            <p><em>Try using the suggested questions in the sidebar to get started.</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input
if prompt := st.chat_input("💬 Ask me about root causes, escalations, or customer insights..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing data and generating insights..."):
            context_snippets = []

            if pdf_text:
                context_snippets.append(f"--- Escalation Audit Report Snippet ---\n{pdf_text[:2000]}...\n")
            else:
                context_snippets.append("--- Escalation Audit Report Snippet ---\n(PDF not loaded or empty)\n")

            if txt_text:
                context_snippets.append(f"--- Operations Report Snippet ---\n{txt_text[:1000]}...\n")
            else:
                context_snippets.append("--- Operations Report Snippet ---\n(TXT not loaded or empty)\n")

            csv_context_notes = []
            if "escalation" in prompt.lower() or "root cause" in prompt.lower() or "failure" in prompt.lower():
                escalated_notes_df = df_csv[df_csv['Action Taken'].str.contains('Escalated|Transferred', case=False, na=False)]
                if not escalated_notes_df.empty:
                    csv_context_notes.extend(escalated_notes_df['Interaction Notes'].head(5).tolist())
            if "sentiment" in prompt.lower() or "customer feedback" in prompt.lower():
                low_sentiment_notes_df = df_csv[df_csv['Sentiment Score'] < 0.3]
                if not low_sentiment_notes_df.empty:
                    csv_context_notes.extend(low_sentiment_notes_df['Interaction Notes'].head(3).tolist())

            if csv_context_notes:
                context_snippets.append(f"--- Relevant Customer Interaction Notes ---\n{' '.join(csv_context_notes)[:1500]}...\n")
            else:
                context_snippets.append("--- Relevant Customer Interaction Notes ---\n(No specific notes found for this query)\n")

            full_context = "\n".join(context_snippets)

            model_prompt = f"""
            You are an expert AI assistant tasked with analyzing customer support and operational data to determine root causes.
            You have access to information from:
            1. An escalation audit report (PDF)
            2. An operations report (TXT)
            3. Customer interaction records (CSV)

            Here is the relevant context extracted from these documents:
            {full_context}

            Based *only* on the provided context, answer the following question. If the information is not in the context, state that clearly.

            User's Question: {prompt}
            """

            try:
                if api_status["configured"] and api_status["model"]:
                    response = api_status["model"].generate_content(model_prompt)
                    ai_response = response.text
                else:
                    ai_response = "⚠️ AI model not available due to API configuration issues. Cannot generate response."
            except Exception as e:
                ai_response = f"❌ I apologize, but I encountered an error when trying to generate a response: {e}\n\nPlease check your API key and connection."

        st.markdown(ai_response)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <small>🤖 Powered by Gemini AI • Built with Streamlit • Root Cause Analysis Assistant v1.0</small>
</div>
""", unsafe_allow_html=True)