import os
import sys
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pickle
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)

from src.rag_engine.llm_generator import RiskExplainerLLM
from src.models.nvdm import NVDM

# Load API key securely
API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="FinRisk-Evo | Risk Terminal", layout="wide", page_icon="🛡️")

# Inject Premium CSS with Google Fonts (Outfit) and Glassmorphism design system
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
    /* Global Styles */
    html, body, .stApp {
        font-family: 'Outfit', sans-serif;
        background-color: #080B10;
        color: #E2E8F0;
    }
    
    /* Premium Title Header */
    .dashboard-header {
        background: linear-gradient(135deg, #1A202C 0%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 25px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #63B3ED 0%, #4FD1C5 50%, #B7791F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-subtitle {
        font-size: 1.0rem;
        color: #718096;
        margin-top: 5px;
        font-weight: 300;
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .metric-value-primary {
        font-size: 2.4rem;
        font-weight: 800;
        color: #F8FAFC;
        background: linear-gradient(135deg, #38BDF8 0%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-value-accent {
        font-size: 2.4rem;
        font-weight: 800;
        color: #F8FAFC;
        background: linear-gradient(135deg, #F59E0B 0%, #EF4444 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    csv_path = os.path.join(PROJECT_ROOT, "data", "master_fres_scores.csv")
    if not os.path.exists(csv_path): 
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if 'category' not in df.columns:
        df['category'] = "STRATEGIC DRIFT"
    if 'similarity' not in df.columns:
        df['similarity'] = 0.5
    return df

def get_nvdm_topics():
    try:
        nvdm_weights = os.path.join(PROJECT_ROOT, "src", "models", "saved_weights", "nvdm_weights.pt")
        vectorizer_path = os.path.join(PROJECT_ROOT, "src", "models", "saved_weights", "nvdm_vectorizer.pkl")
        
        if not os.path.exists(nvdm_weights) or not os.path.exists(vectorizer_path):
            return ["NVDM weights/vectorizer not found. Process reports to train."]
            
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        vocab = vectorizer.get_feature_names_out()
        vocab_size = len(vocab)
        
        model = NVDM(vocab_size=vocab_size, n_topics=20)
        model.load_state_dict(torch.load(nvdm_weights, map_location='cpu'))
        model.eval()
        
        decoder_weights = model.decoder.weight.data.numpy()
        topics = []
        for k in range(20):
            top_word_indices = decoder_weights[:, k].argsort()[::-1][:8]
            top_words = [vocab[i] for i in top_word_indices]
            topics.append(f"Topic {k+1:02d}: " + ", ".join(top_words))
        return topics
    except Exception as e:
        return [f"Could not load NVDM topics: {e}"]

def main():
    df = load_data()
    llm = RiskExplainerLLM(API_KEY) if API_KEY else None

    # --- SIDEBAR: PDF Operations & Configuration ---
    with st.sidebar:
        st.markdown("<h2 style='color:#38BDF8;'>FinRisk-Evo 🛡️</h2>", unsafe_allow_html=True)
        st.markdown("Annual Report Narrative Intelligence System")
        st.markdown("---")
        
        st.markdown("### 📂 Ingest Corporate Reports")
        b_file = st.file_uploader("1. Baseline Document (e.g. FY24)", type="pdf")
        t_file = st.file_uploader("2. Target Document (e.g. FY25)", type="pdf")
        
        st.markdown("### ⚙️ Pipeline Control")
        if st.button("🚀 Run Temporal Alignment", type="primary", use_container_width=True):
            if b_file and t_file:
                with st.spinner("Executing full pipeline... Parsing layout geometric bounds, selecting TNCRM masking text, learning topics, and scoring temporal drift..."):
                    raw_dir = os.path.join(PROJECT_ROOT, "data", "01_raw")
                    os.makedirs(raw_dir, exist_ok=True)
                    
                    b_path = os.path.join(raw_dir, b_file.name)
                    t_path = os.path.join(raw_dir, t_file.name)
                    
                    with open(b_path, "wb") as f: f.write(b_file.getbuffer())
                    with open(t_path, "wb") as f: f.write(t_file.getbuffer())
                    
                    from src.run_pipeline import execute_full_pipeline
                    execute_full_pipeline(b_path, t_path)
                    st.success("Pipeline executed successfully!")
                    st.rerun()
            else:
                st.error("Please upload both baseline and target PDFs.")
        
        st.markdown("---")
        st.markdown("### 📑 RAG Summary Explainer")
        st.info("Uses Google Gemini 2.5 Flash to synthesize temporal risk drift using retrieval-augmented citations.")

    # --- HEADER PANEL ---
    st.markdown("""
    <div class="dashboard-header">
        <div class="header-title">🛡️ FinRisk-Evo: Financial Risk Intelligence</div>
        <div class="header-subtitle">Quantifying the Trajectory of Unstructured Annual Report Risk Drift using Numerically-Enhanced Hybrid Transformers (VAE + BiGRU-Attention)</div>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.info("👈 Upload two annual report PDFs in the sidebar to run the temporal risk intelligence pipeline.")
        return

    # --- METRICS TILES ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Paragraphs Processed</div>
            <div class="metric-value-primary">{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        critical_count = len(df[df['fres'] > 80])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Escalated Risk Alerts</div>
            <div class="metric-value-accent">{critical_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        max_fres = df['fres'].max() if not df.empty else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Peak Risk Escalation Score</div>
            <div class="metric-value-accent">{max_fres:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        mean_conf = df['confidence'].mean() if not df.empty else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mean Inference Confidence</div>
            <div class="metric-value-primary">{mean_conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- MAIN ANALYSIS TABS ---
    tab_matrix, tab_topology, tab_diagnostics, tab_rag = st.tabs([
        "🛡️ Risk Escalation Ledger", 
        "📊 3D Risk Topology", 
        "🧪 Neural Model Diagnostics", 
        "🕵️ News Validator & RAG explainer"
    ])

    # --- TAB 1: RISK LEDGER ---
    with tab_matrix:
        st.markdown("### 🗃️ Temporal Risk Ledger")
        
        # Filtering controls
        col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
        with col_f1:
            cats = ["ALL"] + sorted(list(df['category'].unique()))
            selected_cat = st.selectbox("Filter by Risk Category", cats)
        with col_f2:
            min_fres = st.slider("Minimum FRES Score", 0, 100, 0)
        with col_f3:
            search_query = st.text_input("🔍 Search Risk Texts", "")
            
        # Apply filters
        filtered_df = df.copy()
        if selected_cat != "ALL":
            filtered_df = filtered_df[filtered_df['category'] == selected_cat]
        filtered_df = filtered_df[filtered_df['fres'] >= min_fres]
        if search_query:
            filtered_df = filtered_df[
                filtered_df['fy25'].str.contains(search_query, case=False) | 
                filtered_df['fy24'].str.contains(search_query, case=False)
            ]
            
        st.markdown(f"Showing **{len(filtered_df)}** flagged escalations after filters.")
        
        # Display risks list
        for i, row in filtered_df.head(20).iterrows():
            fres_val = row['fres']
            color_badge = "#EF4444" if fres_val > 75 else "#F59E0B" if fres_val > 45 else "#10B981"
            
            with st.container():
                st.markdown(f"""
                <div style="background:rgba(30, 41, 59, 0.25); border:1px solid rgba(255,255,255,0.05); border-radius:10px; padding:18px; margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="background:{color_badge}33; color:{color_badge}; border:1px solid {color_badge}; border-radius:20px; padding:3px 12px; font-size:0.8rem; font-weight:600; text-transform:uppercase;">
                            {row['category']}
                        </span>
                        <span style="font-size:1.1rem; font-weight:800; color:{color_badge};">
                            FRES: {row['fres']:.1f}
                        </span>
                    </div>
                    <div style="font-size:0.9rem; color:#94A3B8; margin-bottom:4px;"><b>FY24 (Baseline Context):</b></div>
                    <div style="font-size:0.95rem; color:#FFFFFF; background:rgba(15, 23, 42, 0.85); padding:10px 14px; border-radius:6px; margin-bottom:8px; border-left:4px solid #64748B; line-height:1.5;">
                        {row['fy24']}
                    </div>
                    <div style="font-size:0.9rem; color:#F87171; margin-bottom:4px;"><b>FY25 (Target Escalation):</b></div>
                    <div style="font-size:0.95rem; color:#FFFFFF; background:rgba(127, 29, 29, 0.45); padding:10px 14px; border-radius:6px; border-left:4px solid #EF4444; line-height:1.5;">
                        {row['fy25']}
                    </div>
                    <div style="font-size:0.8rem; color:#718096; margin-top:8px;">
                        Attention: <b>{row['attn_25']:.1f}%</b> | Bayesian Confidence: <b>{row['confidence']:.1f}%</b> | Topic Dispersion: <b>{row['dispersion']:.2f}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # --- TAB 2: RISK TOPOLOGY ---
    with tab_topology:
        st.markdown("### 📊 Multi-Dimensional Risk Space")
        
        c_top1, c_top2 = st.columns([2, 1])
        with c_top1:
            st.markdown("#### 3D Neural Scatter Topology")
            fig = go.Figure(data=[go.Scatter3d(
                x=df['attn_25'], y=df['fres'], z=df['dispersion'],
                mode='markers', 
                marker=dict(
                    size=df['confidence']/8, 
                    color=df['fres'], 
                    colorscale='Jet',
                    opacity=0.8,
                    line=dict(width=1, color='rgba(255,255,255,0.1)')
                ),
                text=df['category'] + "<br>Score: " + df['fres'].round(1).astype(str) + "%"
            )])
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='Attention Target %', backgroundcolor="rgb(8,11,16)", gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(title='FRES (Risk)', backgroundcolor="rgb(8,11,16)", gridcolor="rgba(255,255,255,0.05)"),
                    zaxis=dict(title='NVDM Latent Dispersion', backgroundcolor="rgb(8,11,16)", gridcolor="rgba(255,255,255,0.05)"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, b=0, t=0), 
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with c_top2:
            st.markdown("#### Risk Distribution by Category")
            cat_counts = df['category'].value_counts().reset_index()
            cat_counts.columns = ['Category', 'Count']
            fig_bar = px.bar(
                cat_counts, 
                x='Count', 
                y='Category', 
                orientation='h', 
                color='Count',
                color_continuous_scale='Viridis',
                template='plotly_dark'
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Paragraphs Flagged",
                yaxis_title=""
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- TAB 3: MODEL DIAGNOSTICS ---
    with tab_diagnostics:
        st.markdown("### 🧪 Model Coherence & Diagnostics")
        
        cd1, cd2 = st.columns([1, 1])
        with cd1:
            st.markdown("#### Variational Topic Coherence (NVDM)")
            st.markdown("The Neural Variational Document Model (NVDM) projects paragraph BoW representations into a 20-dimensional continuous topic space. Below are the decoded top vocabulary keywords associated with the learned topics:")
            
            topics = get_nvdm_topics()
            for topic in topics:
                st.markdown(f"`{topic}`")
                
        with cd2:
            st.markdown("#### Bayesian Inference Confidence (BiGRU)")
            st.markdown("Our Gated Recurrent Unit incorporates Monte Carlo Dropout. Standard deviations over 10 forward passes are converted into Bayesian Confidence. High-risk predictions with low confidence (high std) represent epistemic uncertainty and are mathematically scaled down:")
            
            # Plotly histogram of confidence levels
            fig_hist = px.histogram(
                df, 
                x='confidence', 
                nbins=15, 
                title="Distribution of Monte Carlo Inference Confidence",
                color_discrete_sequence=['#4FD1C5'],
                labels={'confidence': 'Bayesian Confidence %'}
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                template='plotly_dark',
                height=300
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.markdown("#### Technical Key Contributions for Research Paper")
            st.markdown("""
            - **Geometric Page Boundary Filtering**: Discards 98% of layout noise, ensuring that headers, footers, and page numbers do not trigger false semantic risk drift alarms.
            - **Targeted Masking MLM (TNCRM)**: Unlike random MLM, it trains FinBERT's attention parameters specifically on numbers and change-related terms, solving financial 'scalar blindness'.
            - **Continuous Semantic Projection**: Mathematically projects paragraph embeddings onto a defined risk vs mitigation axis to dynamically scale risk scores without using rigid keyword check lists.
            - **Bayesian Epistemic Gating**: Fuses BiGRU outputs with uncertainty estimates derived from Monte Carlo dropout.
            """)

    # --- TAB 4: RAG AUDITOR ---
    with tab_rag:
        st.markdown("### 🕵️ Retrieval-Augmented Explainer & News Validation")
        st.markdown("Select a flagged temporal risk below to generate an executive explanation using Gemini and validate the report claims against external mock news feeds:")
        
        selected_risk_idx = st.selectbox("Select a Risk Node to Audit", df.index)
        
        if selected_risk_idx is not None and selected_risk_idx < len(df):
            row = df.loc[selected_risk_idx]
            
            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                st.markdown("#### Selected Risk Node")
                st.markdown(f"**Category:** `{row['category']}` | **FRES Score:** `{row['fres']:.1f}/100` | **Confidence:** `{row['confidence']:.1f}%`")
                st.markdown(f"**Baseline FY24:**")
                st.info(row['fy24'])
                st.markdown(f"**Target FY25:**")
                st.error(row['fy25'])
                
            with col_b2:
                st.markdown("#### AI Executive Risk Summary (RAG)")
                if st.button("✨ Synthesize Summary", key="btn_rag_sum"):
                    if llm is None:
                        st.warning("Set GEMINI_API_KEY in the environment to enable AI summaries.")
                    else:
                        with st.spinner("Querying Gemini 2.5 Flash..."):
                            summary = llm.generate_executive_summary(row['fy24'], row['fy25'], row['fres'])
                            st.success(summary)
                        
                st.markdown("#### Neural News Validator (MNLI)")
                # Simulate News Validator
                st.markdown("Compare reports with market news. Input simulated external market news below:")
                news_input = st.text_area(
                    "Simulated Market News", 
                    "Mahindra has experienced supply delays in parts due to domestic shipping bottlenecks, affecting tractor volumes.",
                    height=100
                )
                
                if st.button("🔬 Validate Claim", key="btn_validate"):
                    with st.spinner("Evaluating cross-document entailment..."):
                        from src.rag_engine.neural_validator import NeuralMarketValidator
                        try:
                            validator = NeuralMarketValidator()
                            validation_score = validator.validate_risk(row['fy25'], news_input)
                            st.metric("Entailment Verification Probability", f"{validation_score * 100:.1f}%")
                            if validation_score > 0.65:
                                st.success("✅ **External News ENTAILS and confirms the risk.**")
                            elif validation_score > 0.35:
                                st.warning("⚠️ **External News is neutral / partially related.**")
                            else:
                                st.error("❌ **External News contradicts or fails to support the claim.**")
                        except Exception as e:
                            st.error(f"Validation Error: {e}")

if __name__ == "__main__":
    main()