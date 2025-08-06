import streamlit as st
import pandas as pd

# --- Mock Data ---
# In the future, this data will come from your AI Rosetta Stone engine.
mock_data = {
    "overall_compliance": 98.2,
    "compliance_snippet": "All rules utilizing the feature 'applicant_age' were tested... maximum observed deviation is 1.2%...",
    "articles": [
        {"name": "EU AI Act - Article 10", "status": "Verified"},
        {"name": "EU AI Act - Article 14", "status": "Warning"},
        {"name": "EU AI Act - Article 17", "status": "Verified"},
        {"name": "EU AI Act - Article 19", "status": "Violation"}
    ]
}

# --- Dashboard UI ---

st.set_page_config(layout="wide", page_title="AI Rosetta Stone")

# Header
st.title("AI Rosetta Stone Dashboard")
st.markdown("---")

# Main content area with columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Overview")
    # Display the big percentage metric
    st.metric(label="Compliance Score", value=f"{mock_data['overall_compliance']}%")
    
    st.subheader("Compliance Verified")
    # Display the text snippet
    st.info(mock_data['compliance_snippet'])

with col2:
    st.header("Articles")
    # Loop through the articles and display them
    for article in mock_data['articles']:
        status = article['status']
        name = article['name']
        
        if status == "Verified":
            st.success(f"✅ {name}: **{status}**")
        elif status == "Warning":
            st.warning(f"⚠️ {name}: **{status}**")
        elif status == "Violation":
            st.error(f"❌ {name}: **{status}**")

st.markdown("---")