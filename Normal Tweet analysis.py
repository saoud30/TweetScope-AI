import streamlit as st
import requests
import pandas as pd
import time
import plotly.express as px
from datetime import datetime
import json
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# API Configuration from environment variables
API_ENDPOINTS = {
    "sentiment": os.getenv('ROBERTA_SENTIMENT_API'),
    "sentiment_distilbert": os.getenv('DISTILBERT_SENTIMENT_API'),
    "language": os.getenv('LANGUAGE_DETECTION_API'),
    "prompt_guard": os.getenv('PROMPT_GUARD_API')
}

# Get Hugging Face token from environment variable
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = [
        'HUGGINGFACE_TOKEN',
        'ROBERTA_SENTIMENT_API',
        'DISTILBERT_SENTIMENT_API',
        'LANGUAGE_DETECTION_API',
        'PROMPT_GUARD_API'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        st.error("Please check your .env file and ensure all required variables are set.")
        st.stop()

headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

def query_model(text, endpoint):
    """Query the Hugging Face model endpoint."""
    try:
        response = requests.post(
            API_ENDPOINTS[endpoint],
            headers=headers,
            json={"inputs": text}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error {response.status_code} from {endpoint}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error querying {endpoint}: {str(e)}")
        return None

def process_sentiment(result):
    """Process sentiment analysis results."""
    try:
        if not result:
            return None, None
            
        # For RoBERTa model
        if isinstance(result, list) and isinstance(result[0], list):
            sentiment_dict = {}
            for label_score in result[0]:
                label = label_score['label']
                score = label_score['score']
                sentiment_dict[label] = score
            max_sentiment = max(sentiment_dict.items(), key=lambda x: x[1])[0]
            return sentiment_dict, max_sentiment
            
        # For DistilBERT model
        elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            label = result[0]['label']
            score = result[0]['score']
            sentiment_dict = {
                'POSITIVE': score if label == 'POSITIVE' else 1 - score,
                'NEGATIVE': score if label == 'NEGATIVE' else 1 - score
            }
            return sentiment_dict, label
            
        return None, None
        
    except Exception as e:
        st.error(f"Error processing sentiment: {str(e)}")
        return None, None

def process_language_detection(result):
    """Process language detection results."""
    try:
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            return result[0]
        return None
    except Exception as e:
        st.error(f"Error processing language detection: {str(e)}")
        return None

def analyze_text(text):
    """Perform comprehensive analysis on the input text."""
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": text
    }
    
    # Get sentiment from both models
    with st.spinner("Analyzing sentiment with RoBERTa..."):
        roberta_result = query_model(text, "sentiment")
        if roberta_result:
            roberta_scores, roberta_sentiment = process_sentiment(roberta_result)
            if roberta_scores and roberta_sentiment:
                results["roberta_sentiment"] = roberta_sentiment
                results["roberta_scores"] = roberta_scores
    
    with st.spinner("Analyzing sentiment with DistilBERT..."):
        distilbert_result = query_model(text, "sentiment_distilbert")
        if distilbert_result:
            distilbert_scores, distilbert_sentiment = process_sentiment(distilbert_result)
            if distilbert_scores and distilbert_sentiment:
                results["distilbert_sentiment"] = distilbert_sentiment
                results["distilbert_scores"] = distilbert_scores
    
    # Get language detection
    with st.spinner("Detecting language..."):
        language_result = query_model(text, "language")
        if language_result:
            language_info = process_language_detection(language_result)
            if language_info:
                results["detected_language"] = language_info
    
    # Check prompt guard
    with st.spinner("Checking content safety..."):
        prompt_guard_result = query_model(text, "prompt_guard")
        if prompt_guard_result:
            results["prompt_guard"] = prompt_guard_result
    
    return results

def create_sentiment_plot(scores, title):
    """Create a bar plot for sentiment scores."""
    if scores:
        df = pd.DataFrame({
            'Sentiment': list(scores.keys()),
            'Score': list(scores.values())
        })
        fig = px.bar(df, x='Sentiment', y='Score', 
                     title=title,
                     range_y=[0, 1])
        fig.update_layout(
            xaxis_title="Sentiment Category",
            yaxis_title="Confidence Score"
        )
        return fig
    return None

def main():
    # Validate environment variables before starting the app
    validate_environment()
    
    st.title("🎭 Tweet Sentiment Analyzer")
    st.write("Analyze tweet sentiment using multiple models and detect language and potential risks.")

    # Input section
    text_input = st.text_area("Enter your tweet:", height=100)
    analyze_button = st.button("Analyze Tweet")

    # Session state initialization
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

    if analyze_button and text_input:
        with st.spinner("Analyzing..."):
            # Perform analysis
            results = analyze_text(text_input)
            
            if results:
                st.session_state.analysis_history.append(results)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Create two columns for the sentiment models
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### RoBERTa Model")
                    if "roberta_sentiment" in results:
                        st.write(f"Overall Sentiment: {results['roberta_sentiment']}")
                        fig1 = create_sentiment_plot(results['roberta_scores'], "RoBERTa Sentiment Scores")
                        if fig1:
                            st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.write("### DistilBERT Model")
                    if "distilbert_sentiment" in results:
                        st.write(f"Overall Sentiment: {results['distilbert_sentiment']}")
                        fig2 = create_sentiment_plot(results['distilbert_scores'], "DistilBERT Sentiment Scores")
                        if fig2:
                            st.plotly_chart(fig2, use_container_width=True)
                
                # Additional analysis results
                st.write("### Additional Analysis")
                if "detected_language" in results:
                    st.write(f"Detected Language: {results['detected_language']['label']} "
                            f"(Confidence: {results['detected_language']['score']:.2%})")
                
                if "prompt_guard" in results:
                    st.write("### Safety Check")
                    st.json(results["prompt_guard"])

        # History section
        if st.session_state.analysis_history:
            st.subheader("Analysis History")
            history_df = pd.DataFrame(st.session_state.analysis_history)
            if not history_df.empty and 'timestamp' in history_df.columns and 'text' in history_df.columns:
                st.dataframe(history_df[['timestamp', 'text']], hide_index=True)
            
            # Export option
            if st.button("Export Analysis History"):
                history_json = json.dumps(st.session_state.analysis_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    file_name="sentiment_analysis_history.json",
                    mime="application/json",
                    data=history_json
                )

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit and Hugging Face models 🤗")

if __name__ == "__main__":
    main()