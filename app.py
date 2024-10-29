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
import google.generativeai as genai
from groq import Groq

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

# Get API tokens from environment variables
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

# Configure Groq
groq_client = Groq(api_key=GROQ_API_KEY)

def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = [
        'HUGGINGFACE_TOKEN',
        'ROBERTA_SENTIMENT_API',
        'DISTILBERT_SENTIMENT_API',
        'LANGUAGE_DETECTION_API',
        'PROMPT_GUARD_API',
        'GEMINI_API_KEY',
        'GROQ_API_KEY'
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        st.error("Please check your .env file and ensure all required variables are set.")
        st.stop()

headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

def query_model(text, endpoint, max_retries=3, retry_delay=20):
    """
    Query the Hugging Face model endpoint with retry mechanism.

    Args:
        text: Text to analyze
        endpoint: Which model endpoint to use
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_ENDPOINTS[endpoint],
                headers=headers,
                json={"inputs": text}
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                error_data = response.json()
                if "estimated_time" in error_data.get("error", ""):
                    # Model is loading
                    if attempt < max_retries - 1:  # Don't show waiting message on last attempt
                        with st.spinner(f"Model is loading... Waiting {retry_delay} seconds. Attempt {attempt + 1}/{max_retries}"):
                            time.sleep(retry_delay)
                        continue
                    else:
                        st.warning(f"Model {endpoint} is still loading. Please try again in a few moments.")
                        return None
            else:
                st.error(f"Error {response.status_code} from {endpoint}: {response.text}")
                return None

        except Exception as e:
            st.error(f"Error querying {endpoint}: {str(e)}")
            return None

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

def get_gemini_suggestions(text, sentiment):
    """Get tweet improvement suggestions from Gemini."""
    try:
        prompt = f"""
        Analyze this tweet and provide specific suggestions:
        Tweet: "{text}"
        Detected sentiment: {sentiment}

        Please provide:
        1. 2-3 ways to improve engagement if the sentiment is negative
        2. 3-5 relevant hashtags that could increase visibility
        3. A short suggestion on the best time to post this kind of content

        Format the response in clear sections.
        """

        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error getting Gemini suggestions: {str(e)}")
        return None

def get_groq_analysis(text, sentiment):
    """Get detailed content analysis from Groq."""
    try:
        prompt = f"""
        As a social media expert, analyze this tweet and provide strategic advice:
        Tweet: "{text}"
        Current sentiment: {sentiment}

        Provide:
        1. Target audience analysis
        2. Content style recommendations
        3. Engagement potential score (1-10)
        4. Specific improvement suggestions

        Keep the response concise and actionable.
        """

        completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=800,
        )

        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting Groq analysis: {str(e)}")
        return None

def analyze_text(text, use_prompt_guard=False):
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

    # Get AI-powered suggestions
    if "roberta_sentiment" in results:
        with st.spinner("Getting Gemini suggestions..."):
            gemini_suggestions = get_gemini_suggestions(text, results["roberta_sentiment"])
            if gemini_suggestions:
                results["gemini_suggestions"] = gemini_suggestions

        with st.spinner("Getting Groq analysis..."):
            groq_analysis = get_groq_analysis(text, results["roberta_sentiment"])
            if groq_analysis:
                results["groq_analysis"] = groq_analysis

    # Check prompt guard only if enabled
    if use_prompt_guard:
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
                     range_y=[0, 1],
                     color='Score',
                     color_continuous_scale='RdYlBu')

        fig.update_layout(
            xaxis_title="Sentiment Category",
            yaxis_title="Confidence Score",
            showlegend=False,
            coloraxis_showscale=False,
            plot_bgcolor='white',
            title_x=0.5
        )

        fig.update_traces(
            hovertemplate="Sentiment: %{x}<br>Score: %{y:.3f}<extra></extra>"
        )

        return fig
    return None

def main():
    # Set page config
    st.set_page_config(
        page_title="Tweet Sentiment Analyzer",
        page_icon="🎭",
        layout="wide"
    )

    # Add custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .sentiment-box {
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Validate environment variables before starting the app
    validate_environment()

    st.title("🎭 Advanced Tweet Sentiment Analyzer")
    st.write("Analyze tweet sentiment using multiple models and detect language and potential risks.")

    # Settings
    with st.expander("Analysis Settings"):
        use_prompt_guard = st.toggle("Enable Content Safety Check", value=False)
        st.info("Content Safety Check uses an additional model to analyze potential risks in the text.")

    # Input section
    col1, col2 = st.columns([3, 1])
    with col1:
        text_input = st.text_area("Enter your tweet:", height=100)
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        analyze_button = st.button("Analyze Tweet", use_container_width=True)
        clear_button = st.button("Clear History", use_container_width=True)

    # Initialize or clear session state
    if clear_button or 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
        st.rerun()

    if analyze_button and text_input:
        with st.spinner("Analyzing..."):
            results = analyze_text(text_input, use_prompt_guard=use_prompt_guard)

            if results:
                st.session_state.analysis_history.append(results)

                # Display results
                st.subheader("Analysis Results")

                # Create two columns for the sentiment models
                col1, col2 = st.columns(2)

                with col1:
                    st.write("### RoBERTa Model")
                    if "roberta_sentiment" in results:
                        sentiment_color = "green" if results['roberta_sentiment'] == "POSITIVE" else "red"
                        st.markdown(
                            f"""<div class='sentiment-box' style='background-color: {sentiment_color}15;'>
                            Overall Sentiment: <strong style='color: {sentiment_color}'>{results['roberta_sentiment']}</strong>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        fig1 = create_sentiment_plot(results['roberta_scores'], "RoBERTa Sentiment Scores")
                        if fig1:
                            st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    st.write("### DistilBERT Model")
                    if "distilbert_sentiment" in results:
                        sentiment_color = "green" if results['distilbert_sentiment'] == "POSITIVE" else "red"
                        st.markdown(
                            f"""<div class='sentiment-box' style='background-color: {sentiment_color}15;'>
                            Overall Sentiment: <strong style='color: {sentiment_color}'>{results['distilbert_sentiment']}</strong>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        fig2 = create_sentiment_plot(results['distilbert_scores'], "DistilBERT Sentiment Scores")
                        if fig2:
                            st.plotly_chart(fig2, use_container_width=True)

                # Additional analysis results
                st.write("### Additional Analysis")
                if "detected_language" in results:
                    st.info(
                        f"📝 Detected Language: **{results['detected_language']['label']}** "
                        f"(Confidence: {results['detected_language']['score']:.2%})"
                    )

                if use_prompt_guard and "prompt_guard" in results:
                    st.write("### Safety Check Results")
                    st.json(results["prompt_guard"])

                # Display AI-powered suggestions
                st.write("### 🤖 AI-Powered Tweet Enhancement Suggestions")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("#### 📈 Gemini Suggestions")
                    if "gemini_suggestions" in results:
                        st.markdown(results["gemini_suggestions"])

                with col2:
                    st.write("#### 🎯 Groq Strategic Analysis")
                    if "groq_analysis" in results:
                        st.markdown(results["groq_analysis"])

        # History section
        if st.session_state.analysis_history:
            st.subheader("Analysis History")
            history_df = pd.DataFrame(st.session_state.analysis_history)
            if not history_df.empty and 'timestamp' in history_df.columns and 'text' in history_df.columns:
                # Format the history dataframe
                display_df = history_df[['timestamp', 'text']].copy()
                display_df.columns = ['Timestamp', 'Tweet']
                st.dataframe(
                    display_df,
                    hide_index=True,
                    use_container_width=True
                )

            # Export option
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Export Analysis History", use_container_width=True):
                    history_json = json.dumps(st.session_state.analysis_history, indent=2)
                    st.download_button(
                        label="Download JSON",
                        file_name="sentiment_analysis_history.json",
                        mime="application/json",
                        data=history_json,
                        use_container_width=True
                    )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            Built with ❤️ using Streamlit and Hugging Face models 🤗
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
