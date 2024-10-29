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
from PIL import Image
import io
import base64

def process_sentiment(result):
    """
    Process sentiment analysis results from the model.
    
    Args:
        result: List of dictionaries containing sentiment scores
        
    Returns:
        tuple: (scores_dict, overall_sentiment)
    """
    try:
        # Extract scores from the first result
        scores = {
            'POSITIVE': result[0][0]['score'],
            'NEGATIVE': result[0][1]['score']
        }
        
        # Determine overall sentiment
        overall_sentiment = 'POSITIVE' if scores['POSITIVE'] > scores['NEGATIVE'] else 'NEGATIVE'
        
        return scores, overall_sentiment
    except (IndexError, KeyError, TypeError) as e:
        st.error(f"Error processing sentiment results: {str(e)}")
        return None, None

def process_language_detection(result):
    """
    Process language detection results from the model.
    
    Args:
        result: List of dictionaries containing language detection scores
        
    Returns:
        dict: Dictionary containing detected language and confidence score
    """
    try:
        # Get the most likely language from the first result
        detected = result[0][0]
        return {
            'label': detected['label'],
            'score': detected['score']
        }
    except (IndexError, KeyError, TypeError) as e:
        st.error(f"Error processing language detection results: {str(e)}")
        return None

def create_sentiment_plot(scores, title):
    """
    Create a bar plot for sentiment scores using plotly.
    
    Args:
        scores: Dictionary containing sentiment scores
        title: Title for the plot
        
    Returns:
        plotly.graph_objects.Figure: The created plot
    """
    try:
        df = pd.DataFrame({
            'Sentiment': list(scores.keys()),
            'Score': list(scores.values())
        })
        
        fig = px.bar(
            df,
            x='Sentiment',
            y='Score',
            title=title,
            color='Sentiment',
            color_discrete_map={
                'POSITIVE': '#00CC96',
                'NEGATIVE': '#EF553B'
            }
        )
        
        fig.update_layout(
            showlegend=False,
            yaxis_range=[0, 1],
            yaxis_title="Confidence Score",
            xaxis_title="",
            title_x=0.5
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating sentiment plot: {str(e)}")
        return None

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

# Update Gemini configuration
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')
# Update to use gemini-1.5-pro for vision as well
gemini_vision_model = genai.GenerativeModel('gemini-1.5-flash')

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
                headers={"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"},
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

def analyze_image(image):
    """Analyze image content using Gemini Vision."""
    try:
        prompt = """
        Analyze this image for a tweet. Provide:
        1. A brief description of what's in the image
        2. The overall mood/sentiment of the image
        3. 3-5 relevant hashtags that would work well
        4. Suggestions for caption text
        5. Best practices for sharing this type of image

        Format the response in clear sections.
        """

        # Convert PIL Image to bytes for Gemini
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes = image_bytes.getvalue()

        response = gemini_vision_model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])
        return response.text
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None

def get_multimodal_suggestions(text, image_analysis, sentiment):
    """Get combined suggestions for text and image content."""
    try:
        prompt = f"""
        Analyze this tweet content and provide holistic suggestions:

        Tweet Text: "{text}"
        Image Analysis: {image_analysis}
        Text Sentiment: {sentiment}

        Please provide:
        1. How well the text and image complement each other (synergy analysis)
        2. 5 relevant hashtags that combine themes from both text and image
        3. Suggestions to improve the text-image combination
        4. Best time to post this type of content
        5. Target audience analysis

        Format the response in clear sections.
        """

        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error getting multimodal suggestions: {str(e)}")
        return None

def get_groq_engagement_strategy(text, image_analysis, sentiment):
    """Get strategic engagement advice from Groq for multimodal content."""
    try:
        prompt = f"""
        As a social media expert, analyze this multimodal tweet content:

        Tweet Text: "{text}"
        Image Content: {image_analysis}
        Current Sentiment: {sentiment}

        Provide a strategic analysis including:
        1. Content optimization strategy (text-image balance)
        2. Engagement potential score (1-10) with explanation
        3. Platform-specific recommendations (Twitter/X specific)
        4. A/B testing suggestions for different caption variations
        5. Call-to-action recommendations

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

def analyze_content(text, image_file=None, use_prompt_guard=False):
    """Perform comprehensive analysis on the input text and image."""
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": text
    }

    # Analyze image if provided
    if image_file is not None:
        with st.spinner("Analyzing image..."):
            image = Image.open(image_file)
            image_analysis = analyze_image(image)
            if image_analysis:
                results["image_analysis"] = image_analysis

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

    # Get multimodal AI suggestions if we have both text and image
    if "roberta_sentiment" in results:
        image_analysis_text = results.get("image_analysis", "No image provided")

        with st.spinner("Getting Gemini multimodal suggestions..."):
            multimodal_suggestions = get_multimodal_suggestions(
                text,
                image_analysis_text,
                results["roberta_sentiment"]
            )
            if multimodal_suggestions:
                results["gemini_suggestions"] = multimodal_suggestions

        with st.spinner("Getting Groq engagement strategy..."):
            groq_strategy = get_groq_engagement_strategy(
                text,
                image_analysis_text,
                results["roberta_sentiment"]
            )
            if groq_strategy:
                results["groq_analysis"] = groq_strategy

    # Check prompt guard if enabled
    if use_prompt_guard:
        with st.spinner("Checking content safety..."):
            prompt_guard_result = query_model(text, "prompt_guard")
            if prompt_guard_result:
                results["prompt_guard"] = prompt_guard_result

    return results

def main():
    # Set page config
    st.set_page_config(
        page_title="TweetScope AI",
        page_icon="🔭",
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
        .image-preview {
            max-width: 300px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Validate environment variables
    validate_environment()

    st.title("🔭 TweetScope AI")
    st.write("Analyze tweets with text and images for optimal engagement.")

    # Settings
    with st.expander("Analysis Settings"):
        use_prompt_guard = st.toggle("Enable Content Safety Check", value=False)
        st.info("Content Safety Check uses an additional model to analyze potential risks in the content.")

    # Input section
    col1, col2 = st.columns([3, 1])
    with col1:
        text_input = st.text_area("Enter your tweet:", height=100)
        uploaded_file = st.file_uploader("Upload an image (optional)", type=['png', 'jpg', 'jpeg'])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        analyze_button = st.button("Analyze Tweet", use_container_width=True)
        clear_button = st.button("Clear History", use_container_width=True)

    # Initialize or clear session state
    if clear_button or 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
        st.rerun()

    if analyze_button and (text_input or uploaded_file):
        with st.spinner("Analyzing..."):
            results = analyze_content(text_input, uploaded_file, use_prompt_guard)

            if results:
                st.session_state.analysis_history.append(results)

                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "Content Analysis", "Strategic Insights"])

                with tab1:
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

                with tab2:
                    if "image_analysis" in results:
                        st.write("### 📸 Image Analysis")
                        st.markdown(results["image_analysis"])

                    if "detected_language" in results:
                        st.write("### 🌐 Language Detection")
                        st.info(
                            f"Detected Language: **{results['detected_language']['label']}** "
                            f"(Confidence: {results['detected_language']['score']:.2%})"
                        )

                with tab3:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("### 🤖 Gemini Suggestions")
                        if "gemini_suggestions" in results:
                            st.markdown(results["gemini_suggestions"])

                    with col2:
                        st.write("### 🎯 Groq Strategic Analysis")
                        if "groq_analysis" in results:
                            st.markdown(results["groq_analysis"])

                if use_prompt_guard and "prompt_guard" in results:
                    st.write("### ⚠️ Safety Check Results")
                    st.json(results["prompt_guard"])

        # History section
        if st.session_state.analysis_history:
            st.write("### 📜 Analysis History")
            history_df = pd.DataFrame(st.session_state.analysis_history)
            if not history_df.empty and 'timestamp' in history_df.columns and 'text' in history_df.columns:
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
                        file_name="tweet_analysis_history.json",
                        mime="application/json",
                        data=history_json,
                        use_container_width=True
                    )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            Built with ❤️ using Streamlit, Hugging Face, Gemini, and Groq
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
