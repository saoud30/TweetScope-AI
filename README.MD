<h1 align="center">🔭 TweetScope AI</h1>

An intelligent tweet analysis platform that observes, analyzes, and enhances your social media content through multiple lenses. Powered by advanced AI models, TweetScope AI provides comprehensive insights for optimal social media engagement.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-FF4B4B)

## 🌟 Overview

TweetScope AI is your advanced social media observatory, providing deep insights into tweet content through:
- Dual-model sentiment analysis
- Advanced image understanding
- Language detection
- Strategic engagement insights
- Content safety verification

## ✨ Features

- **Dual Sentiment Analysis**: Leverages both RoBERTa and DistilBERT models for robust sentiment detection
- **Image Content Analysis**: Uses Google's Gemini Vision for detailed image content understanding
- **Language Detection**: Automatically identifies the language of tweet text
- **Content Safety Checking**: Optional prompt guard for content moderation
- **Strategic Insights**: 
  - Gemini-powered multimodal content suggestions
  - Groq-powered engagement strategy analysis
  - Platform-specific optimization tips
- **Interactive Visualization**: Beautiful charts and graphs for sentiment analysis
- **Analysis History**: Track and export your content analysis history
- **Responsive UI**: Clean, modern interface with dark/light mode support

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- API keys for:
  - Hugging Face
  - Google Gemini
  - Groq
  - Access to specific model endpoints

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Saoud30/TweetScope-AI
cd TweetScope-AI 
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file jsut like the `.env.example` file in the project root and add your API keys:
```env
HUGGINGFACE_TOKEN=your_token_here
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
ROBERTA_SENTIMENT_API=your_endpoint_here
DISTILBERT_SENTIMENT_API=your_endpoint_here
LANGUAGE_DETECTION_API=your_endpoint_here
PROMPT_GUARD_API=your_endpoint_here
```

### Running the Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your web browser to use the application.

## 📊 Features in Detail

### Sentiment Analysis
- Dual model approach for more accurate sentiment detection
- Visual representation of sentiment scores
- Confidence metrics for each analysis

### Image Analysis
- Object and scene detection
- Mood/sentiment analysis
- Hashtag suggestions
- Caption recommendations

### Strategic Insights
- Text-image synergy analysis
- Posting time recommendations
- Target audience insights
- A/B testing suggestions
- Engagement optimization tips

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Models**: 
  - Google Gemini (Text & Vision)
  - Groq Mixtral
  - Hugging Face Transformers
- **Visualization**: Plotly
- **Image Processing**: Pillow
- **Others**: Pandas, Requests, Python-dotenv

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Streamlit for the amazing web app framework
- Hugging Face for transformer models
- Google for Gemini AI
- Groq for their LLM API

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

Project Link: [https://github.com/Saoud30/TweetScope-AI](https://github.com/Saoud30/TweetScope-AI)
- Email: mohdsarimansari30.com
- Twitter: [@Shazyansar](https://twitter.com/Shazyansar)


<div align="center">
  Made with ❤️ by the Shazy
</div>