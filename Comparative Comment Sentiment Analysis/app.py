import streamlit as st
import requests
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import os
from groq import Groq

# Loading Environment Variables
load_dotenv()

# API Configs
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq API client
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except:
    groq_client = None

def classify_sentiment(score):
    if score>=0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def compare_sentiments(comment1, comment2):
    analyzer = SentimentIntensityAnalyzer()
    
    score1 = analyzer.polarity_scores(comment1)
    score2 = analyzer.polarity_scores(comment2)

    label1 = classify_sentiment(score1['compound'])
    label2 = classify_sentiment(score2['compound'])

    return score1,score2,label1,label2
    


# Getting AI Analysis from groq
def get_groq_analysis(comment1, comment2):
    if not groq_client:
        return "Groq API not Available. Please check API key."
    try:
        prompt = f"""Compare and analyze the following:
        Original Tweet: {comment1}
        Reply: {comment2}
        Determine whether the reply agrees, disagrees, or is neutral with respect to the original.
        Return the output in the following format:
        Label: <Agreement/Disagreement/Neutral>\n
        Explanation: <your reasoning in one or two sentences> """
        chat_completion = groq_client.chat.completions.create(
            messages = (
                {"role": "system", "content": "You're an assistant that compares two texts for sentiment and provides insight."},
                {"role": "user", "content": prompt}
            ),
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=500
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error getting Groq Analysis {e}"

def main():
    st.set_page_config(
        page_title="Comment Sentiment Analysis",
        layout="wide"
    )
    st.title("Comment Sentiment Analysis")
    st.markdown("To analyze the sentiment of a user comment and classify as positive, negative or neutral")

    col1, col2 = st.columns(2)
    with col1:
        text1 = st.text_area("Original Tweet")
    with col2:
        text2 = st.text_area("Reply to Tweet")
    
    if st.button("Run analysis"):
        if not text1 or not text2:
            st.warning("Please fill in both inputs.")
        else:
            score1,score2,label1,label2 = compare_sentiments(text1,text2)
            st.subheader("Sentiment Classification")
            col3, col4 = st.columns(2)
            with col3:
                st.metric(label="Original Tweet Sentiment", value=label1)
                st.metric(label="Compound Score", value=f"{score1['compound']:.2f}")
            with col4:
                st.metric(label="Reply Sentiment", value=label2)
                st.metric(label="Compound Score", value=f"{score2['compound']:.2f}")
            
            st.subheader("GROQ Insight")
            groq_summary=get_groq_analysis(text1,text2) 
            st.info(groq_summary)

            # Bar Chart 
            fig = go.Figure(data=[
                go.Bar(name='Original Tweet', x=['Negative','Neutral','Positive'], y=[score1['neg'],score1['neu'],score1['pos']],marker_color='green'),
                go.Bar(name="Reply Tweet",x=['Negative','Neutral','Positive'],y=[score2['neg'],score2['neu'],score2['pos']],marker_color = 'blue')
            ])
            fig.update_layout(
                title="Sentiment Breakdown",
                xaxis_title="Sentiment Type",
                yaxis_title="Intensity",
                barmode='group'
            )
            st.subheader("Sentiment Breakdown Chart")
            st.plotly_chart(fig, use_container_width=True)
            
            # Final Verdict
            label_line = groq_summary.splitlines()[0]
            label_value = label_line.replace("Label:", "").strip().lower()
            st.subheader("Final Verdict")
            if label_value == "agreement":
                st.success("The reply is in agreement with the original tweet.")
            elif label_value == "disagreement":
                st.error("The reply disagrees with the original tweet.")
            elif label_value == "neutral":
                st.info("The reply is neutral with respect to the original tweet.")
            else:
                st.warning("Unable to determine final verdict.")


if __name__=="__main__":
    main()
