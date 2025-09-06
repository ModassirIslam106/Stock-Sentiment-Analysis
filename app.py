import streamlit as st
import pandas as pd
import joblib     # for loding the saved pipeline
from preprocessor import HeadlinePreprocessor

# Load the Trained Pipeline
model = joblib.load("news_model.pkl")

#  Streamlit Page Config

st.set_page_config(
    page_title="Stock Sentiment Analyzer", 
    page_icon="📰", 
    layout="wide"
)

# Title & Description
st.title("Stock Market News Sentiment Analyzer")
st.markdown(
    """
    This app predicts whether **news headlines** will have a 
    **Positive 📈** or **Negative 📉** impact on the stock market.
    
    🔹 The model is trained on financial news data using a **Random Forest Classifier**.  
    🔹 Enter up to 25 headlines (like in your dataset).  
    🔹 The app will preprocess automatically using the saved pipeline.  
    """
)

# 📌 Input Section (Form)
with st.form("news_form"):
    st.write(" Enter up to 25 headlines")
    headlines = []

    # Collect 25 headlines (user can leave some empty)
    for i in range(25):
        text = st.text_input(f"Headline {i+1}", "")
        headlines.append(text)

    # Submit Button
    submitted = st.form_submit_button("🔮 Predict Sentiment")


# 📌 Prediction
if submitted:
    # Convert user input into DataFrame (same structure as training)
    input_df = pd.DataFrame([headlines])

    # Make prediction using pipeline
    prediction = model.predict(input_df)[0]

    # Convert numeric label into sentiment
    sentiment = " Positive Impact" if prediction == 1 else " Negative Impact"

    # Display result
    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(sentiment)
    else:
        st.error(sentiment)


#  Footer
st.markdown(
    """
    ---
    ✅ Built with **Streamlit** & **Random Forest**  
    🚀 Model trained on Kaggle Stock Sentiment Dataset  
    """
)