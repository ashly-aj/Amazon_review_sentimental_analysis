import streamlit as st
import joblib
import base64
import pandas as pd


#load the pre-trained model and vectorizer
model=joblib.load("sentiment_model.pkl")
vectorizer=joblib.load("vectorizer.pkl")

#background image
def get_base64_image(file_path):
    with open(file_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    return encoded


# Convert the local image
image_path = "amazon.jpg"  # Replace with your local image path
base64_image = get_base64_image(image_path)

# Apply CSS to set the background
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_image}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
# img=Image.open('alzheimers.jpg')
# st.image(img,width=450)



#Streamlit UI
st.title("Amazon Product Reviews Sentiment Analysis")
st.markdown("Enter an Amazon product review below to get its sentiment (Positive/Negative)")

#user input (Amazon review text)
review_text=st.text_area("Enter Review", "Type your review here..")

#function to predict sentiment
def predict_sentiment(review):
    #preprocess and vectorize the input review
    review_vectorized=vectorizer.transform([review])
    #predict sentiment
    sentiment=model.predict(review_vectorized)[0]
    #return sentiment
    return sentiment

#button to trigger predcition
if st.button("Analyze Sentiment"):
    if review_text:
        sentiment=predict_sentiment(review_text)
        sentiment_label="Positive" if sentiment==1 else "Negative"
        st.write(f"Sentiment of the review: **{sentiment_label}**")
    else:
        st.warning("Please enter a review text!")
