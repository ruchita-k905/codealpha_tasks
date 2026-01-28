import streamlit as st
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FAQ Data
faq_data = {
    "What is CodeAlpha?": "CodeAlpha is a technology-focused organization offering internships and training programs.",
    "What is the duration of the internship?": "The internship duration is one month.",
    "Which domain is this internship related to?": "This internship is related to Artificial Intelligence.",
    "Is the internship paid?": "This internship is primarily a learning-based opportunity.",
    "What skills will I gain?": "You will gain practical knowledge in AI concepts, Python, and real-world problem solving.",
    "Is this internship beneficial for students?": "Yes, it helps students gain hands-on experience and industry exposure."
}

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Preprocess FAQs
questions = list(faq_data.keys())
processed_questions = [preprocess(q) for q in questions]

# Vectorization
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

# Streamlit UI
st.title("🤖 FAQ Chatbot")
st.write("Ask any question related to CodeAlpha AI Internship")

user_query = st.text_input("You:")

if st.button("Ask"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        processed_query = preprocess(user_query)
        query_vector = vectorizer.transform([processed_query])

        similarity = cosine_similarity(query_vector, faq_vectors)
        index = similarity.argmax()
        score = similarity[0][index]

        if score < 0.2:
            st.error("Sorry, I couldn't understand your question.")
        else:
            st.success(f"Bot: {faq_data[questions[index]]}")

# To run the file
# python -m streamlit run faq_chatbot.py