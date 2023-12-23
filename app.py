import nltk
import random
import string
import warnings
from flask import Flask, render_template, request

warnings.filterwarnings('ignore')

# Your existing code

app = Flask(__name__)

# Reading the text file and preprocessing

def read_text_file(file_path):
    with open(file_path, 'r', errors='ignore') as file:
        raw_text = file.read().lower()
    return raw_text

raw_text = read_text_file('C:\\Users\\Lenovo\\.spyder-py3\\templates\\textfile.txt')

sent_tokens = nltk.sent_tokenize(raw_text)
word_tokens = nltk.word_tokenize(raw_text)

sent_tokens = sent_tokens[:4]
word_tokens = word_tokens[:4]

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "hai","halo","Hallo")
GREETING_RESPONSES = ["sup", "hey", "iyyah", "hi there", "hello", "hallo","knffh","I am glad! you are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    chatbot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        chatbot_response = chatbot_response + "I am sorry! I don't understand you"
        return chatbot_response
    else:
        chatbot_response = chatbot_response + sent_tokens[idx]
        return chatbot_response

# Flask routes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_input = request.form['user_input']
    user_response = user_input.lower()

    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            return "You're welcome!"
        else:
            if greeting(user_response) is not None:
                return greeting(user_response)
            else:
                response_text = response(user_response)
                sent_tokens.remove(user_response)
                return  response_text
    else:
        return "Bye! Have a great time!"

if __name__ == "__main__":
    app.run(debug=True)
