import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Send a GET request to the Wikipedia article
url = "https://en.wikipedia.org/wiki/Machine_learning"
response = requests.get(url)

# Parse the HTML content of the page with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Remove all script and style elements
for script in soup(["script", "style"]):
    script.decompose()

# Get the text from the HTML content
text = soup.get_text()

# Break the text into lines and remove leading and trailing space on each
lines = (line.strip() for line in text.splitlines())

# Break multi-headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

# Remove special characters and digits
text =''.join(chunk for chunk in chunks if chunk)

# Tokenize the text
tokens = word_tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Convert the filtered tokens back into a string
text =''.join(filtered_tokens)

# Split the text into training and testing sets
train_text, test_text = train_test_split([text], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the training text and transform both the training and testing text
X_train = vectorizer.fit_transform(train_text)
y_train = [1] * len(train_text)
X_test = vectorizer.transform(test_text)
y_test = [1] * len(test_text)

# Train a Multinomial Naive Bayes classifier on the training data
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Use the trained classifier to make predictions on the test data
accuracy = clf.score(X_test, y_test)

print("Accuracy:", accuracy)