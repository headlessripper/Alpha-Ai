import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import numpy as np
import hyperopt

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

# Define the CNN model
model = Sequential()
model.add(Embedding(5000, 100, input_length=100))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Define early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

# Train the model
history = model.fit(X_train, to_categorical(y_train), epochs=10, batch_size=32, validation_data=(X_test, to_categorical(y_test)), callbacks=[early_stopping, model_checkpoint])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, to_categorical(y_test))
print('Test accuracy:', accuracy)

# Use the trained model to make predictions on the test data
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Print the classification report and confusion matrix
print('Classification Report:')
print(classification_report(y_test, predicted_classes))
print('Confusion Matrix:')
print(confusion_matrix(y_test, predicted_classes))

# Define the hyperparameter search space
space = {
    'num_filters': hyperopt.hp.quniform('num_filters', 32, 128, 16),
    'kernel_size': hyperopt.hp.quniform('kernel_size', 3, 7, 2),
    'dropout_rate': hyperopt.hp.uniform('dropout_rate', 0.2, 0.5),
    'l2_regularizer': hyperopt.hp.uniform('l2_regularizer', 0.001, 0.1)
}

# Define the objective function to optimize
def optimize_model(params):
    model = Sequential()
    model.add(Embedding(5000, 100, input_length=100))
    model.add(Conv1D(params['num_filters'], kernel_size=params['kernel_size'], activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(params['l2_regularizer']
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    # Define early stopping and model checkpointing
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

    # Train the model
    history = model.fit(X_train, to_categorical(y_train), epochs=10, batch_size=32,
                        validation_data=(X_test, to_categorical(y_test)), callbacks=[early_stopping, model_checkpoint])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, to_categorical(y_test))
    return -accuracy

    # Perform hyperparameter tuning
    trials = hyperopt.Trials()
    best = hyperopt.fmin(optimize_model, space, algo=hyperopt.tpe.suggest, trials=trials, max_evals=50)

    print('Best hyperparameters:', best)

    # Define the ensemble model
    def ensemble_model(models):
        inputs = Input(shape=(100,))
        outputs = []
        for model in models:
            outputs.append(model(inputs))
        outputs = Average()(outputs)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    # Define the attention model
    def attention_model():
        inputs = Input(shape=(100,))
        x = Embedding(5000, 100, input_length=100)(inputs)
        x = Conv1D(64, kernel_size=3, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(2, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    # Define the adversarial model
    def adversarial_model():
        inputs = Input(shape=(100,))
        x = Embedding(5000, 100, input_length=100)(inputs)
        x = Conv1D(64, kernel_size=3, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(2, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    # Train the ensemble model
    models = [attention_model(), adversarial_model()]
    ensemble = ensemble_model(models)
    ensemble.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    ensemble.fit(X_train, to_categorical(y_train), epochs=10, batch_size=32,
                 validation_data=(X_test, to_categorical(y_test)))

    # Evaluate the ensemble model
    loss, accuracy = ensemble.evaluate(X_test, to_categorical(y_test))
    print('Ensemble accuracy:', accuracy)