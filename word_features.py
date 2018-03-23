# # Sentiment analysis : word features

# # Training process

# ### import required modules

import csv
import pandas as pd
import numpy as np
import tempfile
import string
import keras.models
import types
import h5py

from nltk.corpus import stopwords as sw
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers.embeddings import Embedding


# ### define some variables

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stopwords = sw.words('english')


# ### function to clean the word

def process_word(word):
    word = lemmatizer.lemmatize(word)
    word = stemmer.stem(word)
    return word


# ### function to process a sentence given in string format

def clean_sentence(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = sentence.lower()
    words = word_tokenize(sentence)
    words = list(filter(lambda word: word not in stopwords, words))
    words = list(map(process_word, words))
    return ' '.join(words)


# ### function to clean the sentence

def clean(row):
    sentence = row['content']
    return clean_sentence(sentence)


# ### function to get the data in the form of a dataframe

def get_dataframe(filename):
    raw_data = dict()
    reader = csv.reader(open(filename))
    headers = list(reader.__next__())
    for header in headers:
        raw_data[header] = list()
    for row in reader:
        raw_data['sentiment'].append(row[0])
        raw_data['content'].append(','.join(row[1:]))
    df = pd.DataFrame(raw_data)
    return df


# ### sentiment one hot encoding function

def sentiment_encode(sentiment):
    def encode(row):
        return 1 if row['sentiment'] == sentiment else 0
    return encode


# ## Making the keras model picklable
# ### source : http://zachmoshe.com/2017/04/03/pickling-keras-models.html

def make_keras_picklable():
    def __getstate__(self):
        model_str = None
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__
    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


# ## Define the basic structure of the model

def model_structure(label='default', random_seed=0, embedding_vector_length=128, max_features=300, 
                     n_units=60, activation='sigmoid', loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']):
    np.random.seed(random_seed)
    model = Sequential()
    embedding_vector_length = embedding_vector_length
    model.add(Embedding(max_features, embedding_vector_length, input_length=max_features))
    model.add(LSTM(n_units))
    model.add(Dense(1, activation=activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print(label.upper())
    print(model.summary())
    print("-------------------------------------------------------------------------------------------------")
    return model

def get_word_features(vectorizer, dataframe):
    return np.array(vectorizer.fit_transform(dataframe['clean_content']).toarray())
