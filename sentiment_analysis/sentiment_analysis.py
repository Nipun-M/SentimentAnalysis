
import numpy as np
import pickle
import os
import pandas as pd

from utility.utility import get_dataframe
import word_features.word_features as wf
import sentiment_features.senti_scores as sf
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers.embeddings import Embedding
from keras_pickle_wrapper import KerasPickleWrapper

# sentiment one hot encoding function

def sentiment_encode(sentiment):
    def encode(row):
        return 1 if row['sentiment'] == sentiment else 0
    return encode


# ## Making the keras model picklable
# ### source : http://zachmoshe.com/2017/04/03/pickling-keras-models.html
# <DEAD CODE> : Failed to store certain states after session ended
# Found keras_pickle_wrapper : used to pickle and unpickle keras objects

# ## Define the basic structure of the model

def model_structure(label='default', random_seed=0, embedding_vector_length=128, max_features=315,
                     n_units=60, activation='sigmoid', loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'],
                     verbose=False):
    np.random.seed(random_seed)
    model = Sequential()
    embedding_vector_length = embedding_vector_length
    model.add(Embedding(max_features, embedding_vector_length, input_length=max_features))
    model.add(LSTM(n_units))
    model.add(Dense(1, activation=activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model_wrapped = KerasPickleWrapper(model)
    if verbose:
        print(label.upper())
        print(model.summary())
        print("-------------------------------------------------------------------------------------------------")
    return model_wrapped



def train_and_store(filename, epochs=5):
    # ### extract word features

    max_features = 300
    vectorizer = TfidfVectorizer(max_features=max_features)
    print("Reading data ...")
    data = get_dataframe(filename)

    print("Getting word features ...")
    word_features = wf.get_word_features(vectorizer, data)

    print("Getting sentiment features ...")
    sentiment_features = sf.get_sentiment_features(data)

    features = np.hstack((word_features, sentiment_features))

    # # TRAINING PHASE

    # ### one hot encode the data
    for sentiment in set(data['sentiment']):
        encoder = sentiment_encode(sentiment)
        data[sentiment] = data.apply(encoder, axis=1)

    models = dict()
    for sentiment in set(data['sentiment']):
        print("training", sentiment, "model")
        models[sentiment] = model_structure(sentiment)
        models[sentiment]().fit(features, data[sentiment], epochs=epochs, batch_size=64)
    print("Done with training models!!!")

    # # SERIALIZE THEM INTO THE FILE

    try:
        print("Trying to create folder objects/ ...")
        os.mkdir("objects")
    except Exception:
        print("Folder already exists!")
    print("Serializing all models and the vectorizer ...")
    for sentiment in set(data['sentiment']):
        pickle.dump(models[sentiment], open("objects/" + sentiment + "_model.pickle", "wb"))
        print("Stored objects/"+sentiment+"_model.pickle")
    pickle.dump(vectorizer, open("objects/vectorizer.pickle", "wb"))
    print("Stored objects/vectorizer.pickle")
    print("FINISH!!!")


def soft_max(probs):
    total = sum(probs.values())
    max_value = 0
    tag = None
    for prob in probs.keys():
        val = probs[prob] / total
        if val > max_value:
            max_value = val
            tag = prob
    return (max_value, tag)


def test(sentence):
    vectorizer = pickle.load(open("objects/vectorizer.pickle", "rb"))
    models = dict()
    sentiments = ['joy', 'fear', 'anger', 'guilt', 'shame', 'sadness', 'disgust']
    for sentiment in sentiments:
        models[sentiment] = pickle.load(open("objects/" + sentiment + "_model.pickle", "rb"))


    test = pd.DataFrame({'content': [sentence]})
    # get features for sentence
    word_features = wf.get_word_features_test(vectorizer, test)
    sentiment_features = sf.get_sentiment_features(test)
    features = np.hstack((word_features, sentiment_features))

    # test against the model
    scores = dict()
    for sentiment in sentiments:
        scores[sentiment] = models[sentiment]().predict(features)[0][0]

    # conclude on the predictions
    predictions = soft_max(scores)

    # return prediction
    return predictions, scores