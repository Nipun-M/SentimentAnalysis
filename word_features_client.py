
import pickle
import os

from word_features import get_dataframe, sentiment_encode, model_structure, make_keras_picklable, clean, get_word_features
from sklearn.feature_extraction.text import TfidfVectorizer


def train_and_store(filename, epochs=5):
    # ### extract word features

    max_features = 300
    vectorizer = TfidfVectorizer(max_features=max_features)
    print("Reading data ...")
    data = get_dataframe(filename)
    print("Cleaning data ...")
    data['clean_content'] = data.apply(clean, axis=1)
    print("Getting word features ...")
    features = get_word_features(vectorizer, data)

    # # TRAINING PHASE

    # ### one hot encode the data
    for sentiment in set(data['sentiment']):
        encoder = sentiment_encode(sentiment)
        data[sentiment] = data.apply(encoder, axis=1)

    make_keras_picklable()
    models = dict()
    for sentiment in set(data['sentiment']):
        print("training",sentiment,"model")
        models[sentiment] = model_structure(sentiment)
        models[sentiment].fit(features, data[sentiment], epochs=epochs, batch_size=64)
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


def test(sentence):
    vectorizer = pickle.load(open("objects/vectorizer.pickle", "rb"))
    models = dict()
    sentiments = ['joy', 'fear', 'anger', 'guilt', 'shame', 'sadness', 'disgust']
    for sentiment in sentiments:
        models[sentiment] = pickle.load(open("objects/" + sentiment + "_model.pickle", "rb"))

    # get features for sentence

    # test against the model

    # conclude on the predictions

    # return prediction
    return None