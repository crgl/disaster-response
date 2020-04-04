import sys

import nltk
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

import numpy as np
import pandas as pd

from sqlalchemy import create_engine

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    table_name = database_filepath.split('/')[-1].split('.')[0]
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns
    return X, y, category_names

def pos_convert(pos):
    if pos.startswith('N'):
        pos = 'n'
    elif pos.startswith('R'):
        pos = 'r'
    elif pos.startswith('V'):
        pos = 'v'
    elif pos.startswith('J'):
        pos = 'a'
    else:
        pos = ''
    return pos != '', pos

def my_lemmatizer(word, lemmatizer=WordNetLemmatizer()):
    word = word.lower().strip()
    valid, pos = pos_convert(pos_tag(word)[0][0])
    if valid:
        word = lemmatizer.lemmatize(word, pos)
    return word

def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    tokens = [tok for tok in map(my_lemmatizer, word_tokenize(text))]
    return tokens


def build_model():
    model = Pipeline([
        ("featurer", TfidfVectorizer(tokenizer=tokenize, stop_words='english')),
        ("classifier", OneVsRestClassifier(LinearSVC()))
    ])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()