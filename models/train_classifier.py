import sys

import numpy as np
import pandas as pd

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.dummy import DummyClassifier

import tensorflow as tf
import tensorflow_hub as hub

import dill

from utils import tokenize

class TunableLogreg(BaseEstimator, ClassifierMixin):
    '''
    Since this is a case where sacrificing precision for recall makes sense,
    we have this wrapper around logistic regression to allow for a tunable bias
    towards positive responses in the form of the threshold parameter.
    '''
    def __init__(self, threshold = 0.4, class_weight='balanced', penalty='l2', dual=False, fit_intercept=True, random_state=None, solver='lbfgs', max_iter=200):
        self.threshold = threshold
        self.class_weight = class_weight
        self.penalty = penalty
        self.dual = dual
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter

    def fit(self, X, y):
        self.logreg = LogisticRegression(class_weight=self.class_weight, penalty=self.penalty, dual=self.dual, fit_intercept=self.fit_intercept, random_state=self.random_state, solver=self.solver, max_iter=self.max_iter)
        try:
            self.logreg.fit(X, y)
        except:
            # This predicts negative for classes with no positive examples
            self.logreg = DummyClassifier(strategy='most_frequent')
            self.logreg.fit(X, y)
        self.classes_ = self.logreg.classes_

    def predict(self, X):
        labels = [1 - int(x[0]) for x in (self.logreg.predict_proba(X) > self.threshold)]
        return np.asarray(labels)

    def predict_proba(self, X):
        return self.logreg.predict_proba(X)

class TweetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        return self.embed(X).numpy()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['embed']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def load_data(database_filepath):
    table_name = database_filepath.split('/')[-1].split('.')[0]
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns
    return X, y, category_names

def build_model():
    model = Pipeline([
        ("featurizer", FeatureUnion([
            ("encoder", TweetTransformer()),
            ("lsa", Pipeline([
                ("tfidf", TfidfVectorizer(tokenizer=tokenize, stop_words='english')),
                ("reducer", TruncatedSVD(n_components=250))
            ]))
        ])),
        ("classifier", MultiOutputClassifier(TunableLogreg()))
    ])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    # The `child_alone` class necessitates these workarounds for predict_proba
    y_probs = np.array([1. - x[:, 0] for x in model.predict_proba(X_test)]).T
    macro_auc = roc_auc_score(Y_test.drop('child_alone', axis=1), np.delete(y_probs, 9, axis=1), average='macro')
    micro_auc = roc_auc_score(Y_test.drop('child_alone', axis=1), np.delete(y_probs, 9, axis=1), average='micro')
    weighted_auc = roc_auc_score(Y_test.drop('child_alone', axis=1), np.delete(y_probs, 9, axis=1), average='weighted')
    print("Average AUC (macro): {}".format(macro_auc))
    print("Average AUC (micro): {}".format(micro_auc))
    print("Average AUC (weighted): {}".format(weighted_auc))
    print(classification_report(Y_test, y_pred, target_names=category_names))

def save_roc_data(model, X_test, Y_test, roc_filepath):
    model_probs = np.array([1. - x[:, 0] for x in model.predict_proba(X_test)]).T
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(36):
        if i == 9:
            continue
        label = Y_test.columns[i]
        fpr[label], tpr[label], _ = roc_curve(Y_test.iloc[:, i], model_probs[:, i])
        roc_auc[label] = auc(fpr[label], tpr[label])
    roc_data = {}
    roc_data['fpr'] = fpr
    roc_data['tpr'] = tpr
    roc_data['auc'] = roc_auc
    with open(roc_filepath, 'wb') as f:
        dill.dump(roc_data, f)

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        dill.dump(model, f)

def main():
    if len(sys.argv) == 4:
        database_filepath, model_filepath, roc_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        print('Training model...')
        model.fit(X_train, Y_train)
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        print('Saving roc data...')
        save_roc_data(model, X_test, Y_test, roc_filepath)

        print('Retraining model on full dataset...')
        model.fit(X, Y)
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument, the filepath of the pickle file to save '\
              'the model to as the second argument, and the filepath of the '\
              'pickle file to save the roc data to as the third argument. \n\n'\
              'Example: python train_classifier.py ../data/DisasterResponse.db classifier.pkl roc.pkl')


if __name__ == '__main__':
    main()