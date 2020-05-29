import sys

sys.path.insert(0, 'models')

import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objs as g_o

import tensorflow_hub as hub

import dill

from sqlalchemy import create_engine

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
with open("models/classifier.pkl", 'rb') as f:
    model = dill.load(f)

model.predict(["Welcome!"])

with open("models/roc.pkl", 'rb') as f:
    roc_data = dill.load(f)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                g_o.Scatter(
                    x=roc_data['fpr'][k],
                    y=roc_data['tpr'][k],
                    name='',
                    mode='lines',
                    showlegend=False,
                    hovertemplate='{} (area = {:.3f})'.format(k, roc_data['auc'][k])
                ) for k in roc_data['auc'].keys()
            ],

            'layout': {
                'title': 'ROC Curves for Each Category',
                'yaxis': {
                    'title': "True Positive Rate"
                },
                'xaxis': {
                    'title': "False Positive Rate"
                },
                'height': 700,
                'hovermode': 'closest'
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()