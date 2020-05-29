# Disaster Response Pipeline Project

A live version of the web dashboard is hosted at disaster-response.crgl.dev. Instructions for running locally are below

### Local Instructions:

1. Run the following commands in the project's root directory to set up the database and model.

    - To run the ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl models/roc.pkl`

2. Run the following command in the app's directory to run your web app.
    `python main.py`

3. Go to http://0.0.0.0:3001/
