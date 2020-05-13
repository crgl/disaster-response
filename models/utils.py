from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag

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
    tokens = [tok for tok in map(my_lemmatizer, word_tokenize(text))]
    return tokens