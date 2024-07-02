import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import string

import contractions
import re
from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer

punctuations = string.punctuation
stopword = list(STOP_WORDS)
stopword[:10]


# Parser
parser = spacy.load("en_core_web_sm")
parser.max_length = 7000000


def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopword and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

def spacy_lemmatizer(sentence):
    doc = parser(sentence)
    lemmatized_words = [token.lemma_ for token in doc]
    lemmatized_words = " ".join([i for i in lemmatized_words])
    return lemmatized_words

def fix_contractions(text):
  expanded_words=[]
  for word in text.split():
    expanded_words.append(contractions.fix(word.lower()))   
  return ' '.join(expanded_words)

def remove_punc(text):
  return re.sub('[%s]' % re.escape(string.punctuation), '' , text)

def remove_white_spaces(text):
  return re.sub(' +', ' ', text)

def tokenize(text):
    word_tok= word_tokenize(text)
    return word_tok

def remove_stop_words(word_tok):
    se= stopwords.words('english')
    word_li=[]
    st_li=[]
    for w in word_tok:
        if w not in se:
            word_li.append(w)
    return " ".join(word_li)


def preprocess(text):
    text= fix_contractions(text)
    text= remove_punc(text)
    text= remove_white_spaces(text)
    tokens= tokenize(text)
    stop_w_rem= spacy_tokenizer(" ".join(tokens))
    lemmatized_words= spacy_lemmatizer(stop_w_rem)
    return lemmatized_words