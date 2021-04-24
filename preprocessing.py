import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet


def preprocess_caption(caption):
    # tokenize and convert to lower case
    caption = [word.lower() for word in caption.split()]

    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    # remove punctuation from each word
    caption = [word.translate(table) for word in caption]

    # remove words with numbers in them
    caption = [word for word in caption if word.isalpha()]

    # store caption as string
    caption = ' '.join(caption)

    return caption


def get_wordnet_pos(word):
    # Map POS tag to first character lemmatize() accepts
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_caption(caption, lemmatizer, stemmer):
    stop_words = list(set(stopwords.words('english')))
    return ' '.join(list(set([stemmer.stem(lemmatizer.lemmatize(word, get_wordnet_pos(word)))
                              for word in nltk.word_tokenize(caption)
                              if word not in stop_words])))
