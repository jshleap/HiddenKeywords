"""
**__utils__.py**
File with utility functions and constants
**Copyright** 2019  Jose Sergio Hleap
"""
import json
import re
from io import BytesIO
from itertools import chain
from os.path import dirname, abspath, join
from string import punctuation, whitespace

import requests
import requests.exceptions
from chardet.universaldetector import UniversalDetector
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords


from ADvantage.__version__ import version

# File Attributes-- -----------------------------------------------------------
__author__ = 'Jose Sergio Hleap'
__version__ = version
__email__ = 'jshleap@gmail.com'

parent_dir = dirname(dirname(abspath(__file__)))
# Extend stop words with a custom file and punctuation
with open(join(parent_dir, 'resources', 'stopwords.txt')) as stpw:
    stopwords = set(stopwords.words('english'))
    stopwords.update(simple_preprocess(stpw.read().strip().replace('\n', ' '),
                                       deacc=True))
    stopwords.update(list(whitespace) + list(punctuation))


# Utility Functions -----------------------------------------------------------
def detect_encoding(filename):
    """
    Detect encoding
    :param filename: name of file
    :return: encoding string
    """
    detector = UniversalDetector()
    with open(filename, 'rb') as socket:
        for line in socket:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
    return detector.result['encoding']


def get_synonyms(word, key):
    """
    Given a word return similar words (synonyms)
    :param word: string word to find synonyms for
    :return: set of synonyms
    """
    if ' ' in word:
        word = '+'.join(word.split())
    rest = "https://www.dictionaryapi.com/api/v3/references/thesaurus/json/%s"
    rest += word + "?key=%s" % key
    js = requests.get(rest)
    js = json.load(BytesIO(js.content))
    return set(chain(*[list(chain(*req['meta']['syns'])) for req in js]))


def pre_clean(text):
    """
    Remove unwanted characters and normalize text to lowercase
    :param text: Text to be pre-cleaned
    :return: pre-cleaned text
    """
    text = text.lower()
    # html tags, special characters, white spaces and digits
    text = re.sub("<!--?.*?-->", "", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    text = re.sub("_.+\s", "", text)
    text = re.sub('\S*@\S*\s?', '', text)
    text = re.sub('\s+', ' ', text)
    #remove repeated words
    text = re.sub(r'\b(\w+\s*)\1{1,}', '\\1', text)
    return text