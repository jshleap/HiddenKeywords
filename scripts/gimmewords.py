"""
**gimmewords.py**
**Copyright** 2019  Jose Sergio Hleap
"""
__author__ = 'Jose Sergio Hleap'
__version__ = '0.1b'
__email__ = 'jshleap@gmail.com'


import multiprocessing
import argparse
import json
import os
import re
from io import BytesIO
from itertools import chain
from string import punctuation, whitespace
import spacy
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec

plt.style.use('ggplot')
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(parent_dir, 'resources', 'stopwords.txt')) as stpw:
    stopwords = set(stopwords.words('english') + stpw.read().strip().split(
        '\n') + list(whitespace) + list(punctuation))


def get_synonyms(word):
    """
    Given a word return similar words
    TODO: Include the API key as variable
    :param word: string word to find synonyms for
    :return: list of synonyms
    """
    if ' ' in word:
        word = '+'.join(word.split())
    rest = "https://www.dictionaryapi.com/api/v3/references/thesaurus/json/"
    rest += word + "?key=c9473e1d-4110-4ee3-a101-778858363e9f"
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
    # html tags, special characters, and digits
    text = re.sub("<!--?.*?-->", "", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    text = re.sub("_.+\s", "", text)
    return text


class GetPages(object):
    def __init__(self, query, num=100, stop=10):
        self.query = query
        self.num = num
        self.stop = stop
        self.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X'
                                      ' 10_11_5) AppleWebKit/537.36 (KHTML, '
                                      'like Gecko) Chrome / 50.0.2661.102 '
                                      'Safari / 537.36'}
        self.pages = []
        self.text = []
        self.search_google()

    def search_google(self):
        """
        Get a set of texts from google given a query

        :return: list of blogs' text
        """
        for j in search(self.query, tld="co.in", num=self.num, stop=self.stop,
                        pause=2):
            self.pages.append(j)
            req = requests.get(j, headers=self.headers)
            html_doc = req.content
            soup = BeautifulSoup(html_doc.decode('ascii', errors='ignore'),
                                 'html.parser')
            try:
                self.text.append('\n'.join([pre_clean(x.get_text()) for x in
                                            soup.find_all('body')]))
            except AttributeError:
                print(req.status_code)
                print(html_doc)


class IdentifyWords(object):
    """
    Use NLP's TF-IDF to identify keywords of a set of documents
    """
    def __init__(self, docs, max_df=0.9, min_df=0.01, max_features=None,
                 n_keywords=10, word2vec=False):
        """
        Constructor
        :param docs: list of strings with the documents to analyze
        :param max_df: When building the vocabulary ignore terms that have a
        document frequency strictly higher than the given threshold
        :param min_df: When building the vocabulary ignore terms that have a
        document frequency strictly lower than the given threshold
        :param max_features: build a vocabulary that only consider the top
        max_features ordered by term frequency
        """
        self.cores = multiprocessing.cpu_count()
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.vocabulary = None
        self.n = n_keywords
        self.docs = docs
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vectorize = CountVectorizer(max_df=max_df, stop_words=stopwords,
                                         lowercase=True,
                                         max_features=max_features)
        self.w2v = Word2Vec(min_count=20, window=2, size=300, sample=6e-5,
                            alpha=0.03, min_alpha=0.0007, negative=20,
                            workers=self.cores - 1)
        self.clean = None
        self.keywords = None
        self.nlp = spacy.load('en', disable=['ner', 'parser'])
        if not word2vec:
            self.text_counts = docs
            self.tf_idf()
        else:
            self.clean_it()
            self.word2vec()

    @property
    def text_counts(self):
        return self.__text_counts

    @text_counts.setter
    def text_counts(self, _):
        self.__text_counts = self.vectorize.fit_transform(self.docs)
        self.vocabulary = self.vectorize.get_feature_names()
        print('Top 10 words in vocabulary')
        print(list(self.vectorize.vocabulary_.keys())[:10])

    def clean_it(self):
        """
        Process and clean documents for word2vec
        :return:
        """
        cleaning = lambda doc: ' '.join([token.lemma_ for token in doc
                                         if not token.is_stop])
        txt = [cleaning(doc) for doc in self.nlp.pipe(
            (pre_clean(x) for x in self.docs), batch_size=5000, n_threads=-1)]
        self.clean = pd.DataFrame({'clean': txt}).dropna().drop_duplicates()

    def word2vec(self):
        """
        Perform word2vec processing
        :return:
        """
        sent = [row.split() for row in self.clean['clean']]
        bigram = Phraser(Phrases(sent, min_count=30, progress_per=10000))
        sentences = bigram[sent]
        self.w2v.build_vocab(sentences, progress_per=10000)
        self.w2v.train(sentences, total_examples=self.w2v.corpus_count,
                       epochs=30, report_delay=1)
        self.w2v.init_sims(replace=True)

    def frequency_explorer(self, outname):
        """
        This function tokenize it and plots the frequency of a text string

        :param outname: name of the plot, including the desired extension
        """
        txt = ' '.join([word for L in self.docs for word in L.split() if
                        word not in stopwords])
        fdist = FreqDist(self.tokenizer.tokenize(txt))
        print(fdist)
        fdist.plot(30, cumulative=False)
        plt.savefig(outname)

    def tf_idf(self):
        """
        Process the textsusing NLP's tf_idf
        """
        self.tfidf_transformer = TfidfTransformer(smooth_idf=True,
                                                  use_idf=True)
        self.tfidf_transformer.fit(self.text_counts)
        self.df_idf = pd.DataFrame(self.tfidf_transformer.idf_,
                                   index=self.vocabulary, columns=["weights"])
        self.keywords = self.df_idf.nlargest(n=self.n, columns="weights")


def scrape_paperspace(url='https://blog.paperspace.com/tag/machine-learning/'):
    """
    Get all the blogs from paperspace blog
    :return: list of all texts
    """
    parent_url = '/'.join(url[:-1].split('/')[:-1])
    html_doc = requests.get(url).content.decode('utf-8')
    soup = BeautifulSoup(html_doc, 'html.parser')
    documents = []
    for tag in soup.find_all('a'):
        if 'class' in tag.attrs:
            if tag.attrs['class'][0] == 'post-card-content-link':
                documents.append(BeautifulSoup(requests.get(
                    parent_url + tag.get('href')), 'html.parser'))
    return documents


def main(query, num, stop, max_df, min_df, max_features, n_keywords, plot_fn):
    pages = GetPages(query, num, stop)
    tfidf = IdentifyWords(pages.text, max_df, min_df, max_features,
                          n_keywords)
    tfidf.frequency_explorer(plot_fn)
    print(tfidf.keywords)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PROG', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('query', help='Original query')
    parser.add_argument('-x', '--max_results', type=int, default=20,
                        help='Maximum number of results from google query')
    parser.add_argument('-s', '--stop', type=int, default=10,
                        help='Number or records to show')
    parser.add_argument('-m', '--max_df', type=float, default=0.9,
                        help='When building the vocabulary ignore terms that '
                             'have a document frequency strictly higher than '
                             'the given threshold')
    parser.add_argument('-l', '--min_df', type=float, default=0.01,
                        help='When building the vocabulary ignore terms that '
                             'have a document frequency strictly lower than '
                             'the given threshold')
    parser.add_argument('-k', '--max_features', type=int, default=None,
                        help='build a vocabulary that only consider the top '
                             'max_features ordered by term frequency')
    parser.add_argument('-n', '--n_keywords', type=int, default=10,
                        help='Maximum number of keywords to retrieve')
    parser.add_argument('-p', '--plot_fn', type=str, default='freq_plot.pdf',
                        help='Name of the plot file')
    args = parser.parse_args()
    main(query=args.query, num=args.max_results, stop=args.stop,
         max_df=args.max_df, min_df=args.min_df, max_features=args.max_features,
         n_keywords=args.n_keywords, plot_fn=args.plot_fn)

