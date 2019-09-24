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
from gensim.utils import simple_preprocess
from gensim.summarization import keywords
import requests.exceptions
from urllib.parse import urlsplit
from collections import deque
from joblib import Parallel, delayed
from chardet.universaldetector import UniversalDetector
import dask

plt.style.use('ggplot')
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(parent_dir, 'resources', 'stopwords.txt')) as stpw:
    stopwords = set(stopwords.words('english') + stpw.read().strip().split(
        '\n') + list(whitespace) + list(punctuation))


def detect_encoding(filename):
    """
    Detect encoding
    :param filename: name of file
    :return:
    """
    detector = UniversalDetector()
    with open(filename, 'rb') as socket:
        for line in socket:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
    return detector.result['encoding']

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
    text = re.sub('\S*@\S*\s?', '', text)
    text = re.sub('\s+', ' ', text)
    text = [x for x in simple_preprocess(text) if x not in stopwords]
    return ' '.join(text)


class GetPages(object):
    def __init__(self, query, num=20, stop=5, depth=3):
        self.query = query
        self.num = num
        self.stop = stop
        self.depth = depth
        self.gsearch = search(self.query, tld="co.in", num=self.num,
                              stop=self.stop, pause=2)
        self.landing_page = next(self.gsearch)
        self.pages = []
        self.text = []

    @property
    def landing_page(self):
        return self.__landing_page

    @landing_page.setter
    def landing_page(self, url):
        results = [dask.delayed(self.read_url)(u) for u in self.crawl(url)]
        out = dask.compute(*results)
        self.__landing_page = ' '.join(out)

    @property
    def text(self):
        return self.__text

    @text.setter
    def text(self, _):
        results = [dask.delayed(self.search_google)(url) for url in self.gsearch]
        out = dask.compute(*results)
        self.__text, self.pages = zip(*out)

    @staticmethod
    def crawl(url, depth=10):
        new_urls = deque([url])
        processed_urls = set()
        local_urls = set()
        foreign_urls = set()
        broken_urls = set()
        count = 0
        while len(new_urls) != 0 and count <= depth:
            count += 1
            url = new_urls.popleft()
            processed_urls.add(url)
            print("Processing %s" % url)

            try:
                response = requests.get(url)
            except(
            requests.exceptions.MissingSchema, requests.exceptions.ConnectionError,
            requests.exceptions.InvalidURL,
            requests.exceptions.InvalidSchema,
            requests.exceptions.Timeout):
                broken_urls.add(url)
                continue
            parts = urlsplit(url)
            base = "{0.netloc}".format(parts)
            strip_base = base.replace("www.", "")
            base_url = "{0.scheme}://{0.netloc}".format(parts)
            path = url[:url.rfind('/')+1] if '/' in parts.path else url
            soup = BeautifulSoup(response.text, "lxml")
            for link in soup.find_all('a'):
                anchor = link.attrs["href"] if "href" in link.attrs else ''
                if anchor.startswith('/'):
                    local_link = base_url + anchor
                    local_urls.add(local_link)
                elif strip_base in anchor:
                    local_urls.add(anchor)
                elif not anchor.startswith('http'):
                    local_link = path + anchor
                    local_urls.add(local_link)
                else:
                    foreign_urls.add(anchor)
            for i in local_urls:
                if not i in new_urls and not i in processed_urls:
                    new_urls.append(i)
        return local_urls

    def search_google(self, url):
        """
        Get a set of texts from google given a query result (url)
        :param url: url to process
        :return: list of blogs' text
        """
        urls = self.crawl(url, depth=self.depth)
        docs = ' '.join([self.read_url(u) for u in urls])
        return url, docs

    @staticmethod
    def read_url(url):
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X '
                                 '10_11_5) AppleWebKit/537.36 (KHTML, like '
                                 'Gecko) Chrome / 50.0.2661.102 Safari / '
                                 '537.36'}
        try:
            req = requests.get(url, headers=headers)
            html_doc = req.content
            soup = BeautifulSoup(html_doc.decode('ascii', errors='ignore'),
                                 'html.parser')
            try:
                return ' '.join([pre_clean(x.get_text()) for x in soup.find_all(
                    'main')])
            except AttributeError:
                print(req.status_code)
                print(html_doc)
        except(
                requests.exceptions.MissingSchema,
                requests.exceptions.ConnectionError,
                requests.exceptions.InvalidURL,
                requests.exceptions.InvalidSchema,
                requests.exceptions.Timeout):
            return ''


class IdentifyWords(object):
    """
    Use NLP's TF-IDF to identify keywords of a set of documents
    """
    def __init__(self, docs, stats, landing_doc, max_df=0.9, min_df=0.01,
                 max_features=None, n_keywords=10, model='word2vec'):
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
        self.pre_keywords = [keywords(x, deacc=True) for x in self.docs]
        self.landing_kw = keywords(landing_doc)
        self.gkp = pd.read_csv(stats, skiprows=[0,1], encoding=detect_encoding(
            stats))
        self.nlp = spacy.load('en', disable=['ner', 'parser'])
        if model == 'tf-idf':
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


def main(query, stats, num, stop, max_df, min_df, max_features, n_keywords,
         plot_fn, model):
    if os.path.isfile('pages.dmp'):
        with open('pages.dmp') as p, open('landing.dmp') as l:
            text = [line for line in p]
            land = [line for line in l]
    else:
        pages = GetPages(query, num, stop)
        text = pages.text
        land = pages.landing_page
        with open('pages.dmp', 'w') as p, open('landing.dmp', 'w') as l:
            p.write(text)
            l.write(land)
    iw = IdentifyWords(text, stats, land, max_df, min_df, max_features,
                       n_keywords, model=model)
    #tfidf.frequency_explorer(plot_fn)
    print('pre-KeyWords\n', iw.pre_keywords)
    print('Landing pages KeyWords\n', iw.landing_kw)
    print('%s Keywords\n' % model, iw.keywords)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PROG', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('query', help='Original query')
    parser.add_argument('stats', help='Google Keyword Planner filename')
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
    parser.add_argument('-M', '--model', default='word2vec',
                        help='NLP model to fit')
    args = parser.parse_args()
    main(query=args.query, stats=args.stats, num=args.max_results,
         stop=args.stop, max_df=args.max_df, plot_fn=args.plot_fn,
         model=args.model, max_features=args.max_features, min_df=args.min_df,
         n_keywords=args.n_keywords)

