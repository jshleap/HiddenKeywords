"""
**gimmewords.py**
**Copyright** 2019  Jose Sergio Hleap
"""
__author__ = 'Jose Sergio Hleap'
__version__ = '0.1b'
__email__ = 'jshleap@gmail.com'

from types import GeneratorType
import argparse
import json
import multiprocessing
import re
import time
from base64 import b64encode
from collections import deque
from http.client import HTTPSConnection
from io import BytesIO
from itertools import chain
from json import dumps
from json import loads
from os.path import join, pardir, abspath, dirname, isfile
from string import punctuation, whitespace
from urllib.parse import urlsplit

import dask
import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import requests.exceptions
import spacy
from bs4 import BeautifulSoup
from chardet.universaldetector import UniversalDetector
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, wrappers, CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.summarization import keywords, textcleaner
from gensim.utils import simple_preprocess
from googlesearch import search
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from ADvantage.scripts.__dataforseo_credentials__ import *

plt.style.use('ggplot')

# Constants -------------------------------------------------------------------
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
    # text to list removing stopwords
    text = [x for x in simple_preprocess(text, deacc=True) if x not in
            stopwords and len(x) != 1]
    return ' '.join(text)


def get_stats(keywords, dfs_login, dfs_pass, label):
    """
    Get SEO statistics for a list of keywords using Data For SEO API

    :param keywords: List of keywords to query
    :param dfs_login: Data for SEO login
    :param dfs_pass: Data for SEO password
    :return: Dataframe with the info
    """
    print('Fetching SEO stats for keywords')
    client = RestClient(dfs_login, dfs_pass)
    # check if I am within the limit of 2500 words
    try:
        assert len(keywords) <= 2500
    except AssertionError:
        print(len(keywords))
        raise
    keywords_list = [
        dict(
            language="en",
            loc_name_canonical="Canada",
            bid=100,
            match="exact",
            keys=keywords
        )
    ]
    response = client.post("/v2/kwrd_ad_traffic_by_keywords",
                           dict(data=keywords_list))
    if response["status"] == "error":
        print("error. Code: %d Message: %s" % (
            response["error"]["code"], response["error"]["message"]))
        raise
    else:
        results = response["results"][0]
    try:
        df = pd.DataFrame(results[0].values(), index=results[0].keys())
    except KeyError:
        df = pd.DataFrame(results.values(), index=results.keys())
    df = df.reset_index().rename(columns={'index': 'Keyword'})
    df['source'] = label
    return df


# Main Classes ----------------------------------------------------------------
class RestClient:
    """
    Class of the third party API to retrive word statistics using Data For SEO.
    This class is taken from TODO: add the link to client.py
    """
    domain = "api.dataforseo.com"

    def __init__(self, username, password):
        """
        Constructor of the Rest Client Class
        :param username: Username (usually email) to Data for SEO API
        :param password: Password to Data for SEO API
        """
        self.username = username
        self.password = password

    def request(self, path, method, data=None):
        """
        Make a rest request
        :param path: Path to service. For our purpose, the keywords
        :param method: Weher a GET or POST operation
        :param data: Search query
        :return: response instance
        """
        connection = HTTPSConnection(self.domain)
        try:
            base64_bytes = b64encode(
                ("%s:%s" % (self.username, self.password)).encode("ascii")
            ).decode("ascii")
            headers = {'Authorization': 'Basic %s' % base64_bytes}
            connection.request(method, path, headers=headers, body=data)
            response = connection.getresponse()
            return loads(response.read().decode())
        finally:
            connection.close()

    def get(self, path):
        """
        Make a GET request
        :param path: Path to service
        :return: response instance
        """
        return self.request(path, 'GET')

    def post(self, path, data):
        """
        Make a POST request
        :param path: path to service
        :param data: query
        :return: response instance
        """
        if isinstance(data, str):
            data_str = data
        else:
            data_str = dumps(data)
        return self.request(path, 'POST', data_str)


class GetPages(object):
    """
    Class to crawl and scrape webpages based on query.
    """
    def __init__(self, query, max_results=20, depth=3):
        """
        Constructor of GetPages class.
        :param query: string with the query to make a google search with
        :param max_results: maximum number of results to retrieve
        :param depth: maximum depth of crawling
        """
        self.query = query
        self.max_results = max_results
        self.stop = max_results
        self.depth = depth
        self.gsearch = search(self.query, tld="co.in", num=self.max_results,
                              stop=self.stop, pause=2)
        self.landing_page = self.query
        self.pages = []
        self.text = []

    @property
    def landing_page(self):
        """
        Getter for landing page
        :return: assigned landing page string
        """
        return self.__landing_page

    @landing_page.setter
    def landing_page(self, query):
        """
        Setter for landing page
        :param query: Requested query
        """
        # TODO: make it more generizable
        if 'www' in query:
            # query is the landing page
            url = next(self.gsearch)
            line = 'Crawling the landing page'
            print(line)
            print('=' * len(line))
            results = [dask.delayed(self.read_url)(u) for u in self.crawl(url)]
            out = dask.compute(*results)
            self.__landing_page = ' '.join(out)
        else:
            self.__landing_page = None

    @property
    def text(self):
        """
        Getter for text attribute
        :return: Assigned text
        """
        return self.__text

    @text.setter
    def text(self, _):
        """
        Setter for the attribute text
        """
        line = 'Crawling Google results'
        print(line)
        print('=' * len(line))
        results = [dask.delayed(self.search_google)(url) for url in
                   self.gsearch]
        out = dask.compute(*results)
        self.__text, self.pages = zip(*out)

    def crawl(self, url):
        """
        Class' crawler up to self.depth. This function retrieve all the urls
        inside the parent url
        :param url: Url to crawl
        :return: local urls
        """
        new_urls = deque([url])
        processed_urls = set()
        local_urls = set()
        foreign_urls = set()
        broken_urls = set()
        count = 0
        while len(new_urls) != 0 and count <= self.depth:
            count += 1
            url = new_urls.popleft()
            processed_urls.add(url)
            # avoid subscription urls, javascript executables, videos, and help
            unwanted = ['rss/', 'javascript', 'comment', '@', 'sign_in',
                        'sign_up', 'www.youtube.', 'help', '/author',
                        '/search/', '/feed/', '?']
            # avoid subscription, help, or ideogram urls
            if any([i.lower() in url.lower() for i in unwanted]) or \
                    self.has_ideograms(url):
                continue
            print("\tProcessing %s" % url)
            try:
                response = requests.get(url)
            except(
                    requests.exceptions.MissingSchema,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.InvalidURL,
                    requests.exceptions.InvalidSchema,
                    requests.exceptions.Timeout):
                broken_urls.add(url)
                continue
            parts = urlsplit(url)
            base = "{0.netloc}".format(parts)
            strip_base = base.replace("www.", "")
            base_url = "{0.scheme}://{0.netloc}".format(parts)
            path = url[:url.rfind('/') + 1] if '/' in parts.path else url
            # TODO: change beatiful soup with lxml package?
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
        urls = self.crawl(url)
        docs = ' '.join([self.read_url(u) for u in urls])
        return docs, url

    @staticmethod
    def has_ideograms(string):
        """
        Identify if any ideograms from Chinese, Japanese, or Korean are
        present in string. It is not comprehensive.
        :param string: string to assess
        :return: Whether the string has or has not ideograms
        """
        # Unicode codes for Chinesse, japanesse and Korean ideograms I will use
        # this to filter out web pages that might still contain them
        unicode_ideograms = r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff' \
                            r'\uf900-\ufaff\uff66-\uff9f]+'
        find_ideograms = re.findall(unicode_ideograms, string)
        if find_ideograms:
            return True
        else:
            return False

    @staticmethod
    def read_url(url):
        """
        Get url html content read it and clean it
        :param url: url to process
        :return: String of cleaned text
        """
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
                text = [simple_preprocess(x.get_text(), deacc=True)
                        for x in soup.find_all('main')]
                return ' '.join(text)
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
    Process a set of documents using NLP. Text summarization through PageRank
    keyword processing, and TF-IDF, word2vec or LDA for text processing and
    topic modelling
    """
    tokenizer = RegexpTokenizer(r'\w+')
    cores = multiprocessing.cpu_count()
    nlp = spacy.load('en', disable=['ner', 'parser'])
    clean = None
    keywords = None
    vocabulary = None
    topics = None
    landing_kw = None
    opt = dict(deacc=True, scores=True)

    def __init__(self, docs, stats, landing_doc, max_df=0.9, min_df=0.01,
                 max_features=100, n_keywords=10, model='word2vec'):
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
        df_opt = dict(skiprows=[0, 1], encoding=detect_encoding(stats),
                      sep='\t')
        self.gkp = pd.read_csv(stats, **df_opt)
        self.n = n_keywords
        self.docs = docs
        self.landing_doc = landing_doc
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.pre_keywords = docs
        self.text_counts = docs
        self.model = model

    @property
    def docs(self):
        """
        Getter for the docs attribute
        :return: a set instance of docs
        """
        return self.__docs

    @docs.setter
    def docs(self, docs):
        self.__docs = [x.strip() for x in docs if x.strip()]

    @property
    def pre_keywords(self):
        """
        Getter of pre_keywords attribute
        :return: set value of pre_keywords
        """
        return self.__pre_keywords

    @pre_keywords.setter
    def pre_keywords(self, _):
        """
        Setter of pre_keywords
        :param docs: list of strings with the documents to analyze
        """
        cleaned = [list(textcleaner.tokenize_by_word(x)) for x in
                   self.clean_it(return_it=True)]
        min_count = max(1, int(len(cleaned) * self.min_df))
        ngrams = self.make_ngrams(cleaned, min_count, self.nlp)
        # ngrams = [self.make_ngrams(tokens, min_count, self.nlp) for tokens in
        #           cleaned if tokens]
        self.__pre_keywords = [keywords(x, **self.opt) for x in ngrams]

    @property
    def landing_doc(self):
        """
        Getter for landing doc
        :return: set landing document
        """
        return self.__landing_doc
    @landing_doc.setter
    def landing_doc(self, landing_doc):
        """
        Setter of landing_doc and laning_kw attributes
        :param landing_doc: text of landing page
        """
        self.__landing_doc = landing_doc
        self.landing_kw = keywords(self.landing_doc, **self.opt)

    @property
    def text_counts(self):
        """
        Text counts (word frequency vector) attribute getter
        :return: set attribute
        """
        return self.__text_counts

    @text_counts.setter
    def text_counts(self, _):
        """
        Text counts (word frequency vector) attribute setter, it will also
        populate the vocabulary attribute
        """
        vectorize = CountVectorizer(max_df=self.max_df, stop_words=stopwords,
                                    lowercase=True,
                                    max_features=self.max_features)
        self.__text_counts = vectorize.fit_transform(self.docs)
        self.vocabulary = vectorize.get_feature_names()
        print('Top 10 words in vocabulary')
        print(list(vectorize.vocabulary_.keys())[:10])

    def clean_it(self, return_it=False):
        """
        Process and clean documents populating the clean attribute with a
        dataframe of document per row or return a list of cleaned strings

        :param return_it: whether to return the list or create a dataframe
        """
        txt = [self.cleaning(doc) for doc in self.nlp.pipe(
            (pre_clean(x) for x in self.docs), batch_size=5000, n_threads=-1)]
        if return_it:
            return txt
        self.clean = pd.DataFrame({'clean': txt}).dropna().drop_duplicates()

    def word2vec(self):
        """
        Perform word2vec processing
        :return: word2vec instance
        """
        self.clean_it()
        w2v = Word2Vec(min_count=20, window=2, size=300, sample=6e-5,
                       alpha=0.03, min_alpha=0.0007, negative=20,
                       workers=self.cores - 1)
        sent = [simple_preprocess(row) for row in self.clean['clean']]
        min_count = max(1, int(len(sent) * self.min_df))
        sentences = self.make_ngrams(sent, min_count, self.nlp)
        w2v.build_vocab(sentences + self.gkp.Keyword.tolist(),
                        progress_per=10000)
        w2v.train(sentences, total_examples=w2v.corpus_count, epochs=30,
                  report_delay=1)
        w2v.init_sims(replace=True)
        return w2v

    def lda(self):
        """
        Perform LDA (Drichlet Allocation) on the cleaned documents using the
        Mallet algorithm (more accurate)
        :return: vocabulary, corpora and ldamallet instance
        """
        self.clean_it()
        sent = [simple_preprocess(row) for row in self.clean['clean']]
        min_count = max(1, int(len(sent) * self.min_df))
        sentences = self.make_ngrams(sent, min_count, self.nlp)
        id2word = Dictionary(sentences)
        corpus = [id2word.doc2bow(text) for text in sentences]
        # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
        mallet_path = abspath(join(dirname(__file__), pardir, 'resources'))
        mallet_path = join(mallet_path, 'mallet-2.0.8', 'bin', 'mallet')
        ldamallet = wrappers.LdaMallet(mallet_path, corpus=corpus,
                                       num_topics=self.n, id2word=id2word)
        lda = wrappers.ldamallet.malletmodel2ldamodel(ldamallet,
                                                      gamma_threshold=0.001,
                                                      iterations=50)
        self.topics = (ldamallet.show_topics(formatted=False))
        coherence_model_ldamallet = CoherenceModel(model=ldamallet,
                                                   texts=sentences,
                                                   dictionary=id2word,
                                                   coherence='c_v')
        self.coherence_ldamallet = coherence_model_ldamallet.get_coherence()
        print('\nCoherence Score: ', self.coherence_ldamallet)
        self.keywords = self.format_topics_sentences(
            ldamodel=ldamallet, corpus=corpus, texts=sentences)
        return id2word, corpus, lda

    @staticmethod
    def cleaning(doc):
        """
        Removing and lemmatizing tokenized document
        :param doc: tokenized document
        :return: string with clean lemmas
        """
        if isinstance(doc, str):
            doc = doc.strip().split()
        elif isinstance(doc, spacy.tokens.doc.Doc):
            pass
        else:
            raise NotImplementedError
        return ' '.join([token.lemma_ for token in doc if not token.is_stop])

    @staticmethod
    def make_ngrams(sent, min_count, nlp):
        """
        Create bi and trigrams using spacy nlp and gensim Phrases
        :param sent: list of preprocessed corpora
        :param min_count: minimum number of words to be taken into account
        :param nlp: instance of spacy nlp
        :return: sentences
        """
        if isinstance(sent, GeneratorType):
            sent = list(sent)
        bigram = Phrases(sent, min_count=min_count, threshold=1)
        trigram = Phrases(bigram[sent], min_count=min_count, threshold=1)
        bigram_mod = Phraser(bigram)
        trigram_mod = Phraser(trigram)
        ngrams = [list(set(bigram_mod[x] + trigram_mod[x])) for x in sent]
        #ngrams = [x for y in ngrams for x in y if x not in stopwords]
        try:
            if isinstance(ngrams[0], str):
                out = [token.lemma_ for token in nlp(" ".join(ngrams))
                       if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']
                       and not token.is_stop]
            else:
                out = [[token.lemma_ for token in nlp(" ".join(gram)) if
                        token.pos_
                        in ['NOUN', 'ADJ', 'VERB',
                            'ADV'] and not token.is_stop]
                       for gram in ngrams]
        except IndexError:
            out = [token.lemma_ for token in nlp(" ".join(ngrams))
                   if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']
                   and not token.is_stop]

        return out

    @staticmethod
    def format_topics_sentences(ldamodel, corpus, texts):
        """
        Get LDA results into a nice table
        https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
        :param ldamodel: Trained model instace
        :param corpus: list of bows
        :param texts: original training set of ngrams
        :return:
        """
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series(
                        [int(topic_num), round(prop_topic, 4),
                         topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution',
                                  'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return sent_topics_df

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
        Process the texts using NLP's tf_idf
        """
        self.tfidf_transformer = TfidfTransformer(smooth_idf=True,
                                                  use_idf=True)
        self.tfidf_transformer.fit(self.text_counts)
        df_idf = pd.DataFrame(self.tfidf_transformer.idf_,
                              index=self.vocabulary, columns=["weights"])
        self.keywords = df_idf.nlargest(n=self.n, columns="weights")
        return df_idf


def main(query, stats, max_results, depth, max_df, min_df, max_features,
         n_keywords, plot_fn, model, email=None):
    """
    Main execution of the code

    :param query: Query to perform Google search with
    :param stats: Google Keyword planner output
    :param max_results: Maximum number of result to get from Google search
    :param depth: Maximum crawling depth
    :param max_df: When building the vocabulary ignore terms that have a
        document frequency strictly higher than the given threshold
    :param min_df: When building the vocabulary ignore terms that have a
        document frequency strictly lower than the given threshold
    :param max_features: maximum number of word feature to get
    :param n_keywords: Number of keywords
    :param plot_fn: Plot filename or None
    :param model: Model to use. Available: tf_idf, lda, word2vec
    :param email: email to send the results to
    :return: dataframe with keywords and their stats
    """
    path = dirname(stats)
    page_file = join(path, 'pages.dmp')
    land_file = join(path, 'landing.dmp')
    if isfile(page_file):
        with open(page_file) as p, open(land_file) as lf:
            text = [line for line in p]
            land = ' '.join([line for line in lf])
    else:
        now = time.time()
        pages = GetPages(query, max_results, depth)
        elapsed = (time.time() - now) / 60
        print("Crawling done in", elapsed, 'minutes')
        to_str = lambda x: x if isinstance(x, str) else '\n'.join(x)
        text = pages.text
        land = to_str(pages.landing_page)
        with open(page_file, 'w') as p, open(page_file, 'w') as lf:
            p.write(to_str(text))
            lf.write(land)
    iw = IdentifyWords(text, stats, land, max_df, min_df, max_features,
                       n_keywords, model=model)
    if model is not None:
        iw.__getattribute__(model)()
        with open('iw.pkcl', 'wb') as p:
            dill.dump(iw, p)
        if plot_fn is not None:
            iw.frequency_explorer(plot_fn)
        print('pre-KeyWords\n', iw.pre_keywords)
        print('Landing pages KeyWords\n', iw.landing_kw)
        print('%s Keywords\n' % model, iw.keywords)
        print("Done")
    else:
        landing_keywords, landing_page_rank = zip(*iw.landing_kw)
        q = np.quantile(landing_page_rank, 0.25)
        # remove poorly connected keywords (likely unimportant)
        idx = [x > q for x in landing_page_rank]
        landing_keywords = np.array(landing_keywords)[idx].tolist()
        gkp_keywords = iw.gkp.Keyword.to_list()
        inferred_keywords = [x[0] for y in iw.pre_keywords if y for x in y]
        inferred_pagerank = [x[1] for y in iw.pre_keywords if y for x in y]
        new_q = np.quantile(inferred_pagerank, 0.25)
        inferred_idx = [x > new_q for x in inferred_pagerank]
        inferred_keywords = np.array(inferred_keywords)[inferred_idx].tolist()
        combined = list(set(landing_keywords + inferred_keywords))
        combined = [x.replace('_', ' ') for y in combined for x in y.split()
                    if x.replace('_', ' ') not in stopwords][:2500]
        if email is not None:
            user = email.split('@')
            fn = join(path, '%s_stats.csv' % user)
        else:
            fn = join(path, 'df_checkpoint.csv')
        if not isfile(fn):
            df_others = get_stats(combined, dfs_login, dfs_pass, 'scraped')
            gkp_kw = [x.replace('_', ' ') for y in gkp_keywords
                      for x in y.split() if x.replace('_', ' ') not in
                      stopwords][:2500]
            df_gkp = get_stats(gkp_kw, dfs_login, dfs_pass, 'GKP')
            df = pd.concat([df_others, df_gkp], ignore_index=True)
            df.to_csv(fn, index=False)
        if email is not None:
            # set bokeh server and send email
            raise NotImplementedError
        return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PROG', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('query', help='Original query')
    parser.add_argument('stats', help='Google Keyword Planner filename')
    parser.add_argument('-x', '--max_results', type=int, default=20,
                        help='Maximum number of results from google query')
    parser.add_argument('-d', '--depth', type=int, default=5,
                        help='Max depth to crawl each result')
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
    parser.add_argument('-p', '--plot_fn', type=str, default=None,
                        help='Name of the plot file')
    parser.add_argument('-M', '--model', default=None,
                        help='NLP model to fit. Available tf_idf, word2vec '
                             'and lda')
    args = parser.parse_args()
    main(query=args.query, stats=args.stats, max_results=args.max_results,
         depth=args.depth, max_df=args.max_df, plot_fn=args.plot_fn,
         model=args.model, max_features=args.max_features, min_df=args.min_df,
         n_keywords=args.n_keywords, email=None)
