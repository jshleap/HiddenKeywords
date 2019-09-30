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
from os.path import join, pardir, abspath, dirname, isfile
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
from gensim.models import Word2Vec, wrappers, CoherenceModel
from gensim.utils import simple_preprocess
from gensim.summarization import keywords
from gensim.corpora import Dictionary
import requests.exceptions
from urllib.parse import urlsplit
from collections import deque
from chardet.universaldetector import UniversalDetector
import dask
import time
import dill
from http.client import HTTPSConnection
from base64 import b64encode
from json import loads
from json import dumps
import numpy as np
from HiddenKeywords.scripts.__dataforseo_credentials__ import *

unicode_ideograms = r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff' \
                    r'\uff66-\uff9f]+'

plt.style.use('ggplot')
parent_dir = dirname(dirname(abspath(__file__)))
with open(join(parent_dir, 'resources', 'stopwords.txt')) as stpw:
    stopwords = set(stopwords.words('english'))
    stopwords.update(simple_preprocess(stpw.read().strip().replace('\n', ' '),
                                       deacc=True))
    stopwords.update(list(whitespace) + list(punctuation))


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
    text = [x for x in simple_preprocess(text,deacc=True) if x not in
            stopwords]
    return ' '.join(text)


class RestClient:
    domain = "api.dataforseo.com"

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def request(self, path, method, data=None):
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
        return self.request(path, 'GET')

    def post(self, path, data):
        if isinstance(data, str):
            data_str = data
        else:
            data_str = dumps(data)
        return self.request(path, 'POST', data_str)


class GetPages(object):
    def __init__(self, query, max_results=20, stop=5, depth=3):
        self.query = query
        self.max_results = max_results
        self.stop = max_results
        self.depth = depth
        self.has_ideograms = lambda x: True if re.findall(unicode_ideograms, x
                                                          ) else False
        self.gsearch = search(self.query, tld="co.in", num=self.max_results,
                              stop=self.stop, pause=2)
        self.landing_page = next(self.gsearch)
        self.pages = []
        self.text = []

    @property
    def landing_page(self):
        return self.__landing_page

    @landing_page.setter
    def landing_page(self, url):
        line = 'Crawling the landing page'
        print(line)
        print('='*len(line))
        results = [dask.delayed(self.read_url)(u) for u in self.crawl(url)]
        out = dask.compute(*results)
        self.__landing_page = ' '.join(out)

    @property
    def text(self):
        return self.__text

    @text.setter
    def text(self, _):
        line = 'Crawling Google results'
        print(line)
        print('=' * len(line))
        results = [dask.delayed(self.search_google)(url) for url in
                   self.gsearch]
        out = dask.compute(*results)
        self.__text, self.pages = zip(*out)

    def crawl(self, url):
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
        urls = self.crawl(url)
        docs = ' '.join([self.read_url(u) for u in urls])
        return docs, url

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
                text = [pre_clean(x.get_text()) for x in soup.find_all('main')]
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
    Use NLP's TF-IDF to identify keywords of a set of documents
    """
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
        opt = dict(deacc=True, scores=True)
        df_opt = dict(skiprows=[0, 1], encoding=detect_encoding(stats),
                      sep='\t')
        self.gkp = pd.read_csv(stats, **df_opt)
        self.cores = multiprocessing.cpu_count()
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.vocabulary = None
        self.n = n_keywords
        self.docs = docs
        self.landing_doc = landing_doc
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.clean = None
        self.keywords = None
        self.pre_keywords = [keywords(x, **opt) for x in self.docs]
        self.landing_kw = keywords(landing_doc, **opt)
        self.nlp = spacy.load('en', disable=['ner', 'parser'])
        self.text_counts = docs
        self.model = model
        self.topics = None

    @property
    def text_counts(self):
        return self.__text_counts

    @text_counts.setter
    def text_counts(self, _):
        vectorize = CountVectorizer(max_df=self.max_df, stop_words=stopwords,
                                    lowercase=True,
                                    max_features=self.max_features)
        self.__text_counts = vectorize.fit_transform(self.docs)
        self.vocabulary = vectorize.get_feature_names()
        print('Top 10 words in vocabulary')
        print(list(vectorize.vocabulary_.keys())[:10])

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
        self.clean_it()
        w2v = Word2Vec(min_count=20, window=2, size=300, sample=6e-5,
                       alpha=0.03, min_alpha=0.0007, negative=20,
                       workers=self.cores - 1)
        sent = [simple_preprocess(row) for row in self.clean['clean']]
        min_count = max(1, int(len(sent)*self.min_df))
        sentences = self.make_ngrams(sent, min_count, self.nlp)
        w2v.build_vocab(sentences + self.gkp.Keyword.tolist(),
                        progress_per=10000)
        w2v.train(sentences, total_examples=w2v.corpus_count, epochs=30,
                  report_delay=1)
        w2v.init_sims(replace=True)
        return w2v

    @staticmethod
    def make_ngrams(sent, min_count, nlp):
        """
        Create bi and trigrams using spacy nlp and gensim Phrases
        :param sent: list of preprocessed corpora
        :param min_count: minimum number of words to be taken into account
        :param nlp: instance of spacy nlp
        :return: sentences
        """
        bigram = Phrases(sent, min_count=min_count, threshold=100)
        trigram = Phrases(bigram[sent], threshold=100)
        bigram_mod = Phraser(bigram)
        trigram_mod = Phraser(trigram)
        ngrams = [bigram_mod[doc] for doc in sent] + [
            trigram_mod[bigram_mod[doc]] for doc in sent]
        return [[token.lemma_ for token in nlp(" ".join(gram)) if token.pos_ in
                 ['NOUN', 'ADJ', 'VERB', 'ADV']] for gram in ngrams]

    def lda(self):
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
        Process the textsusing NLP's tf_idf
        """
        self.tfidf_transformer = TfidfTransformer(smooth_idf=True,
                                                  use_idf=True)
        self.tfidf_transformer.fit(self.text_counts)
        df_idf = pd.DataFrame(self.tfidf_transformer.idf_,
                              index=self.vocabulary, columns=["weights"])
        self.keywords = self.df_idf.nlargest(n=self.n, columns="weights")
        return df_idf


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


def get_stats(keywords, dfs_login, dfs_pass):
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
    return df


def main(query, stats, max_results, depth, max_df, min_df, max_features, n_keywords,
         plot_fn, model):
    if isfile('pages.dmp'):
        with open('pages.dmp') as p, open('landing.dmp') as l:
            text = [line for line in p]
            land = ' '.join([line for line in l])
    else:
        now = time.time()
        pages = GetPages(query, max_results, depth)
        elapsed = (time.time() - now) / 60
        print("Crawling done in", elapsed, 'minutes')
        to_str = lambda x: x if isinstance(x, str) else '\n'.join(x)
        text = pages.text
        land = to_str(pages.landing_page)
        with open('pages.dmp', 'w') as p, open('landing.dmp', 'w') as l:
            p.write(to_str(text))
            l.write(land)
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
        landing_keywords, landing_pageRank = zip(*iw.landing_kw)
        q = np.quantile(landing_pageRank, 0.25)
        idx = [x > q for x in landing_pageRank]
        landing_keywords = np.array(landing_keywords)[idx].tolist()
        gkp_keywords = iw.gkp.Keyword.to_list()
        inferred_keywords = [x[0] for y in iw.pre_keywords if y for x in
                             y for y in iw.pre_keywords if x]
        inferred_pagerank = [x[1] for y in iw.pre_keywords if y for x in
                             y for y in iw.pre_keywords if x]
        new_q = np.quantile(inferred_pagerank, 0.25)
        inferred_idx = [x > new_q for x in inferred_pagerank]
        inferred_keywords = np.array(inferred_keywords)[inferred_idx].tolist()
        combined = list(set(landing_keywords + gkp_keywords + inferred_keywords
                            ))
        combined = [x.replace('_', ' ') for y in combined for x in y.split()
                    if x.replace('_', ' ') not in stopwords][:2500]
        df = get_stats(combined, dfs_login, dfs_pass)
        df = df.reset_index().rename(columns={'index': 'Keyword'})
        df.loc[df.Keyword.isin(iw.gkp.Keyword), 'source'] = 'GKP'
        df.loc[~df.Keyword.isin(iw.gkp.Keyword), 'source'] = 'scraped'
        df.to_csv('df_checkpoint.csv', index=False)
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
         n_keywords=args.n_keywords)
