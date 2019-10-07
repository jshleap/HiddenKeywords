"""
**gimmewords.py**
**Copyright** 2019  Jose Sergio Hleap
"""
__author__ = 'Jose Sergio Hleap'
__version__ = '0.1b'
__email__ = 'jshleap@gmail.com'

import argparse
import multiprocessing
import sys
from os.path import isfile
from os.path import pardir
from types import GeneratorType

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, wrappers, CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.summarization import keywords, textcleaner
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from ADvantage.scripts._utils import *

plt.style.use('ggplot')

# Constants -------------------------------------------------------------------
INCLUDING_FILTER = ['NN', 'JJ', 'MD', 'VB', 'NP']
EXCLUDING_FILTER = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'LS', 'MD' 'PD', 'PO',
                    'UH']
WINDOW_SIZE = 5


# Main Classes ----------------------------------------------------------------
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

    def __init__(self, docs, landing_doc, max_df=0.9, min_df=0.01,
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
        self.__pre_keywords = list(set([(x[0].strip().replace('_', ' '), x[1])
                                        for y in ngrams for x in keywords(
                ' '.join(y), **self.opt)]))

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
        Setter of landing_doc and landing_kw attributes
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
        # ngrams = [x for y in ngrams for x in y if x not in stopwords]
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


def main(text, land, max_df, min_df, max_features, n_keywords, model,
         out_fn, prefix=''):
    """
    Main function to execute locksmith
    :param text: Filename of query documents to analyze
    :param land: Filename of original documents to extend words from
    :param max_df: When building the vocabulary ignore terms that have a
    document frequency strictly higher than the given threshold
    :param min_df: When building the vocabulary ignore terms that have a
    document frequency strictly lower than the given threshold
    :param max_features: maximum number of word feature to get
    :param n_keywords: Number of keywords
    :param out_fn: Filename of outputs
    :param model: Model to use. Available: tf_idf, lda, word2vec
    :return: List of keywords and instance of IdentifyKeywords
    """
    if isinstance(text, str):
        if isfile(text):
            with open(text) as txt, open(land) as lnd:
                text = txt.read().strip()
                land = lnd.read().strip()
    elif isinstance(text, list) or isinstance(text, str) or \
            isinstance(text, tuple):
        pass
    else:
        raise NotImplementedError
    iw = IdentifyWords(text, land, max_df, min_df, max_features,
                       n_keywords, model=model)
    if model is not None:
        iw.__getattribute__(model)()
        with open('%s_iw.pkcl' % prefix, 'wb') as p:
            dill.dump(iw, p)
        if out_fn is not None:
            iw.frequency_explorer('%s_freq.pdf' % out_fn)
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
        inferred_keywords, inferred_pagerank = zip(*iw.pre_keywords)
        # inferred_keywords = [x[0] for y in iw.pre_keywords if y for x in y]
        # inferred_keywords = [x[1] for y in iw.pre_keywords if y for x in y]
        new_q = np.quantile(inferred_pagerank, 0.25)
        inferred_idx = [x > new_q for x in inferred_pagerank]
        inferred_keywords = np.array(inferred_keywords)[inferred_idx].tolist()
        combined = list(set(landing_keywords + inferred_keywords))
        combined = [x.replace('_', ' ') for y in combined for x in y.split()
                    if x.replace('_', ' ') not in stopwords]
        with open('%s_keywords.list' % out_fn, 'w') as kw:
            kw.write('\n'.join(combined))

        return combined, iw


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PROG', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', help='Filename of the query documents')
    parser.add_argument('landing', help='Filename of the landing page document'
                        )
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
    parser.add_argument('-o', '--out_fn', type=str, default='keywords',
                        help='Name of the plot file')
    parser.add_argument('-M', '--model', default=None,
                        help='NLP model to fit. Available tf_idf, word2vec '
                             'and lda')
    args = parser.parse_args()
    main(text=args.query, max_df=args.max_df, min_df=args.min_df,
         out_fn=args.plot_fn, model=args.model, max_features=args.max_features,
         n_keywords=args.n_keywords)
