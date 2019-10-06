"""
**ADvantage.py**
Leveraging Google search API, obtain related web pages to your landing page to
create a corpora to extract keywords from, using NLP TextRank Algorithm. Then,
using the branch and bound algorithm solve the combinatorial search of the best
combination of words that minimizes cost while maximizing exposure.
**Copyright** 2019  Jose Sergio Hleap
"""
# Imports ---------------------------------------------------------------------
from ADvantage.scripts.__utils__ import *
from ADvantage.scripts.keyword_stats import get_stats
from ADvantage.scripts.critters import main as crawl
from ADvantage.scripts.locksmith import main as locksmith
from pandas import read_csv, concat
from os.path import isfile
import argparse

# File Attributes -------------------------------------------------------------
__author__ = 'Jose Sergio Hleap'
__version__ = version
__email__ = 'jshleap@gmail.com'


def ad_vantage(query, stats, max_results, depth, max_df, min_df, max_features,
               n_keywords, plot_fn, model, dfs_login, dfs_pass, email=None):
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
    :param dfs_login: Login for Data For SEO API
    :param dfs_pass: Password for Data For SEO API
    :return: dataframe with keywords and their stats
    """
    path = dirname(stats)
    df_opt = dict(skiprows=[0, 1], encoding=detect_encoding(stats),
                  sep='\t')
    gkp = read_csv(stats, **df_opt)
    gkp_keywords = gkp.Keywords.tolist()
    text, land, pages = crawl(query, path, max_results, depth)
    combined, iw = locksmith(text, land, max_df, min_df, max_features,
                             n_keywords, model, plot_fn)
    combined = combined[:2500]
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
        df = concat([df_others, df_gkp], ignore_index=True)
        df.to_csv(fn, index=False)
    if email is not None:
        # set bokeh server and send email
        raise NotImplementedError
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PROG', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('query', help='Query to search for. For ADvantage, '
                                      'this is your landing page url')
    parser.add_argument('stats', help='Google Keyword Planner (GKP) output')
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
    ad_vantage(query=args.query, stats=args.stats, max_df=args.max_df,
               min_df=args.min_df, max_results=args.max_results,
               max_features=args.max_features, n_keywords=args.n_keywords,
               model=args.model, dfs_login=args.dfs_login,
               dfs_pass=args.dfs_pass, email=args.email)
