"""
**ADvantage.py**
Leveraging Google search API, obtain related web pages to your landing page to
create a corpora to extract keywords from, using NLP TextRank Algorithm. Then,
using the branch and bound algorithm solve the combinatorial search of the best
combination of words that minimizes cost while maximizing exposure.
**Copyright** 2019  Jose Sergio Hleap
"""

# Imports ---------------------------------------------------------------------
from ADvantage.scripts._utils import *
from ADvantage.scripts.keyword_stats import get_stats
from ADvantage.scripts.critters import main as crawl
from ADvantage.scripts.locksmith import main as locksmith
from ADvantage.scripts.__credentials__ import PASSWORD, MY_ADDRESS
from pandas import read_csv, concat
from os.path import isfile
import argparse
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# File Attributes -------------------------------------------------------------
__author__ = 'Jose Sergio Hleap'
__version__ = version
__email__ = 'jshleap@gmail.com'


# Utility functions -----------------------------------------------------------
def send_email(email, url):
    """
    Send email containing the url with the bokeh dashboard

    :param email: User email
    :param url: url to dashboard
    """
    server = smtplib.SMTP(host='smtp-mail.outlook.com',
                          port=587)  # TODO:change to gmail
    server.starttls()
    server.login(MY_ADDRESS, PASSWORD)
    message = '''
    Dear ADvantage User,
    thank you for using this app, and I hope you find it useful. Please send
    feedback through this email or through github (jshleap/ADvantage).
    your bokeh dashboard is available at %s.
    Make sure you save your exploration table since the dashboard will only be
    availale for one view. Once you close it or it reaches one day it will be
    erased.
    
    Thanks!
    ''' % url
    msg = MIMEMultipart()
    msg['From'] = MY_ADDRESS
    msg['To'] = email
    msg['Subject'] = "ADvantage Results"
    msg.attach(MIMEText(message, 'plain'))

    server.send_message(msg)


# Classes and main function ---------------------------------------------------
class MissingCredentials(Exception):
    pass


def ad_vantage(query, stats, max_results, depth, max_df, min_df, max_features,
               n_keywords, out_fn, model, dfs_login, dfs_pass, email=None):
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
    :param out_fn: output filename or None
    :param model: Model to use. Available: tf_idf, lda, word2vec
    :param email: email to send the results to
    :param dfs_login: Login for Data For SEO API
    :param dfs_pass: Password for Data For SEO API
    :return: dataframe with keywords and their stats
    """
    if dfs_login is None:
        from ADvantage.scripts.__credentials__ import dfs_login, \
            dfs_pass
        if dfs_login == '':
            raise MissingCredentials('Missing Data For SEO credentials')
    path = dirname(stats)
    if email is not None:
        user = email.split('@')[0]
        fn = join(path, '%s_stats.csv' % user)
    else:
        user = ''
        fn = join(path, 'df_checkpoint.csv')
    df_opt = dict(skiprows=[0, 1], encoding=detect_encoding(stats),
                  sep='\t')
    gkp = read_csv(stats, **df_opt)
    gkp_keywords = gkp.Keyword.tolist()
    text, land, pages = crawl(query, path, max_results, depth, prefix=user)
    combined, iw = locksmith(text, land, max_df, min_df, max_features,
                             n_keywords, model, out_fn, prefix=user)
    combined = combined[:2500]
    if not isfile(fn):
        df_others = get_stats(combined, dfs_login, dfs_pass, 'scraped')
        gkp_kw = [x.replace('_', ' ') for y in gkp_keywords
                  for x in y.split() if x.replace('_', ' ') not in
                  stopwords][:2500]
        df_gkp = get_stats(gkp_kw, dfs_login, dfs_pass, 'GKP')
        df = concat([df_others, df_gkp], ignore_index=True)
        df.to_csv(fn, index=False)
    else:
        df = read_csv(fn)
    if email is not None:
        # set bokeh server and send email
        url= '?' # TODO: set bokeh server url
        send_email(email, url)
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
    parser.add_argument('-q', '--min_df', type=float, default=0.01,
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
    parser.add_argument('-x', '--max_results', type=int, default=100,
                        help='Maximum number of results from google query')
    parser.add_argument('-d', '--depth', type=int, default=3,
                        help='Max depth to crawl each result')
    parser.add_argument('-l', '--dfs_login', default=None,
                        help='Login for the DataForSEO API')
    parser.add_argument('-p', '--dfs_pass', default=None,
                        help='Password for the DataForSEO API')
    parser.add_argument('-e', '--email', default=None,
                        help='Email to send the result to')
    args = parser.parse_args()
    ad_vantage(query=args.query, stats=args.stats, max_df=args.max_df,
               max_features=args.max_features, min_df=args.min_df,
               max_results=args.max_results, n_keywords=args.n_keywords,
               model=args.model, dfs_login=args.dfs_login, depth=args.depth,
               dfs_pass=args.dfs_pass, email=args.email, out_fn=args.out_fn)
