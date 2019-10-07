"""
**critters.py**
Using Google search API, search similar webpages to a query, and then crawl
those pages and return the concatenated text
**Copyright** 2019  Jose Sergio Hleap
"""

import argparse
import time
from collections import deque
from os.path import isfile
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from dask import delayed, compute
from googlesearch import search
from tqdm import tqdm

# Imports ---------------------------------------------------------------------
from ADvantage.scripts._utils import *

# File Attributes -------------------------------------------------------------
__author__ = 'Jose Sergio Hleap'
__version__ = version
__email__ = 'jshleap@gmail.com'


# Utility functions -----------------------------------------------------------
def to_str(x):
    """
    Transform to string
    :param x: input to be join in string if iterable
    :return: joined string
    """
    if x:
        if not isinstance(x, str):
            x = '\n'.join(x)
    else:
        raise Exception('Nothing in input')
    return x

# Main Classes ----------------------------------------------------------------
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
        if ('www' in query) or ('/' in query) or ('.' in query):
            # query is the landing page
            url = next(self.gsearch)
            line = 'Crawling the landing page'
            print(line)
            print('=' * len(line))
            results = [delayed(self.read_url)(u) for u in self.crawl(url)]
            out = compute(*results)
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
        results = [delayed(self.search_google)(url) for url in
                   self.gsearch]
        out = compute(*results)
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
                response = requests.get(url, headers={'Accept-Encoding':
                                                          'identity'})
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
            all_links = soup.find_all('a')
            desc = 'Getting all links'
            for link in tqdm(all_links, total=len(all_links), desc=desc):
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
            desc = 'Getting local urls'
            for i in tqdm(local_urls, total=len(local_urls), desc=desc):
                if i not in new_urls and not i in processed_urls:
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
                                 '537.36',
                   'Accept-Encoding': 'identity'}
        try:
            req = requests.get(url, headers=headers)
            html_doc = req.content
            soup = BeautifulSoup(html_doc.decode('ascii', errors='ignore'),
                                 'html.parser')
            try:
                text = [simple_preprocess(x.get_text(), deacc=True)
                        for x in soup.find_all('main')]
                if not text:
                    return ''
                if isinstance(text[0], list):
                    text = [x for y in text for x in y if x]
                return ' '.join(text)
            except(AttributeError, IndexError) as e:
                print(req.status_code)
                print(html_doc)
                raise e
        except(
                requests.exceptions.MissingSchema,
                requests.exceptions.ConnectionError,
                requests.exceptions.InvalidURL,
                requests.exceptions.InvalidSchema,
                requests.exceptions.Timeout):
            return ''


def main(query, outpath, max_results=100, depth=3, prefix=''):
    page_file = join(outpath, '%s_pages.dmp' % prefix)
    land_file = join(outpath, '%s_landing.dmp' % prefix)
    if isfile(page_file):
        with open(page_file) as p, open(land_file) as lf:
            text = [line for line in p]
            land = ' '.join([line for line in lf])
        pages = None
    else:
        now = time.time()
        pages = GetPages(query, max_results, depth)
        elapsed = (time.time() - now) / 60
        print("Crawling done in", elapsed, 'minutes')
        text = pages.text
        land = to_str(pages.landing_page)
        with open(page_file, 'w') as p, open(land_file, 'w') as lf:
            p.write(to_str(text))
            lf.write(land)
    return text, land, pages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PROG', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('query', help='Query to search for. For ADvantage, '
                                      'this is your landing page url')
    parser.add_argument('outpath', help='Path to where outputs are to be '
                                        'stored')
    parser.add_argument('-m', '--max_results', type=int, default=100,
                        help='Maximum number of results from google query')
    parser.add_argument('-d', '--depth', type=int, default=3,
                        help='Max depth to crawl each result')

    args = parser.parse_args()
    main(query=args.query, outpath=args.outpath, max_results=args.max_results,
         depth=args.depth)
