"""
**keyword_stats.py**
Use Data For SEO API to get keywords statistics such as daily cost and daily
impressions
**Copyright** 2019  Jose Sergio Hleap
"""
# Imports ---------------------------------------------------------------------
from ADvantage.scripts._utils import *
import argparse
from base64 import b64encode
from http.client import HTTPSConnection
from pandas import DataFrame
from os.path import isfile

# File Attributes -------------------------------------------------------------
__author__ = 'Jose Sergio Hleap'
__version__ = version
__email__ = 'jshleap@gmail.com'


# Utility Functions -----------------------------------------------------------
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
        df = DataFrame(results[0].values(), index=results[0].keys())
    except KeyError:
        df = DataFrame(results.values(), index=results.keys())
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
            return json.loads(response.read().decode())
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
            data_str = json.dumps(data)
        return self.request(path, 'POST', data_str)


def main(keywords, dfs_login, dfs_pass, outfn, label=None):
    """
    Main funtion of keywords stats. This function will execute keyword_stats
    and produce a csv table

    :param keywords: List of keywords or filename with keywords
    :param dfs_login: Login for Data For SEO API
    :param dfs_pass: Password for Data For SEO API
    :param label: Label for the current set of words
    :param outfn: Outfile name of the table
    :return: Dataframe with data for SEO stats
    """
    if isfile(keywords):
        with open(keywords) as infile:
            keywords = infile.read().strip().split()
    df = get_stats(keywords, dfs_login, dfs_pass, label)
    df.to_csv(outfn, index=False)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PROG', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-k, --keywords', nargs='+',
                        help='Keywords to get stats from. If a single item and'
                             ' is a file in the current working directory, it '
                             'will assume a keywords file with one keyword per'
                             ' line')
    parser.add_argument('-l', '--dfs_login', default=None,
                        help='Login for the DataForSEO API')
    parser.add_argument('-p', '--dfs_pass', default=None,
                        help='Password for the DataForSEO API')
    parser.add_argument('-L', '--label', default=None,
                        help='Label for keyword set')
    parser.add_argument('-o', '--outfn', default=None,
                        help='Filename of resulting table')
    args = parser.parse_args()
    main(keywords=args.keywords, dfs_login=args.dfs_login, outfn=args.outfn,
         dfs_pass=args.dfs_pass, label=args.label)