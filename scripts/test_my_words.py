"""
**test_my_words.py**
**Copyright** 2019  Jose Sergio Hleap
"""
__author__ = 'Jose Sergio Hleap'
__version__ = '0.1b'
__email__ = 'jshleap@gmail.com'

from os.path import join, abspath, pardir, dirname
from googleads import adwords


def get_google_stats(query, adwords_client, PAGE_SIZE=1, offset=0):
    selector = {'ideaType': 'KEYWORD',
                'requestType': 'STATS',
                'requestedAttributeTypes': [
                    'AVERAGE_CPC', 'CATEGORY_PRODUCTS_AND_SERVICES',
                    'COMPETITION', 'EXTRACTED_FROM_WEBPAGE', 'IDEA_TYPE',
                    'KEYWORD_TEXT', 'SEARCH_VOLUME',
                    'TARGETED_MONTHLY_SEARCHES'],
                'paging': dict(startIndex=str(offset),
                               numberResults=str(PAGE_SIZE)),
                'searchParameters': dict(
                    xsi_type='RelatedToQuerySearchParameter',
                    queries=query)
                }

    targeting_idea_service = adwords_client.GetService('TargetingIdeaService',
                                                   version='v201809')

    page = targeting_idea_service.get(selector)
    print(page)


if __name__ == '__main__':
    query = ['machine learning']
    yaml = join(abspath(join(dirname(__file__), pardir)), 'resources',
                'googleads.yaml')
    print(yaml)
    yaml = '/Users/jshleap/Playground/Insight/HiddenKeywords/resources/googleads.yaml'
    adwords_client = adwords.AdWordsClient.LoadFromStorage(yaml)
    adwords_client.SetClientCustomerId('145-662-7103')
    get_google_stats(query, adwords_client)