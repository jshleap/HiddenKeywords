# ADvantange
If you have tried to push forward a SEO campaign in a field  that moves
quickly, you might find that tools such as Google Keyword Planner (GPK)
might not give you the most comprehensive set of keywords to select
from. In fact, Google itself have manifested that over 15% of queries 
submitted have never been seen before by the search engine, and therefore
not available for the GKP ranking. The ADvantage app fills this gap by 
leveraging the information of your landing page, and links therein, to 
search for related **current** content to construct an up-to-date text 
corpora. It then uses Natural Language Processing (NLP) text 
summarization through PageRank. This will leave you with two long sets of
keywords: The GKP-suggested set, and the extended ADvantage set. 
ADvantage go then to optimize the best basket of words to minimize your
daily cost and cost per click (CPC), while maximizing daily impressions,
ad index position, and daily clicks. ADvantage does this through 
combinatorial optimization using the branch and bound algorithm. This 
optimization will guarantee you the best set of words (given the 
statistics) even if no better words are found through web scraping.

In this README you will find:
1. [Think of the cost](#think-of-the-cost): Brief explanation of the 
business value of ADvantage
2. [How does it work?](#how-does-it-work): A more detailed explanation of
the methods used by ADvantage
3. [Credentials for third party API](#credentials-for-third-party-api):
previous to setting up the app, make sure you have these credentials


## Think of the cost
Google ads work by bidding on keywords. This means that in order to have
your campaign to be on top of the list when the keyword has been engaged, 
you will need to outbid your competitors for that specific set of keywords.
This drives the prices of certain words up, while leaving related, yet 
potentially useful words out for grabs. This tool aims to help you reduce
the cost of your campaigns while keeping track of new topics and 
associated keywords. ADvantage will provide you with an interactive 
dashboard, where you can update the daily cost you are allowed (or want)
to spend. Once the desired basket of words of your choice have been 
identified, you can download the associated table for you to proceed 
with the campaign.

## How does it work?
ADvantage works in three modules:
1. [Get similar content](#get-similar-content)
2. [Scrape and crawl](#scape-and-crawl)
3. [Combinatorial optimization](#combinatorial-optimization)

###  Get similar content
<img src="img/get_corpora.png" >
### Scrape and crawl
### Combinatorial optimization

## Credentials for third party API
To get the apropriate statistics of the keywords, ADworks required the 
use of a third party (paid) API called [Data for SEO](www.dataforseo.com
). 
In order to effectively use ADvantage, you need to create an account and
provide the credentials during the [setup process](#setup).




