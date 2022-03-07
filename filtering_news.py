import pandas as pd
from urllib.parse import urlparse
from urllib.parse import parse_qs
import requests
from bs4 import BeautifulSoup
import re
import string
import nltk
from nltk.corpus import stopwords
from urllib3.connection import HTTPConnection
from newspaper import Article, Config
from newspaper.article import ArticleException


class CategorizedNews(object):
    """
    A class to filter newspapers' news.
    """

    def __init__(self, links):
        """Initializing class
        :param links: a dataframe's column
        """
        self.links = links

    def filtering_tweets(self):
        """
        A method to filter news taking the category from the url's path
        :return: political and economic news
        """
        political_economic_link = [ ]
        try:
            for link in self.links:
                parsed_url = urlparse(link)
                if ('politica' in parsed_url.path.lower()) or ('economia' in parsed_url.path.lower()) or (
                        'interni' in parsed_url.path.lower()):
                    political_economic_link.append(link)

            return political_economic_link

        except Exception as e:
            print(e)
            return political_economic_link

    def filtering_tweets_query(self):
        """
        A method to filter taking the category  news from the url's query
        :return: political and economic news
        """
        political_economic_link = [ ]
        try:
            for link in self.links:
                parsed_url = urlparse(link)
                parsed_query = parse_qs(parsed_url.query)
                for value in parsed_query.values():
                    if ('politica' in value[ 0 ]) or ('economia' in value[ 0 ].lower()) or (
                            'palazzi&potere' in value[ 0 ].lower()):
                        political_economic_link.append(link)

            return political_economic_link
        except Exception as e:
            print(e)
            return political_economic_link

    def filtering_tweets_parsed(self):
        """
        A method to filter news taking the category through the html parsing of the url.
        :return: political and economic news
        """
        political_economic_link = [ ]
        try:
            for link in self.links:
                if link != '[]':
                    page = requests.get(link.replace("'", "").replace("[", "").replace("]", "")).text
                    soup = BeautifulSoup(page, "html.parser")
                    articles = soup.find_all("article")
                    for item in articles:
                        if item.select_one("li") != None:
                            category = item.select_one("li").text
                            if (category.strip() == 'Politica') or (category.strip() == "Economia"):
                                political_economic_link.append(link)
            return political_economic_link
        except Exception as e:
            print(e)
            return political_economic_link


user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 30


class ExtractArticle(object):
    """
    A class to extract newspapers' articles and his tags
    """

    def __init__(self, links):

        """Initializing class
        :param links: a dataframe's column
        """
        self.links = links

    def extract_text(self):
        """
            A method to extract news's text
            :return: articles' text
                """
        text = [ ]
        for link in self.links:
            link = link.replace("'", "").replace("[", "").replace("]", "")
            if ('repubblica' in urlparse(link).netloc) or ('ilpost' in urlparse(link).netloc) or (
                    'liberoquotidiano' in urlparse(link).netloc) or ('ilgiornale' in urlparse(link).netloc) or (
                    'ilmessaggero' in urlparse(link).netloc) or ('corriere' in urlparse(link).netloc) or (
                    'lastampa' in urlparse(link).netloc) or ('unionesarda' in urlparse(link).netloc) or (
                    'ilfattoquotidiano' in urlparse(link).netloc):
                article = Article(link, config=config)
                try:
                    article.download()
                    article.parse()
                except ArticleException:
                    pass
                text.append(article.text)
            else:
                text.append(None)

        return text


def remove_stopwords(text):
    """
        A function that tokenizes, filters the punctuation and the stopwords
        :return: a clean list of tokens
    """
    clean_text = [ ]
    italian_stopwords = stopwords.words('italian')
    for token in nltk.word_tokenize(text):
        if token.lower() not in italian_stopwords:
            if token not in string.punctuation and token.isalpha():
                token.replace('\n', '')
                clean_text.append(token.lower())
            else:
                pass
        else:
            pass
    return clean_text


if __name__ == '__main__':
    df = pd.read_csv('./data/tweets_newspapers_2021_third_quarter.csv', sep=',')
    filtered_tweets = df[ df[ 'Link' ].isin(CategorizedNews(df[ 'Link' ]).filtering_tweets()) ]
    filtered_tweets_parsed = df[
        (df[ 'Username' ] == 'ilpost') & (df[ 'Link' ].isin(CategorizedNews(df[ 'Link' ]).filtering_tweets_parsed())) ]
    filtered_tweets_query = df[ (df[ 'Username' ] == 'fattoquotidiano') & (
        df[ 'Link' ].isin(CategorizedNews(df['Link']).filtering_tweets_query())) ]
    df.drop_duplicates(inplace=True)
    df = pd.concat([ filtered_tweets, filtered_tweets_query, filtered_tweets_parsed ])
    df[ 'DateTime' ] = [ date[ :10 ] for date in df[ 'DateTime' ] ]
    df[ 'Text' ] = [ re.sub(r"http\S+", '', item) for item in df[ 'Text' ] ]
    df[ 'TextTokenized' ] = [ remove_stopwords(text) for text in df[ 'Text' ] ]
    articles_text = ExtractArticle(df[ 'Link' ]).extract_text()
    df[ 'Article' ] = articles_text
    df = df[ df[ 'Article' ] != None ]
    df.to_csv('./data/tweets_newspapers_2021_third_quarter_filtered.csv', index=False)
