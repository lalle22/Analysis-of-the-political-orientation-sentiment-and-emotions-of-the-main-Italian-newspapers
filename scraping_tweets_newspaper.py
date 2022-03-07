import time
import csv
import requests
import snscrape.modules.twitter as sntwitter
import pandas as pd
import os
import sys


class ScraperNewsTweets(object):
    """
          A class to represent a tweets scraper.
          Attributes
          ----------
          start : str
              Scraping start date
          until : str
               Scraping end date
          user : str
              Twitter username

          Methods
          -------
          dataframe_tweets():
               A method to scrape tweets with the attributes established and creating a pd.DataFrame
               with DateTime, Id, Text, Link, Username of the single tweet
          """

    def __init__(self, start, until, user):
        """"
        Args:
        start (str): the established start for scraping
        until (str): the established end for scraping
        user (str): twitter username of newspaper

        """
        self.start = start
        self.until = until
        self.user = user

    def longurl(self, links):
        """
        Unshorting link
        :param link: tweet.outlinks
        :return: unshorted link or empty list if link is invalid
        """
        long_url = []
        try:
            for link in links:
                print(link)
                url = requests.head(link, allow_redirects=True, timeout=30).url
                long_url.append(url)
            return long_url

        except Exception as e:
            print(e)
            return long_url

    def scraping_news(self):
        """
        Scraping newspapers' tweets and saving the date, the text, the link associated with the news and username of newspaper
        :return: dataframe of the information saved
        """
        tweets_list = []
        try:
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(
                    f'from:{self.user} since:{self.start} until:{self.until}]').get_items()):
                tweets_list.append([ tweet.date, tweet.content, tweet.outlinks, self.user ])
            return pd.DataFrame(tweets_list, columns=['DateTime', 'Text', 'Link', 'Username'])

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[ 1 ]
            print(str(e), fname, exc_tb.tb_lineno)
            return pd.DataFrame(tweets_list, columns=['DateTime', 'Text', 'Link', 'Username'])

    def scraping_news_unshort(self):
        """
            Scraping newspapers' tweets and saving the date, the text,
            the link associated with the news and username of newspaper
            :return: dataframe of the information saved
        """
        tweets_list = []
        try:
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(
                    f'from:{self.user} since:{self.start} until:{self.until}]').get_items()):
                if tweet.outlinks:
                    long_url = self.longurl(tweet.outlinks)
                    tweets_list.append([ tweet.date, tweet.content, long_url, self.user ])
            return pd.DataFrame(tweets_list, columns=['DateTime', 'Text', 'Link', 'Username'])

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(str(e), fname, exc_tb.tb_lineno)
            return pd.DataFrame(tweets_list, columns=['DateTime', 'Text', 'Link', 'Username'])


if __name__ == '__main__':
    datestart = '2021-01-01'
    datefinish = '2021-03-31'
    tw_Repubblica = ScraperNewsTweets(datestart, datefinish, 'repubblica').scraping_news()
    time.sleep(600)
    tw_FattoQuotidiano = ScraperNewsTweets(datestart, datefinish, 'fattoquotidiano').scraping_news_unshort()
    time.sleep(600)
    tw_Ilgiornale = ScraperNewsTweets(datestart, datefinish, 'ilgiornale').scraping_news()
    time.sleep(600)
    tw_UnioneSarda = ScraperNewsTweets(datestart, datefinish, 'UnioneSarda').scraping_news_unshort()
    time.sleep(600)
    tw_Ilmessaggero = ScraperNewsTweets(datestart, datefinish, 'ilmessaggeroit').scraping_news()
    time.sleep(600)
    tw_Libero = ScraperNewsTweets(datestart, datefinish, 'Libero_official').scraping_news()
    time.sleep(600)
    tw_IlPost = ScraperNewsTweets(datestart, datefinish, 'ilpost').scraping_news()
    time.sleep(600)
    tw_Corriere = ScraperNewsTweets(datestart, datefinish, 'Corriere').scraping_news_unshort()
    time.sleep(600)
    tw_LaStampa = ScraperNewsTweets(datestart, datefinish,'LaStampa').scraping_news_unshort()
    df = pd.concat([tw_IlPost, tw_Libero, tw_Ilgiornale, tw_Ilmessaggero,
                     tw_FattoQuotidiano, tw_UnioneSarda, tw_Corriere, tw_LaStampa])

    df.to_csv('./data/tweets_giornali_2021_first_semester.csv', index=False)





