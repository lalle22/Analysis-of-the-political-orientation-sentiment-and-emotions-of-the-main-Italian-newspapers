import os
import sys
import time
import snscrape.modules.twitter as sntwitter
import pandas as pd


class ScraperPartiesTweets:
    """
          A class to represent a tweets scraper for parties.

          Attributes
          ----------
          start : str
              Scraping start date
          until : str
               Scraping end date
          user : str
              Twitter username
          max_tweet : int
               Default value of tweets' number

          Methods
          -------
          scraping_tweet_to_df():
               A method to scrape tweets with the attributes established and creating a pd.DataFrame
               with DateTime, Text, Username of the single tweet
          """
    def __init__(self, start, until, user, max_tweet):
        self.start = start
        self.until = until
        self.user = user
        self.max_tweet = max_tweet

    def scraping_tweet_to_df(self):
        tweets_list = []
        try:
            for i, tweet in enumerate(
                    sntwitter.TwitterSearchScraper(
                        f'from:{self.user} since:{self.start} until:{self.until}').get_items()):
                if i > self.max_tweet:
                    break
                else:
                    tweets_list.append([tweet.date, tweet.content, tweet.username])
            return pd.DataFrame(tweets_list, columns=['DateTime', 'Text', 'Username'])
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(str(e), fname, exc_tb.tb_lineno)
            return pd.DataFrame(tweets_list, columns=['DateTime', 'Text', 'Username'])



if __name__ == '__main__':
    datestart = '2016-01-01'
    datefinish = '2021-08-31'
    tweet_num = 3000

    tw_pd = ScraperPartiesTweets(datestart, datefinish, 'pdnetwork', tweet_num).scraping_tweet_to_df()
    print('pd done...pause')
    time.sleep(1800)
    tw_leu = ScraperPartiesTweets(datestart, datefinish, 'liberi_uguali', tweet_num).scraping_tweet_to_df()
    print('leu done...pause')
    time.sleep(1800)
    tw_rifondazione = ScraperPartiesTweets(datestart, datefinish,'direzioneprc', tweet_num).scraping_tweet_to_df()
    print('prc done...pause')
    time.sleep(1800)
    tw_forzaitalia = ScraperPartiesTweets(datestart, datefinish,'forza_italia', tweet_num).scraping_tweet_to_df()
    print('forzaitalia...pause')
    time.sleep(1800)
    tw_piueuropa = ScraperPartiesTweets(datestart, datefinish,'Piu_Europa', tweet_num).scraping_tweet_to_df()
    print('piueuropa...pause')
    time.sleep(1800)
    tw_cinquestelle = ScraperPartiesTweets(datestart, datefinish,'Mov5Stelle', tweet_num).scraping_tweet_to_df()
    print('cinquestelle done...pause')
    time.sleep(1800)
    tw_lega = ScraperPartiesTweets(datestart, datefinish,'LegaSalvini', tweet_num).scraping_tweet_to_df()
    print('lega done...pause')
    time.sleep(1800)
    tw_fratelliditalia = ScraperPartiesTweets(datestart, datefinish,'FratellidItalia', tweet_num).scraping_tweet_to_df()
    print('fratelliditalia done...pause')
    time.sleep(1800)
    tw_casapound = ScraperPartiesTweets(datestart, datefinish,'CasaPoundItalia', tweet_num).scraping_tweet_to_df()
    print('casapound done...finished')

    df_parties = pd.concat([tw_pd, tw_leu, tw_rifondazione, tw_forzaitalia,
                            tw_piueuropa, tw_cinquestelle, tw_lega,
                            tw_fratelliditalia, tw_casapound], keys=['DateTime', 'Text', 'Username'])

    df_parties.to_csv('./data/tweets_partiti.csv', index=False)
