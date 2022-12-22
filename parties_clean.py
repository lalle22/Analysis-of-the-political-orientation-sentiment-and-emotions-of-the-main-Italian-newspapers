import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import CategorizedPlaintextCorpusReader, stopwords
import string
import json



class DataFrameCleaner:
    """
             A class to clean the dataframe from urls and a part of the Datetime
             Attributes
             ----------
             dataframe : pandas.DataFrame

    """

    def __init__(self, dataframe):
        self.df = dataframe

    def datetime_cleaner(self):
        """
            A method to clean irrelevant info from the datetime
            :return: a pandas.Series with the dates
        """
        dates = []
        for item in self.df.iloc[:, 0]:
            dates.append(item[:9])
        self.df.iloc[:, 0] = pd.Series(dates)

    def remove_urls(self):
        """
            A method to remove the urls from the feature's values
            :return: the dataframe without urls in the relative column
        """
        tweet_no_url = [re.sub(r"http\S+", '', item) for item in self.df.iloc[:, 1]]
        self.df.iloc[:, 1] = tweet_no_url


class CreatingCorpus:
    """
            A class to create a corpus for the political orientation
            Attributes
            ----------
            dataframe : pandas.DataFrame

    """

    def __init__(self, dataframe):
        self.df = dataframe
        pass

    def corpus_neutro(self):
        """
            A method to obtain the neutro orientation corpus
            :return: a list with the tweets of the neutro parties
        """
        usernames = ['Piu_Europa', 'forza_italia', 'Mov5Stelle']
        df_filtered = self.df[self.df.Username.isin(usernames)]
        return list(pd.Series(df_filtered['Text']))

    def corpus_right(self):
        """
            A method to obtain the right orientation corpus
            :return: a list with the tweets of the right parties
        """
        usernames = ['LegaSalvini', 'FratellidItalia', 'CasaPoundItalia']
        df_filtered = self.df[self.df.Username.isin(usernames)]
        return list(pd.Series(df_filtered['Text']))

    def corpus_left(self):
        """
            A method to obtain the left orientation corpus
            :return: a list with the tweets of the left parties
        """
        usernames = ['pdnetwork', 'liberi_uguali', 'direzioneprc']
        df_filtered = self.df[self.df.Username.isin(usernames)]
        return list(pd.Series(df_filtered['Text']))


def remove_stopwords(text):
    """
        A function that tokenizes and filters the punctuation and the stopwords
        :return: a clean list of tokens
    """
    italian_stopwords = stopwords.words('italian')
    clean_corpus = []
    for item in text:
        for token in nltk.word_tokenize(item):
            if token.lower() not in italian_stopwords:
                if token not in string.punctuation and token.isalpha():
                    clean_corpus.append(token.lower())
                else:
                    pass
            else:
                pass
    return str(clean_corpus)


if __name__ == '__main__':
    df1 = pd.read_csv('./data/tweets_partiti.csv', sep=',')
    df_test = DataFrameCleaner(df1)
    df_test.datetime_cleaner()
    df_test.remove_urls()
    conditions = [
        ((df1['Username'] == 'Piu_Europa') | (df1['Username'] == 'forza_italia') | (df1['Username'] == 'Mov5Stelle')),
        ((df1['Username'] == 'pdnetwork') | (df1['Username'] == 'liberi_uguali') | (df1['Username'] == 'direzioneprc')),
        ((df1['Username'] == 'LegaSalvini') | (df1['Username'] == 'FratellidItalia') | (
                df1['Username'] == 'CasaPoundItalia'))
    ]
    values = ['neutro', 'sinistra', 'destra']
    df1['Orientamento'] = np.select(conditions, values)

    df1.to_csv('./data/tweets_partiti_orient.csv', index=False)  # salvo il dataframe con l' orientamento

    corpus_to_filter = CreatingCorpus(df1)
    left_to_filter = corpus_to_filter.corpus_left()
    right_to_filter = corpus_to_filter.corpus_right()
    neutro_to_filter = corpus_to_filter.corpus_neutro()

    tweets_left_tokens = remove_stopwords(left_to_filter)
    tweets_right_tokens = remove_stopwords(right_to_filter)
    tweets_neutro_tokens = remove_stopwords(neutro_to_filter)

    with open('./data/corpus_sinitra.txt', 'w') as f:
        f.write(json.dumps(tweets_left_tokens))

    with open('./data/corpus_destra.txt', 'w') as f:
        f.write(json.dumps(tweets_right_tokens))

    with open('./data/corpus_neutro.txt', 'w') as f:
        f.write(json.dumps(tweets_neutro_tokens))




