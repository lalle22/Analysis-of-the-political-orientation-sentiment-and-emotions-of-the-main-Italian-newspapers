from textblob import TextBlob
import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import text2emotion as te
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


#
def percentage(part, whole):
    """
    :param part: a part of a quantity
    :param whole: the total amount of a quantity
    :return: percentage of a quantity as a rate per 100
    """
    return 100 * float(part) / float(whole)


def evaluating_sentiment(list_sentences):
    """
    :param list_sentences: a list of single sentence of a text
    :return:sentiment of a text based on the average value of sentiments of the single sentence
    """
    positive = 0
    negative = 0
    neutral = 0
    neutral_list = [ ]
    negative_list = [ ]
    positive_list = [ ]
    for element in list_sentences:
        analyzer = SentimentIntensityAnalyzer().polarity_scores(element)
        neg = analyzer[ 'neg' ]
        neu = analyzer[ 'neu' ]
        pos = analyzer[ 'pos' ]

        if neg > pos:
            negative_list.append(element)
            negative += 1
        elif pos > neg:
            positive_list.append(element)
            positive += 1
        elif pos == neg:
            neutral_list.append(element)
            neutral += 1

        positive = percentage(positive, len(list_sentences))
        negative = percentage(negative, len(list_sentences))
        neutral = percentage(neutral, len(list_sentences))

        if (positive > neutral) and (positive > negative):
            return 'positive'
        elif (negative > neutral) and (negative > positive):
            return 'negative'
        else:
            return 'neutral'


def get_key(diz):
    """
    Finding keys' name of a list of values
    :param diz: it's a dictionary
    :return: the corresponding keys' name of the max values
    """
    keys = [ ]
    for key, value in diz.items():
        if value == max(diz.values()):
            keys.append(key)
    return keys


class SentimentAnalysisEmotionDetection(object):
    """
    Calculating sentiment and detecting emotion of an article and tweet
    """

    def __init__(self, dataframe):
        """
        Initializing class
        :param dataframe: a pandas's dataframe
        """
        self.dataframe = dataframe

    def get_article_sentiment(self):
        """
        Calculating articles' semtiment dividing text in sentences
        :return: articles'sentiment
        """
        articles = self.dataframe[ 'Article' ]
        sentences_list = [ nltk.tokenize.sent_tokenize(article, language="italian") for article in articles if article ]
        sentiment_text = [ ]
        for sent in sentences_list:
            sentiment_text.append(evaluating_sentiment(sent))
        return sentiment_text

    def get_tweet_sentiment(self):
        """
         Calculating the tweets' sentiment using TextBlob module.
        """
        tweets = self.dataframe[ 'Text' ]
        sentiment_tweets = [ ]
        for tweet in tweets:
            analysis = TextBlob(tweet)
            if analysis.sentiment.polarity > 0:
                sentiment_tweets.append('positive')
            elif analysis.sentiment.polarity == 0:
                sentiment_tweets.append('neutral')
            else:
                sentiment_tweets.append('negative')
        return sentiment_tweets

    def get_emotion_article(self):
        """
        Calculating articles' emotion using Text2emotion.
        :return: the neutral emotion or no particular emotion or the two highest emotions of articles.
        """
        articles = self.dataframe[ 'Article' ]
        emotions = [ te.get_emotion(text) for text in articles ]
        values_emotions = [ ]
        for element in emotions:
            if all(item == 0 for item in element.values()):
                values_emotions.append([ 'neutral emotion' ])
            elif max(element.values()) == 1.0:
                values_emotions.append([ name for name, val in element.items() if val == 1.0 ])
            elif len(get_key(element)) > 2:
                values_emotions.append([ 'no particular emotion' ])
            else:
                name_emotions = get_key(element)
                values_emotions.append(name_emotions)
        return values_emotions

    def get_emotion_tweets(self):
        """
        Calculating tweets' emotion using Text2emotion.
        :return: the neutral emotion or no particular emotion or the two highest emotions of tweets.
        """
        tweets = self.dataframe[ 'Text' ]
        emotions = [ te.get_emotion(tweet) for tweet in tweets ]
        values_emotions = [ ]
        for element in emotions:
            if all(item == 0 for item in element.values()):
                values_emotions.append([ 'neutral emotion' ])
            elif max(element.values()) == 1.0:
                values_emotions.append([ name for name, val in element.items() if val == 1.0 ])
            elif len(get_key(element)) > 2:
                values_emotions.append([ 'no particular emotion' ])
            else:
                name_emotions = get_key(element)
                values_emotions.append(name_emotions)
        return values_emotions


if __name__ == "__main__":
    df_medias = pd.read_csv("./data/tweets_newspapers_2021_filtered.csv", sep=',')
    df_medias = df_medias.dropna()
    df_medias[ 'SentimentText' ] = SentimentAnalysisEmotionDetection(df_medias).get_article_sentiment()
    df_medias[ 'SentimentTweet' ] = SentimentAnalysisEmotionDetection(df_medias).get_tweet_sentiment()
    df_medias[ 'EmotionsTweet' ] = SentimentAnalysisEmotionDetection(df_medias).get_emotion_tweets()
    df_medias[ 'EmotionsText' ] = SentimentAnalysisEmotionDetection(df_medias).get_emotion_article()

    sns.countplot(x='Username', hue='SentimentTweet', data=df_medias,
                  palette='pastel').set_title("Newspapers' tweets sentiment", fontdict={'fontsize': 20})
    sns.set(rc={'figure.figsize': (10, 10)})
    sns.set_style("whitegrid")

    sns.countplot(x='Username', hue='SentimentText', data=df_medias,
                  palette='Set2').set_title("Newspapers' articles sentiment", fontdict={'fontsize': 20})
    sns.set(rc={'figure.figsize': (10, 10)})
    sns.set_style("whitegrid")

    df_medias.to_csv("./data/tweets_newspapers_2021_analyzed.csv")


    df_parties = pd.read_csv("./data/tweets_parties_orient.csv", sep=',')
    df_parties = df_parties.dropna()
    df_parties[ 'SentimentTweet' ] = SentimentAnalysisEmotionDetection(df_parties).get_tweet_sentiment()
    df_parties[ 'EmotionsTweet' ] = SentimentAnalysisEmotionDetection(df_parties).get_emotion_tweets()

    categories = list(set(df_parties.EmotionsTweet.values))


    def emotions_values(df, orientamento):
        """
        A function to count emotions' values
        :param df: dataframe pandas
        :param orientamento: politician orientation
        :return: a dictionary with emotions and their counts
        """
        emotions = {}
        for emotion in categories:
            emotions[ emotion ] = len(
                df[ (df[ 'Orientamento' ] == orientamento) & (df[ 'EmotionsTweet' ] == emotion) ]) / len(
                df[ df[ 'Orientamento' ] == orientamento ])
        return emotions


    sinistra_values = emotions_values(df_parties, 'sinistra')
    neutro_values = emotions_values(df_parties, 'neutro')
    destra_values = emotions_values(df_parties, 'destra')

    dataset_sinistra = df_parties[ df_parties[ 'Orientamento' ] == 'sinistra' ]
    dataset_neutro = df_parties[ df_parties[ 'Orientamento' ] == 'neutro' ]
    dataset_destra = df_parties[ df_parties[ 'Orientamento' ] == 'destra' ]

    def spider_plot(dataset, dict_values, orient):
        """
        A function to plot a spider plot for every party
        :param dataset: party's dataframe
        :param dict_values: emotions' values
        :param orient: politician orientation
        :return: spider plot
        """
        fig = px.line_polar(dataset, r=list(dict_values.values()),
                        theta=list(set(df_parties.EmotionsTweet)), line_close=True, title="Spider plot for " + orient)
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
        fig.show()

    spider_plot(dataset_sinistra, sinistra_values, 'sinistra')
    spider_plot(dataset_neutro, neutro_values, 'neutro')
    spider_plot(dataset_destra, destra_values, 'destra')

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(sinistra_values.values()),
        theta=categories,
        name='Sinistra'
    ))
    fig.add_trace(go.Scatterpolar(
        r=list(neutro_values.values()),
        theta=categories,
        name='Neutro'
    ))
    fig.add_trace(go.Scatterpolar(
        r=list(destra_values.values()),
        theta=categories,
        name='Destra'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[ 0, 1 ]
            )),
        showlegend=True
    )
    fig.show()

    sns.countplot(x='Orientamento', hue='SentimentTweet', data=df_parties, palette='pastel').set_title(
        "Parties' tweets sentiment", fontdict={'fontsize': 20})
    sns.set(rc={'figure.figsize': (10, 10)})
    sns.set_style("whitegrid")

    df_parties.to_csv("./data/tweets_parties_orient_analyzed.csv")
