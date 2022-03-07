import pickle
import string
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def remove_stopwords(text):
    """
        A function that tokenizes, filters the punctuation and the stopwords
        :return: a clean list of tokens
    """
    clean_text = []
    italian_stopwords = stopwords.words('italian')
    for token in nltk.word_tokenize(text):
        if token.lower() not in italian_stopwords:
            if token not in string.punctuation and token.isalpha():
                clean_text.append(token.lower())
            else:
                pass
        else:
            pass
    return str(clean_text)


def random_forest():
    '''
    training and testing of the random forest model on party tweets with the dataset of 6000 observations per orientation
    '''

    X = df_to_train['Text']
    y = df_to_train['Orientamento']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    vectorizer = TfidfVectorizer(
        strip_accents='unicode',
        decode_error='ignore',
        lowercase=False,
        norm='l2'
    )
    pipeline = Pipeline([('vect', vectorizer),
                         ('clf', RandomForestClassifier(
                             n_estimators=500,
                             max_features=None,
                             n_jobs=-1,
                             random_state=910,
                             oob_score=True,
                         ))])

    # fitting our model and save it in a pickle for later use
    model = pipeline.fit(X_train, y_train)
    with open('RandomForest.pickle', 'wb') as f:
        pickle.dump(model, f)
    ytest = np.array(y_test)
    print(f'matrice di confusione:{confusion_matrix(ytest, model.predict(X_test))}')
    print(f'report :\n {classification_report(ytest, model.predict(X_test))}')


def random_forest_on_medias():
    '''
    testing of model saved on .picke file on newspaper articles
    '''
    loaded_model = pickle.load(open('RandomForest.pickle', 'rb'))

    articles_libero = [remove_stopwords(str(tweet)).replace('[', '').replace(']', '') for tweet in
                       df_medias['Text'].loc[df_medias['Username'].isin(['Libero_official'])]]
    res_libero = loaded_model.predict(articles_libero)

    ilgiornale_articles = [remove_stopwords(str(tweet)).replace('[', '').replace(']', '') for tweet in
                           df_medias['Text'].loc[df_medias['Username'].isin(['ilgiornale'])]]
    res_ilgiornale = loaded_model.predict(ilgiornale_articles)

    ilfatto_articles = [remove_stopwords(str(tweet)).replace('[', '').replace(']', '') for tweet in
                        df_medias['Text'].loc[df_medias['Username'].isin(['fattoquotidiano'])]]
    res_ilfatto = loaded_model.predict(ilfatto_articles)

    unione_articles = [remove_stopwords(str(tweet)).replace('[', '').replace(']', '') for tweet in
                       df_medias['Text'].loc[df_medias['Username'].isin(['UnioneSarda'])]]
    res_unione = loaded_model.predict(unione_articles)

    messaggero_articles = [remove_stopwords(str(tweet)).replace('[', '').replace(']', '') for tweet in
                           df_medias['Text'].loc[df_medias['Username'].isin(['ilmessaggeroit'])]]
    res_messaggero = loaded_model.predict(messaggero_articles)

    lastampa_articles = [remove_stopwords(str(tweet)).replace('[', '').replace(']', '') for tweet in
                         df_medias['Text'].loc[df_medias['Username'].isin(['LaStampa'])]]
    res_lastampa = loaded_model.predict(lastampa_articles)

    repubblica_articles = [remove_stopwords(str(tweet)).replace('[', '').replace(']', '') for tweet in
                           df_medias['Text'].loc[df_medias['Username'].isin(['repubblica'])]]
    res_repubblica = loaded_model.predict(repubblica_articles)

    ilpost_articles = [remove_stopwords(str(tweet)).replace('[', '').replace(']', '') for tweet in
                       df_medias['Text'].loc[df_medias['Username'].isin(['ilpost'])]]
    res_ilpost = loaded_model.predict(ilpost_articles)

    corriere_articles = [remove_stopwords(str(tweet)).replace('[', '').replace(']', '') for tweet in
                         df_medias['Text'].loc[df_medias['Username'].isin(['Corriere'])]]
    res_corriere = loaded_model.predict(corriere_articles)
    '''
    plotting results
    '''
    newspapers_list = [res_libero, res_ilgiornale, res_ilfatto, res_unione, res_messaggero,
                       res_lastampa, res_repubblica, res_ilpost, res_corriere]
    left_values = list(np.count_nonzero(item == 'sinistra') for item in newspapers_list)
    neutro_values = list(np.count_nonzero(item == 'neutro') for item in newspapers_list)
    right_values = list(np.count_nonzero(item == 'destra') for item in newspapers_list)

    plt.style.use('ggplot')
    n = 9
    fig, ax = plt.subplots()
    index = np.arange(n)
    bar_width = 0.3
    opacity = 0.9
    ax.bar(index, left_values, bar_width, alpha=opacity, color='r',
           label='sinistra')
    ax.bar(index + bar_width, neutro_values, bar_width, alpha=opacity, color='b',
           label='neutro')
    ax.bar(index + 2 * bar_width, right_values, bar_width, alpha=opacity,
           color='k', label='destra')
    ax.legend(ncol=3)

    ax.set_xlabel('Giornali')
    ax.set_ylabel('Numero articoli')
    ax.set_title('Predizioni orientamento')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(('Libero', 'ilGiornale', 'ilFattoQuotidiano', 'UnioneSarda', 'ilMessaggero', 'laStampa',
                        'Repubblica', 'ilPost', 'Corriere della Sera'))
    ax.legend(ncol=3)
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('./data/tweets_parties_orient.csv', sep=',')
    df = df.dropna()

    '''creation of pandas dataframe for model training, with 6000 random observations for each orientation'''

    df_sinistra = df[df['Orientamento'].values == 'sinistra']
    df_neutro = df[df['Orientamento'].values == 'neutro']
    df_destra = df[df['Orientamento'].values == 'destra']

    df_train_sinistra = df_sinistra.sample(n=6000, random_state=10)
    df_train_neutro = df_neutro.sample(n=6000, random_state=10)
    df_train_destra = df_destra.sample(n=6000, random_state=10)

    df_to_train = pd.concat([df_train_sinistra, df_train_neutro, df_train_destra],
                            keys=['DateTime', 'Text', 'Username', 'Orientamento'])

    tweets = [remove_stopwords(str(tweet)).replace('[', '').replace(']', '') for tweet in df_to_train['Text']]
    df_to_train['Text'] = tweets

    '''importing and testing the model in the newspaper's articles'''

    df_medias = pd.read_csv('./data/tweets_newspapers_2021_filtered.csv', sep=',')
    df_medias.dropna()
    random_forest()
    random_forest_on_medias()
