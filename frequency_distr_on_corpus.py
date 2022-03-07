import string
import nltk

translator = str.maketrans('', '', string.punctuation)

with open('./data/corpus_sinitra.txt', 'r') as r:
    corpus_sinistra = r.read().replace('[', '').replace(']', '')

corpus_sinistra = [word.translate(translator) for word in corpus_sinistra.split(',')]

print(corpus_sinistra[:20])

with open('./data/corpus_neutro.txt', 'r') as r:
    corpus_neutro = r.read().replace('[', '').replace(']', '')

corpus_neutro = [word.translate(translator) for word in corpus_neutro.split(',')]
print(corpus_neutro[:20])

with open('./data/corpus_destra.txt', 'r') as r:
    corpus_destra = r.read().replace('[', '').replace(']', '')

corpus_destra = [word.translate(translator) for word in corpus_destra.split(',')]

print(corpus_destra[:20])

words_sinistra = nltk.FreqDist(corpus_sinistra)
words_neutro = nltk.FreqDist(corpus_neutro)
words_destra = nltk.FreqDist(corpus_destra)

# words_sinistra.plot(50, cumulative=True, title='Frequenze cumulate per la sinistra')
# words_neutro.plot(50, cumulative=True, title='Frequenze cumulate per i tweets neutrali')
# words_destra.plot(50, cumulative=True, title='Frequenze cumulate per la destra')
