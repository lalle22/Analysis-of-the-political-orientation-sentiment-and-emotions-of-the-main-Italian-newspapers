#  Analysis  

Il seguente branch effettua le analisi finali per quanto riguarda sentiment ed emotions,  sia per
quanto concerne la creazione di un classificatore per predire l' orientamento politico di una testata
giornalistica.


### Sentiment ed emotions
Nel file 'sentiment_analysis_detecting.py' si effettua l' analisi del sentiment e delle emotions 
per i tweet dei partiti politici e per quanto riguarda le testate giornalistiche, andando a utilizzare 
per le emotions, i .csv precedentemente salvati con al loro interno le feature con i sentiment estratti.

### Random forest

Per classificare le osservazioni dei partiti politici si è utilizzata una Random Forest, nel file 'rf_classifier.py',
che crea un modello andando a utilizzare 6000 osservazioni per ciascun orientamento, salvandolo nel file 'RandomForest.pickle' 
che verrà usato per effettuare le predizioni sugli articoli dei giornali

### Frequency distribution 

Si è voluto analizzare la FreqDistr per ciascun orientamento politico, creando nel file 'frequency_distr_on_corpus.py' 
i relativi cumulative plot prendendo come dati di input i corpus precedentemente creati nel branch data_cleaning.
























