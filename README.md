# Analisi dell' orientamento politico, del sentiment e dell' emotions delle principali testate giornalistiche italiane 

Il seguente progetto ha come obbiettivo quello di analizzare l'orientamento politico e 
il sentiment/emotions
di diverse testate giornalistiche italiane attraverso l' analisi dei tweets e degli articoli.

### Obbiettivo e metodologie 

Attraverso l' estrazione di tweets e articoli per diverse testa giornalistiche italiane, 
cerchiamo di costruire un modello di classificazione capace di interpretare l' orientamento politico
attraverso un analisi della similarità fra i corpus ricavati dai tweets di partiti politici appartenenti 
a ogni area dello spettro politico, così da capire per ciascuna testata giornalistica, quale si avvicina più a essa.
Per ciascun partito, per ciascuna testata e per ciascun orientamento, inoltre, effettuiamo un 
un analisi del sentiment e dell' emotions.

### Come è articolato il progetto? 

Il progetto si compone in 3 parti generali:

* Scraping dei tweets e degli articoli
  - scraping dei tweets dei partiti politici 
  - scraping dei tweets delle testate giornalistiche 
  - scraping degli articoli collegati ai tweets per categoria 'politica' ed 'economia'
* Tokenizzazione e pulizia dei dataset ai fini di analisi
  - filtraggio dei tweets per categoria 
* Analisi e modellizzazione dei dati
  - analisi della similarità per ciascuna testata con i corpus di sinistra, di destra e neutrale
  - creazione dei modelli di classificazione 
  - analisi del sentiment e dell' emotions per 
  ciascun partito, testata giornalistica e orientamento politico attraverso 
  spiderplot
* Test e rappresentazioni grafiche dei modelli
  - validazione del modello 
  - rappresentazioni grafiche 

























