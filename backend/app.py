import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
import io       #viene utilizzato per il grafico 
import base64   #viene utilizzato per il grafico 


# scarico i tokenizer di nltk
nltk.download('punkt')


from flask import Flask, request, jsonify
from flask_cors import CORS
import os


app = Flask(__name__)  #questa riga crea una nuova applicazione flask 

# Permetti solo richieste dal dominio specificato
CORS(app, origins=["http://localhost:3000"])  

UPLOAD_FOLDER = "uploads" #Questa riga definisce la variabile UPLOAD_FOLDER come la cartella uploads, che sarà utilizzata per memorizzare i file che l'utente carica nel backend.

# se la cartella uploads non esiste allora la crea 
if not os.path.exists(UPLOAD_FOLDER): 
    os.makedirs(UPLOAD_FOLDER)









def prediciGenere(content):

     # PARTIZIONE DELLA TRAMA 
    #######################################################################################################

    # a questo punto bisognerebbe dividere ogni testo in 4 sezioni: introduzione 20% Sviluppo 40% climax 20% risoluzione 20%
    # Questa è una funzione a partire da una trama e dalle percentuali, restituisce un dizionario dove le chiavi sono introduzione, 
    # sviluppo climax e climax e i valori sono i testi 
    def partiziona_trama(trama, percentuali=[0.2, 0.4, 0.2, 0.2]):
        parole = trama.split()  #creazione di una lista con tutte le parole del testo
        lunghezza_trama = len(parole) #calcolo la lunghezza delle parole di tutto il testo 

    # calcolo la lunghezza dei testi per ogni sottosezione 
        lunghezza_introduzione = int(lunghezza_trama * percentuali[0])
        lunghezza_sviluppo = int(lunghezza_trama * percentuali[1])
        lunghezza_climax = int(lunghezza_trama * percentuali[2])
        lunghezza_risoluzione = int(lunghezza_trama * percentuali[3])

    # per ogni sezione creo una stringa 
        introduzione = ' '.join(parole[:lunghezza_introduzione])
        sviluppo = ' '.join(parole[lunghezza_introduzione:lunghezza_introduzione + lunghezza_sviluppo])
        climax = ' '.join(parole[lunghezza_introduzione + lunghezza_sviluppo:lunghezza_introduzione + lunghezza_sviluppo + lunghezza_climax])
        risoluzione = ' '.join(parole[lunghezza_introduzione + lunghezza_sviluppo + lunghezza_climax:lunghezza_introduzione + lunghezza_sviluppo + lunghezza_climax + lunghezza_risoluzione])

    # ritorno il dizionario 
        return {
            'introduzione': introduzione,
            'sviluppo': sviluppo,
            'climax': climax,
            'risoluzione': risoluzione
        }

    #creazione di un dizionario in cui ho il testo dell'utente partizionato 
    trama_divisa = partiziona_trama(content)

    #######################################################################################################







    #CREAZIONE DIZIONARIO EMOLEX
    #######################################################################################################

    temp = [line.strip().split('\t') for line in open('Afrikaans-NRC-EmoLex.txt').readlines()]


    emolex = {}
    for l in temp:
        try:
            key = l[0]
            vals =l[1:-1]
            vals = [int(s) for s in vals]
            emolex[key] = vals
        except ValueError:
            print(l)
    # in questo modo ho creato un dizionario chiave valore in cui per ogni parola ho una lista di valori associati ad una emozione ciascuno
    print('--------------------------------------------------')


    #######################################################################################################





    #CREAZIONE DIZIONARIO PER OGNI EMOZIONE 
    #######################################################################################################
    #pip install stanza
    import stanza
    nlpit = stanza.Pipeline('en', processors = 'tokenize, lemma, pos') #qui in realtà la posizione non mi serve


    # funzione che prende in entrata un testo e il database con le emozioni
    def creaLemmi(text):
        document = nlpit(text) # viene creato un documento processato
        lemmi = [word.lemma for word in document.iter_words()] # viene creata una lista con tutti i lemmi del testo
        return lemmi




    import nltk
    nltk.download('punkt_tab')
    from nltk.tokenize import word_tokenize
    import numpy as np

    # Funzione per calcolare le emozioni in base a NRC-emolex
    def emocount(text, lexres):
        listaLemmi= creaLemmi(text) # qui chiamo la funzione creaLemmi, quindi listaLemmi è una lista con tutti i lemmi di 'text'
        out = np.zeros(10)  # creo un array di zeri
        for lemma in listaLemmi:
            if lemma in lexres:
                e = lexres[lemma]
                out += e  # Somma le emozioni per ciascuna parola
        return out

    # Colonne per le emozioni
    emotion_labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust']

    # Funzione per aggiungere l'analisi delle emozioni per ogni parte del testo
    def analyze_sentiment(trama_divisa): # questa funzione riceve in input una riga del database
        emotions_intro = emocount(trama_divisa['introduzione'],emolex)/0.2
        emotions_sviluppo = emocount(trama_divisa['sviluppo'], emolex)/0.4
        emotions_climax = emocount(trama_divisa['climax'], emolex)/0.2
        emotions_risoluzione = emocount(trama_divisa['risoluzione'], emolex)/0.2
        #emotions_intro, emotions_svilupp... sono gli array con i valori delle emozioni 
        
    # ritorno delle serie di pandas che contengono il nome della colonna e il contenuto
        return {
        'intro_emotions': emotions_intro,
        'sviluppo_emotions': emotions_sviluppo,
        'climax_emotions': emotions_climax,
        'risoluzione_emotions': emotions_risoluzione
        }



    #creo un dizionario in cui ogni chiave(parte del testo) ha come valore la lista di emozioni 
    emotion_dict = {}

    emotion_dict = analyze_sentiment(trama_divisa)

    #######################################################################################################




    #CREAZIONE GRAFICI
    #######################################################################################################
    emotion_labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust']

    # Parti del testo
    text_parts = ['Intro', 'Sviluppo', 'Climax', 'Risoluzione']

    # Creazione del grafico
    plt.figure(figsize=(10, 6))  

    # Per ogni emozione, tracciamo una linea che rappresenta la sua intensità in ciascuna parte del testo
    for i, emotion in enumerate(emotion_labels):
        #si crea una lista che raccoglie i valori di intensità dell'emozione per ciascuna parte del testo (intro, sviluppo, climax, risoluzione).
        emotion_values = [   
            emotion_dict['intro_emotions'][i],
            emotion_dict['sviluppo_emotions'][i],
            emotion_dict['climax_emotions'][i],
            emotion_dict['risoluzione_emotions'][i]
        ]
        plt.plot(text_parts, emotion_values, marker='o', label=emotion)
        #Questa riga traccia una linea sul grafico. text_parts è una lista che contiene le etichette delle parti del testo (['Intro', 'Sviluppo', 'Climax', 'Risoluzione']),
        # emotion_values è la lista dei valori di intensità dell'emozione in quelle parti del testo.
        #marker='o' specifica che ad ogni punto della linea sarà presente un cerchio (un marcatore).
        #label=emotion etichetta la linea con il nome dell'emozione, che sarà usato nella leggenda del grafico.

    # Aggiungiamo titolo e etichette
    plt.title('Relazione tra le Parti del Testo e l\'Intensità delle Emozioni')
    plt.xlabel('Parte del Testo')
    plt.ylabel('Intensità dell\'Emozione')
    plt.legend(title='Emozioni', bbox_to_anchor=(1.05, 1), loc='upper left')  #bbox_to_anchor=(1.05, 1) posiziona la legenda in alto a destra
    plt.tight_layout() #per non far uscire la leggenda fuori dalla visualizzazione 
    plt.grid(True)  #aggiunta di una griglia al grafico 

    #salvataggio immagine 
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()

    # spiegazione:
    #img = io.BytesIO()
    # Crea un buffer di memoria (oggetto BytesIO), che permette di memorizzare i dati binari dell'immagine del grafico senza doverli scrivere su un file fisico.
    
    # plt.savefig(img, format='png')
    # Salva il grafico come immagine nel formato png nel buffer img (anziché in un file sul disco). Il formato è specificato come 'png'.
    
    # img.seek(0)
    # Ripristina il puntatore del buffer img all'inizio, in modo che i dati dell'immagine possano essere letti correttamente.
    
    # img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    # Codifica i dati binari dell'immagine nel formato base64, che è un formato di codifica utile per trasferire dati binari in un formato di testo (ad esempio, per inviarli su una rete o per visualizzarli in un browser web).
    # img.getvalue() restituisce i dati dell'immagine dal buffer.
    # base64.b64encode() codifica i dati in base64.
    # decode('utf-8') converte la codifica base64 in una stringa di testo.


    #######################################################################################################



    #PREDIZIONE
    #######################################################################################################
    
    df = pd.read_parquet("dataframeRiassunti.parquet")
    df_clean = df[['Group', 'clean_summary']]

    X = df_clean['clean_summary']  # Colonna contenente i riassunti
    y = df_clean['Group']  # Colonna contenente i generi

    # Tokenizzazione dei riassunti
    X_tokenized = X.apply(word_tokenize)

    # Addestramento di Word2Vec sui riassunti tokenizzati
    word2vec_model = Word2Vec(sentences=X_tokenized, vector_size=300, epochs= 10, window=5, min_count=1, workers=4)

    # Funzione per calcolare la media dei vettori Word2Vec delle parole in un riassunto
    def media_vettori(riassunto):
        vettori = [word2vec_model.wv[word] for word in riassunto if word in word2vec_model.wv]
        if len(vettori) > 0:
            return np.mean(vettori, axis=0)
        else:
            return np.zeros(word2vec_model.vector_size)

    # Creazione di un DataFrame con i vettori medi per ogni riassunto
    X_word2vec = np.array([media_vettori(riassunto) for riassunto in X_tokenized])

    # Codifica dei generi (da stringa a intero)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Divisione del dataset in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X_word2vec, y_encoded, test_size=0.2, random_state=42)

    # Addestramento di un modello di classificazione (Random Forest)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Valutazione del modello
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Funzione per predire il genere di un nuovo riassunto
    def predici_genere(riassunto):
        riassunto_tokenized = word_tokenize(riassunto)
        riassunto_vec = media_vettori(riassunto_tokenized).reshape(1, -1) 
        pred_genere = model.predict(riassunto_vec)
        return le.inverse_transform(pred_genere)[0]

    
    genere_predetto = predici_genere(content)
    return genere_predetto, img_base64

    print(f'Il genere predetto è: {genere_predetto}')

    #######################################################################################################












@app.route("/upload", methods=["POST"])  #Questo definisce un route per la web app Flask, accessibile all'URL /upload. Il route accetta richieste HTTP di tipo POST, poiché presumibilmente l'utente invierà un file attraverso un form o una richiesta POST.
def upload_file():
    if "file" not in request.files:    #Controlla se la chiave "file" è presente nella parte files della richiesta
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":   #controlla se il file è vuoto
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)  #viene definito dove salvare il file 
    file.save(file_path)   #viene salvato il file 

    # Apriamo e leggiamo il contenuto del file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()   #content è il file inserito dall'utente in formato stringa
    
    content,img_base64 = prediciGenere(content) #funzione che prende in entrata content e predice il genere 
    #content = genere predetto 
    #img_base64 = grafico delle emozioni 


    return jsonify({"message": "File uploaded successfully", "filename": file.filename, "content": content, "graph": img_base64})

if __name__ == "__main__":   #Questo controllo verifica se lo script è eseguito direttamente. Se è così, 
    app.run(debug=True)                        #l'applicazione Flask viene avviata in modalità debug, utile per lo sviluppo in
                            #quanto fornisce più dettagli sugli errori e ricarica automaticamente il server quando ci sono modifiche al codice.
    
