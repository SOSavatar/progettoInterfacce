
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
#from tensorflow.keras.preprocessing.text import Tokenizer

import nltk
import json
import re
import csv
from tqdm import tqdm
pd.set_option('display.max_colwidth', 300)


#APERTURA BOOKSUMMARY.TXT E CREAZIONE DEL DATABASE
#######################################################################################################

data = []

with open("booksummaries.txt", 'r') as f:
    #apro il file in modalità lettura e lo assegno a f
    reader = csv.reader(f, dialect='excel-tab')
    #viene creato il l'oggetto reader attraverso il modulo csv.Viene specificato il parametro dialect='excel-tab'
    #per indicare che il file è delimitato da tabulazioni (\t), e non da virgole come avviene solitamente nei file CSV.
    for row in tqdm(reader):
        #tqdm serve solo per vedere graficamente l'avanzamento del processo
        data.append(row)


book_index = []
book_id = []
book_author = []
book_name = []
summary = []
genre = []
a = 1
for i in tqdm(data):
    book_index.append(a)
    a = a+1
    book_id.append(i[0])
    book_name.append(i[2])
    book_author.append(i[3])
    genre.append(i[5])
    summary.append(i[6])

df = pd.DataFrame({'Index': book_index, 'ID': book_id, 'BookTitle': book_name, 'Author': book_author,
                    'Genre': genre, 'Summary': summary}).copy()
#######################################################################################################


# PREPROCESSING GENERI 
#######################################################################################################

#qui sto trasformando tutto in minuscolo
df['Genre'] = df['Genre'].str.lower()
#elimino tutto ciò che potrebbe essere html
df['Genre'] = df['Genre'].apply(lambda x: re.sub('(<.*?>)', ' ', x)) 


#modo lungo per eliminare le celle vuote 
index=0
for element in df['Genre']:
    if element=='':
        df=df.drop(index, axis=0)

    index= index+1

#print(df.count())  #questo mi serviva solo per vedere il numero di elementi 

# Step 2: Function to parse JSON and extract genre values
def extract_genres(genre_str):
    # Parse the JSON string and extract the values (genre names)
    genre_dict = json.loads(genre_str)
    return list(genre_dict.values())

# Step 3: Apply the function to the 'genre' column and create a new 'genre_new' column
df['Genre_new'] = df['Genre'].apply(extract_genres)

# Optionally, inspect the DataFrame to ensure everything is correct
#print(df[['Genre', 'Genre_new']].head())
##############################################################################################
print(df['Genre_new'])




#creazione dei gruppi (sovragenere)
#######################################################################################################
import pandas as pd
from collections import Counter

# Creiamo il dizionario dei generi raggruppati
genre_groups = {
    'Fiction letteraria e realismo': ['fiction', 'bildungsroman', 'roman à clef', 'novel', 'literary fiction', 'psychological novel', 'autobiographical novel', 'literary realism', 'existentialism', 'memoir'],
    'Fantasy e Fantascienza': ['fantasy', 'high fantasy', 'urban fantasy', 'sword and sorcery', 'epic science fiction and fantasy', 'science fiction', 'cyberpunk', 'postcyberpunk', 'soft science fiction', 'hard science fiction', 'space opera', 'dying earth subgenre', 'new weird', 'alternate history', 'time travel', 'steampunk', 'utopian and dystopian fiction', 'speculative fiction'],
    'Romance e Amore': ['romance novel', 'historical romance', 'regency romance', 'paranormal romance', 'chick lit', 'georgian romance', 'elizabethan romance', 'medieval romance', 'chivalric romance', 'indian chick lit'],
    'Giallo, Thriller e Mistero': ['detective fiction', 'thriller', 'mystery', 'police procedural', 'locked room mystery', 'whodunit', 'crime fiction', 'suspense', 'noir', 'spy fiction'],
    'Horror e Sopranaturale': ['gothic fiction', 'supernatural', 'horror', 'vampire fiction', 'zombie', 'apocalyptic and post-apocalyptic fiction', 'ghost story', 'alien invasion', 'human extinction', 'paranormal romance'],
    'Commedia e Satira': ['comedy', 'black comedy', 'comic novel', 'satire', 'comic science fiction', 'comic fantasy', 'parody', 'farce', 'romantic comedy'],
    'Avventura e Azione': ['adventure', 'adventure novel', 'naval adventure', 'sea story', 'robinsonade', 'lost world', 'western', 'space western'],
    'Saggi, non-fiction e testi tecnici': ['non-fiction', 'popular science', 'autobiography', 'biography', 'essay', 'history', 'popular culture', 'encyclopedia', 'travel literature', 'social sciences', 'military history', 'reference', 'self-help', 'cookbook'],
    'Letteratura sperimentale e innovativa': ['experimental literature', 'ergodic literature', 'collage', 'new weird', 'mashup', 'pastiche', 'prose poetry', 'epistolary novel'],
    'Generi tematici e di critica sociale': ['anti-war', 'polemic', 'social commentary', 'political philosophy', 'social criticism', 'sociology', 'feminist science fiction', 'transhumanism', 'utopian fiction', 'dystopia', 'conspiracy fiction', 'anti-nuclear']
}

# Creiamo una funzione per trovare il sovragruppo dominante
def find_genre_group(genres):
    group_counts = Counter()   # è un dizionario in cui le chiavi sono i sovragruppi e i valori sono i contatori
    for genre in genres:
        for group, group_genres in genre_groups.items():
            if genre in group_genres:   #qui ho cambiato una cosa (il lower())
                group_counts[group] += 1
    if group_counts:
        return group_counts.most_common(1)[0][0]  # Restituisce il sovragruppo con il maggior numero di generi
    return 'Non classificato'



# Applichiamo la funzione al DataFrame
df['Group'] = df['Genre_new'].apply(find_genre_group)
#######################################################################################################








#selezione dei 3 sovragruppi principali 
#######################################################################################################
group_counts = df['Group'].value_counts() #crea una series in cui per ogni gruppo ho il numero di elementi associati a quel gruppo

valid_groups = group_counts[group_counts >= 2000].index  #L'operazione group_counts >= 2000 crea una Serie booleana che indica quali gruppi soddisfano la condizione.

# Filtra il DataFrame per mantenere solo i gruppi validi
df = df[df['Group'].isin(valid_groups)]  # Mantieni solo i gruppi con almeno 100 righe

# Ora per ogni Group, limitiamo a 3000 righe
df = df.groupby('Group').head(3000)
#######################################################################################################




# PREPROCESSING RIASSUNTI 
#######################################################################################################


#vado a fare la pulizia di tutti i riassunti 
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
#qui ho creato un set(stop_words) con tutte le parole inglesi più utilizzate(articoli,preposizioni etc...). Mi servirà in seguito 
#per pulire i testi 

def clean_summary(text):
    # Rimuovi punteggiatura e caratteri speciali
    text = re.sub(r'\W', ' ', text) #\W include qualsiasi tipo di carettere speciale 
    # Converti in minuscolo
    text = text.lower()
    # Rimuovi stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    #qui vengono eliminate tutte le stop_words
    return text

# Applica la funzione di pulizia al campo Summary
df['clean_summary'] = df['Summary'].apply(clean_summary)
#qui viene applicato il metodo clean_summary ad ogni riga della colonna Summary e il risultato viene portato in una nuova colonna 
#chiamata 'clean_summary'

#print(df['clean_summary'][1])
#######################################################################################################

df.to_parquet("dataframeRiassunti.parquet")
