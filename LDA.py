import pandas as pd
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phraser, Phrases
import gensim.corpora as corpora
import spacy, gensim, time, re

# Set as a global variable to load only once. Faster script.
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
stop_words = stopwords.words('english')

def standardize_text(text_values):
    '''
    Standardizing list of text(strings) for creating a more accurate NLP model. It can also be used just to standardize text.
    The input argument is a list of strings. e.g. ['Dummy text 1','dummy text 2'] or DataFrame Series
    '''
    
    for index in range(len(text_values)):
        text_values[index] = re.sub(r'http\S+', "", text_values[index])
        text_values[index] = re.sub(r'@\S+', "", text_values[index])
        text_values[index] = re.sub(r'[^A-Za-z0-9(),!?@\'\`\"\_\n]', ' ', text_values[index])
        text_values[index] = re.sub(r'@', 'at', text_values[index])
        text_values[index] = text_values[index].lower().strip()
        text_values[index] = text_values[index].replace('"','').replace("'",'')
        text_values[index] = ' '.join(text_values[index].split())

def sentence_tokenizer(text):
    '''
    Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long.
    '''

    return simple_preprocess(text, deacc=True)

def sentence_lemmatizer(tokenized_sentence, allowed_tags=['NOUN', 'VERB']):
    '''
    Lemmatizing the sentences using spacy library.
    '''

    doc = nlp(' '.join(tokenized_sentence))
    return [token.lemma_ for token in doc if token.pos_ in allowed_tags]

def remove_stopwords(tokenized_sentence):
    '''
    Remove stop words from the tokenized text. Input is a list of strings.
    '''

    return [word for word in tokenized_sentence if word not in stop_words]

def make_bigrams(list_of_docs):
    '''
    Making Phrases using gensim.models.phrases.Phrases and Phraser classes
    '''
    bigram = Phrases([remove_stopwords(sentence_tokenizer(text)) for text in list_of_docs], min_count=5, threshold=25)
    bigram_model = Phraser(bigram)
    return bigram_model

def main():

    DATA_PATH = "input_data\\lda-input.csv" # Path to data (.csv file)
    TEXT_COLUMN = 'text' # Column inside the csv that contains textual data

    data = pd.read_csv(DATA_PATH)
    standardize_text(data[TEXT_COLUMN])

    bigrams = make_bigrams(data[TEXT_COLUMN]) # high time consumption ~8-10 seconds
 
    start = time.time()

    for i, text in enumerate(data[TEXT_COLUMN]):
    
        tokens = sentence_tokenizer(text)
        bigram = bigrams[tokens]
        lemmatized_text = sentence_lemmatizer(bigram) # pass additional allowed_tags variable if you want to include more parts of speech. By default Nouns and Verbs are included.

        id2word = corpora.Dictionary([lemmatized_text])
        corpus = id2word.doc2bow(lemmatized_text)
       
        try:
            lda_model = gensim.models.ldamodel.LdaModel(corpus=[corpus],
                                            id2word=id2word,
                                            num_topics=1, 
                                            random_state=25,
                                            update_every=1,
                                            #chunksize=100,
                                            passes=20,
                                            alpha='auto')
                                            #per_word_topics = True)
        except ValueError:
            print("ValueError: There is no enough textual data to perform LDA topic classification. Row ", i, " caused an error")
            

        print(i, lda_model.print_topics())

    end = time.time()
    print(end-start)


if __name__ == "__main__":
    main()