import pandas as pd
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phraser, Phrases
import gensim.corpora as corpora
import spacy, gensim, time, re, sys

# Set as a global variable to load only once. Faster script.
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.max_length = 10683397 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! increase this number if the maximum limit error is raised.
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

def document_tokenizer(list_of_docs):
    '''
    Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long.
    '''

    return [simple_preprocess(text, deacc=True) for text in list_of_docs]

def document_lemmatizer(list_of_docs, allowed_tags=['NOUN', 'VERB', 'ADJ', 'ADV']):
    '''
    Lemmatizing the sentences using spacy library.
    '''
    texts_out = []
    for sent in list_of_docs:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_tags])
    return texts_out

def remove_stopwords(list_of_tokens):
    '''
    Remove stop words from the tokenized text. Input is a list of strings.
    '''
   
    return [[ word for word in tokens if word not in stop_words] for tokens in list_of_tokens]

def make_bigrams(tokens):
    '''
    Making Phrases using gensim.models.phrases.Phrases and Phraser classes
    '''
    bigram = Phrases(tokens, min_count=5, threshold=25)
    bigram_model = Phraser(bigram)
    return [bigram_model[doc] for doc in tokens]

def main():

    DATA_PATH = "input_data\\lda-input.csv" # Path to data (.csv file)
    TEXT_COLUMN = 'text' # Column inside the csv that contains textual data

    start = time.time()
    print("\nReading Data...")
    data = pd.read_csv(DATA_PATH)
    print("Data Loaded. Now standardizing and cleaning textual data.")
    standardize_text(data[TEXT_COLUMN])
    print("Standardization Complete!")
    print("Starting Text Tokenization...")
    tokens = document_tokenizer(data[TEXT_COLUMN])
    print("Tokenization finished successfully!")
    print("Removing stopwords...")
    tokens = remove_stopwords(tokens)
    print("Stop words removed!")
    print("Creating phrases...")
    tokens = make_bigrams(tokens)
    print("Phrases Created!")
    print("Starting to lemmatize the text. !! NOTE: High memory consumption when dealing with large datasets.")
    tokens = document_lemmatizer(tokens)
    print("Lemmatization finished successfully!")
    print("Creating a Dictionary...")
    id2word = corpora.Dictionary(tokens)
    print("Done! Now creating a bag of words for each document / creating corpus...")
    corpus = [id2word.doc2bow(text) for text in tokens]
    print("Text pre-proccesing has finished succesefully!\n")
    print("Now training the Latent Dirichlet Allocation(LDA) model for topic classification!\n")

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                    id2word=id2word,
                                    num_topics=10, 
                                    random_state=100,
                                    update_every=1,
                                    chunksize=100,
                                    passes=10,
                                    alpha='auto',
                                    per_word_topics = True)

        
    print("Training has finished succesefully!\n")
    print("Top 10 topics:\n")

    for topic in lda_model.print_topics():
        print(topic)

    end = time.time()
    print("\nScript working time: ", end-start, "seconds")


if __name__ == "__main__":
    main()