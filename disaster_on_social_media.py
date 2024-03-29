from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import re

def standardize_text(text_values):
    '''
    Standardizing list of text(strings) for creating a more accurate NLP model. It can also be used just to standardize text.
    The input argument is a list of strings. e.g. ['Dummy text 1','dummy text 2']
    Return: "list of cleaned text values" 
    '''
    
    for index in range(len(text_values)):
        text_values[index] = re.sub(r'http\S+', "", text_values[index])
        text_values[index] = re.sub(r'@\S+', "", text_values[index])
        text_values[index] = re.sub(r'[^A-Za-z0-9(),!?@\'\`\"\_\n]', ' ', text_values[index])
        text_values[index] = re.sub(r'@', 'at', text_values[index])
        text_values[index] = text_values[index].lower().strip()

    return text_values

def regex_tokenizer(text_values, regex):
    '''
    A function that returns a new list of strings based on RegexpTokenizer transformation from NLTK library.
    Input: regex_tokenizer("list of strings","regular expression")
    Return: "list of tokenized sentences"
    '''

    tokenizer = RegexpTokenizer(regex)
    return list(map(tokenizer.tokenize, text_values))

def inspect_text_data(text_values, graph_flag = False):
    '''
    The function inspects textual data. Gives information about a number of words, number of unique words and sentence length (max length).
    Input: inspect_text_data("list of string tokens", "graph_flag") -> graph_flag default state is False, if True, Histogram is displayed showing sentence length information.
    '''

    all_words = [ word for token in text_values for word in token]
    sentence_length = [len(token) for token in text_values]
    vocabulary = list(set(all_words))

    print("%s words in total, with a vocabulary size of %s" % (len(all_words), len(vocabulary)))
    print("Max sentence length is %s" % max(sentence_length))

    if graph_flag:
        fig = plt.figure(figsize=(5, 5)) 
        plt.xlabel('Sentence length')
        plt.ylabel('Number of sentences')
        plt.hist(sentence_length)
        plt.show()

def count_vectorizer(text_values):
    '''
    Creating "Bag of Words" using CountVectorizer() which converts a collection of text documents to a matrix of token counts
    Input: count_vectorizer("list of strings")
    Return: "Bag of words","CountVectorizer"
    '''
    cv = CountVectorizer()
    bow = cv.fit_transform(text_values)

    return bow, cv
    
def main():

    source_path = '..\\input_data\\socialmedia-disaster-tweets-DFE.csv' # https://www.kaggle.com/jannesklaas/disasters-on-social-media
    
    data = pd.read_csv(source_path, encoding='latin')
    working_data = data.loc[:,['text', 'choose_one', 'class_label']]

    working_data['text'] = standardize_text(working_data['text'].tolist())
    working_data['clean_text_tokens'] = regex_tokenizer(working_data['text'].tolist(), r'\w+')

    #inspect_text_data(working_data['clean_text_tokens'].values, True)

    text_data = working_data['text'].tolist()
    label_data = working_data['class_label'].tolist()

    #Creating train and test data for the ML model
    x_train, x_test, y_train, y_test = train_test_split(text_data, label_data, test_size=0.2, random_state=42)




    count_vectorizer(working_data['text'].tolist())

if __name__ == "__main__":
    main()