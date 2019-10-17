from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd
import re


def standardize_text(text_values):
    '''
    Standardizing list of text(strings) for creating a more accurate NLP model. It can also be used just to standardize text.
    The input argument is a list of strings. e.g. ['Dummy text 1','dummy text 2']
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
    '''

    tokenizer = RegexpTokenizer(regex)
    return list(map(tokenizer.tokenize, text_values))

def main():

    source_path = '..\\input_data\\socialmedia-disaster-tweets-DFE.csv' # https://www.kaggle.com/jannesklaas/disasters-on-social-media
    
    data = pd.read_csv(source_path, encoding='latin')
    working_data = data.loc[:,['text', 'choose_one']]

    working_data['text'] = standardize_text(working_data['text'].values)
    working_data['clean_text_tokens'] = regex_tokenizer(working_data['text'].values, r'\w+')

    print(working_data.tail())

if __name__ == "__main__":
    main()