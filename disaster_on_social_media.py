import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer

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

def main():

    source_path = 'input_data\\socialmedia-disaster-tweets-DFE.csv'
    
    data = pd.read_csv(source_path, encoding='latin')
    working_data = data[['text', 'choose_one']]
    working_data['text'] = standardize_text(working_data['text'].values)

    print(working_data.head())

if __name__ == "__main__":
    main()