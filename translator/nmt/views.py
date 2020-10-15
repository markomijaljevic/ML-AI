from django.shortcuts import render
import keras
import pickle
import numpy as np
import json
from pathlib import Path
from django.http import HttpResponse
from nltk.translate.bleu_score import sentence_bleu

BASE_DIR = Path(__file__).resolve().parent
MODEL = {0 : 'auto', 
        'HR_EN' : 'hr_en_trained', 'EN_HR' : 'en_hr_trained', 'HR_IT' : 'hr_it_trained', 
        'IT_HR' : 'it_hr_trained', 'HR_DE' : 'hr_de_trained', 'DE_HR' : 'de_hr_trained', 
        'HR_FR' : 'hr_fr_trained', 'FR_HR' : 'fr_hr_trained', 'HR_ES' : 'hr_es_trained', 
        'ES_HR' : 'es_hr_trained'}

# Create your views here.

def get_landing_page(request):
    return render(request, 'nmt/landing_page.html')


def translate_sentence(input_seq, lang_id):

    encoder_model = keras.models.load_model(str(BASE_DIR) + "\\LSTM_models\\" + MODEL[lang_id]   + "\\encoder_model")
    decoder_model = keras.models.load_model(str(BASE_DIR) + "\\LSTM_models\\" + MODEL[lang_id]   + "\\decoder_model")

    file = open(str(BASE_DIR) + '\\LSTM_models\\' + MODEL[lang_id]   + '\\word2idx_outputs', 'rb')
    word2idx_outputs = pickle.load(file)
    file.close()

    file = open(str(BASE_DIR) + '\\LSTM_models\\' + MODEL[lang_id]   + '\\idx2word_target', 'rb')
    idx2word_target = pickle.load(file)
    file.close()

    file = open(str(BASE_DIR) + '\\LSTM_models\\' + MODEL[lang_id]   + '\\max_out_len', 'rb')
    max_out_len = pickle.load(file)
    file.close()

    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']
    output_sentence = []

    for _ in range(max_out_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)

def translate(request, sentence, lang_id):

    reference = None

    if '|' in sentence:
        sentence, reference = [sent.strip() for sent in sentence.split('|')[:2]]
 
    file = open(str(BASE_DIR) + '\\LSTM_models\\' + MODEL[lang_id]   + '\\max_input_len', 'rb')
    max_input_len = pickle.load(file)
    file.close()

    file = open(str(BASE_DIR) + '\\LSTM_models\\' + MODEL[lang_id]   + '\\word2idx_inputs', 'rb')
    word2idx_inputs = pickle.load(file)
    file.close()

    tokens = [token.lower() for token in sentence.split()]

    tokens_to_integers_list = []

    for token in tokens:
        if token in word2idx_inputs.keys():
            tokens_to_integers_list.append(word2idx_inputs[token])

    length = len(tokens_to_integers_list)
    input_seq = [[0 for _ in range(0, max_input_len - length)] + tokens_to_integers_list]
    translated_sentence = translate_sentence(input_seq, lang_id)

    bleu_score = 0
    if reference:
        bleu_score = sentence_bleu([reference.lower().split()], translated_sentence.split())

    return HttpResponse(json.dumps({'trans':translated_sentence, 'bleu':bleu_score}))

