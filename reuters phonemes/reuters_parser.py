#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''Module used to parse reuters texts into a database of phonemes predictions.'''


import collections
import os
import string
from lxml import etree


DICT_FILEPATH = os.path.join('.', 'cmudict.dict')

REUTERS_FILEPATH_BASE = os.path.join('.', 'reut2')
REUTERS_FILEPATH_TYPE = 'sgm'
NUM_REUTERS_FILES = 22

OUTPUT_FILEPATH = os.path.join('.', 'reuters_phonemes.csv')

PHONEME_WINDOW_SIZE = 3


def load_word_to_phonemes(dict_filepath):
    word_to_phonemes = {}
    with open(dict_filepath) as fin:
        for line in fin:
            line_list = line.split('#')[0].rstrip().split()
            word = line_list[0].rstrip('()0123456789')
            if word in word_to_phonemes:
                # Word with more than one pronunciation. Only save the first one.
                continue
            phonemes = []
            for phoneme in line_list[1:]:
                phonemes.append(phoneme.rstrip('012'))
            word_to_phonemes[word] = phonemes
    return word_to_phonemes


def get_list_of_words(raw_text):
    table = str.maketrans({key: ' ' for key in string.punctuation})
    clean_text = raw_text.translate(table)
    return clean_text.split()


def process_text(raw_text, word_to_phonemes, output_file):
    last_phonemes = collections.deque(maxlen=PHONEME_WINDOW_SIZE)
    list_of_words = get_list_of_words(raw_text)
    for word in list_of_words:
        if word in word_to_phonemes:
            curr_phonemes = word_to_phonemes[word]
            save_phonemes(last_phonemes, curr_phonemes, output_file)
        else:
            last_phonemes.clear()


def save_phonemes(last_phonemes, curr_phonemes, output_file):
    for phoneme in curr_phonemes:
        if len(last_phonemes) == PHONEME_WINDOW_SIZE:
            save_output(last_phonemes, phoneme, output_file)
        last_phonemes.append(phoneme)


def save_output(attributes, class_name, output_file):
    print(','.join([*attributes, class_name]), file=output_file)
    output_file.flush()


def get_raw_texts(reuters_filepath):
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(reuters_filepath, parser=parser)
    root = tree.getroot()
    assert len(root) > 1
    for body_tag in root.iter('BODY'):
        yield body_tag.text


def main():
    word_to_phonemes = load_word_to_phonemes(DICT_FILEPATH)
    with open(OUTPUT_FILEPATH, 'w') as output_file:
        reuters_filepaths = [
            '%s-%s.%s' % (REUTERS_FILEPATH_BASE, str(num).zfill(3), REUTERS_FILEPATH_TYPE)
            for num in range(NUM_REUTERS_FILES)]
        for reuters_filepath in reuters_filepaths:
            print('filepath:', reuters_filepath)
            for raw_text in get_raw_texts(reuters_filepath):
                process_text(raw_text, word_to_phonemes, output_file)



if __name__ == '__main__':
    main()
