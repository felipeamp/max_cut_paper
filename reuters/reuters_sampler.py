#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''Module used to group rare phonemes in reuters phonemes dataset and save only a portion of them.
'''

import random

MIN_NUM_CLASSES = 10
MAX_NUM_CLASSES = 15
NUM_ATTRIBUTES = 3
NUM_OUTPUT_SAMPLES = 20000
PHONEMES_DECREASING_FREQUENCY_ORDER = [
    'AH', 'T', 'N', 'IH', 'S', 'R', 'D', 'L', 'K', 'Z', 'EH', 'IY', 'ER', 'M', 'P', 'DH', 'AE', 'V',
    'UW', 'F', 'AA', 'EY', 'B', 'AO', 'W', 'AY', 'OW', 'NG', 'SH', 'Y', 'G', 'HH', 'JH', 'CH', 'AW',
    'UH', 'TH', 'OY', 'ZH']
RANDOM_SEED = 19880531
REUTERS_FILEPATH = 'reuters_phonemes.csv'


def write_output_header(fout):
    header = ['a_%s' % attrib_num for attrib_num in range(NUM_ATTRIBUTES)]
    header.append('class')
    print(','.join(header), file=fout)


def generate_dataset(num_classes):
    random.seed(RANDOM_SEED)
    phonemes_to_keep = set(PHONEMES_DECREASING_FREQUENCY_ORDER[:num_classes])
    output_filepath = 'reuters_phonemes_%s_samples_%s_classes.csv' % (NUM_OUTPUT_SAMPLES,
                                                                      num_classes)
    ok_samples_indices = []
    with open(REUTERS_FILEPATH) as fin:
        for line_num, line in enumerate(fin):
            if line[line.rindex(',') + 1:].rstrip() in phonemes_to_keep:
                ok_samples_indices.append(line_num)
        assert NUM_OUTPUT_SAMPLES <= len(ok_samples_indices)
        samples_indices_to_write = set(random.sample(ok_samples_indices, NUM_OUTPUT_SAMPLES))
        fin.seek(0)
        with open(output_filepath, 'w') as fout:
            write_output_header(fout)
            for line_num, line in enumerate(fin):
                if line_num in samples_indices_to_write:
                    print(line, end='', file=fout)

def main():
    for num_classes in range(MIN_NUM_CLASSES, MAX_NUM_CLASSES + 1):
        print('num_classes:', num_classes)
        generate_dataset(num_classes)


if __name__ == '__main__':
    main()
