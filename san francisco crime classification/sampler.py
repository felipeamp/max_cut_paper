#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''Module used to get random samples from the San Francisco Crime Classification.'''

import random

CLASSES_DECREASING_FREQUENCY_ORDER = [
    'LARCENY/THEFT', 'OTHER OFFENSES', 'NON-CRIMINAL', 'ASSAULT', 'DRUG/NARCOTIC', 'VEHICLE THEFT',
    'VANDALISM', 'WARRANTS', 'BURGLARY', 'SUSPICIOUS OCC', 'MISSING PERSON', 'ROBBERY', 'FRAUD',
    'FORGERY/COUNTERFEITING', 'SECONDARY CODES', 'WEAPON LAWS', 'PROSTITUTION', 'TRESPASS',
    'STOLEN PROPERTY', 'SEX OFFENSES FORCIBLE', 'DISORDERLY CONDUCT', 'DRUNKENNESS',
    'RECOVERED VEHICLE', 'KIDNAPPING', 'DRIVING UNDER THE INFLUENCE', 'RUNAWAY', 'LIQUOR LAWS',
    'ARSON', 'LOITERING', 'EMBEZZLEMENT', 'SUICIDE', 'FAMILY OFFENSES', 'BAD CHECKS', 'BRIBERY',
    'EXTORTION', 'SEX OFFENSES NON FORCIBLE', 'GAMBLING', 'PORNOGRAPHY/OBSCENE MAT', 'TREA']
MIN_NUM_CLASSES = 15
MAX_NUM_CLASSES = 15
RANDOM_SEED = 19880531
FULL_DATASET_FILEPATH = 'data.csv'


def generate_dataset(num_classes):
    random.seed(RANDOM_SEED)
    classes_to_keep = set(CLASSES_DECREASING_FREQUENCY_ORDER[:num_classes - 1])
    output_filepath = 'san_francisco_crime_all_samples_%s_classes.csv' % num_classes
    with open(FULL_DATASET_FILEPATH) as fin:
        with open(output_filepath, 'w') as fout:
            is_header = True
            for line_num, line in enumerate(fin):
                last_comma_index = line.rindex(',')
                if is_header:
                    print(line, end='', file=fout)
                    is_header = False
                # elif line_num in samples_indices_to_write:
                elif line[last_comma_index + 1:].rstrip() in classes_to_keep:
                    print(line, end='', file=fout)
                else:
                    output_line = line[:last_comma_index + 1] + 'others'
                    print(output_line, file=fout)

def main():
    for num_classes in range(MIN_NUM_CLASSES, MAX_NUM_CLASSES + 1):
        print('num_classes:', num_classes)
        generate_dataset(num_classes)


if __name__ == '__main__':
    main()
