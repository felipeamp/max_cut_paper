#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''Module used to prepare the shelter animal outcomes database.'''


INPUT_FILEPATH = 'data.csv'
OUTPUT_FILEPATH = 'shelter_animal_outcomes.csv'

NUM_CLASSES = 23

CLASSES_DECREASING_FREQUENCY_ORDER = [
    'Adoption/', 'Transfer/Partner', 'Return_to_owner/', 'Adoption/Foster', 'Transfer/SCRP',
    'Euthanasia/Suffering', 'Euthanasia/Aggressive', 'Adoption/Offsite', 'Died/In Kennel',
    'Euthanasia/Behavior', 'Euthanasia/Rabies Risk', 'Euthanasia/Medical', 'Died/In Foster',
    'Died/', 'Died/Enroute', 'Euthanasia/Court/Investigation', 'Transfer/', 'Died/At Vet',
    'Died/In Surgery', 'Transfer/Barn', 'Euthanasia/', 'Adoption/Barn']

NUM_ATTRIBUTES = 7
AGE_INDEX = 4
BREED_INDEX = 5
TIME_TO_DAYS = {
'1 year': 365,
'2 years': 2 * 365,
'3 weeks': 3 * 7,
'1 month': 1 * 30,
'5 months': 5 * 30,
'4 years': 4 * 365,
'3 months': 3 * 30,
'2 weeks': 2 * 7,
'2 months': 2 * 30,
'10 months': 10 * 30,
'6 months': 6 * 30,
'5 years': 5 * 365,
'7 years': 7 * 365,
'3 years': 3 * 365,
'4 months': 4 * 30,
'12 years': 12 * 365,
'9 years': 9 * 365,
'6 years': 6 * 365,
'1 weeks': 1 * 7,
'11 years': 11 * 365,
'4 weeks': 4 * 7,
'7 months': 7 * 30,
'8 years': 8 * 365,
'11 months': 11 * 30,
'4 days': 4,
'9 months': 9 * 30,
'8 months': 8 * 30,
'15 years': 15 * 365,
'10 years': 10 * 365,
'1 week': 1 * 7,
'0 years': 0 * 365,
'14 years': 14 * 365,
'3 days': 3,
'6 days': 6,
'5 days': 5,
'5 weeks': 5 * 7,
'2 days': 2,
'16 years': 16 * 365,
'1 day': 1,
'13 years': 13 * 365,
'17 years': 17 * 365,
'18 years': 18 * 365,
'19 years': 19 * 365,
'20 years': 20 * 365
}


def header():
    output_list = [
        'AnimalType', 'SexUponOutcome', 'AgeUponOutcome', 'Breed1', 'Breed2', 'Color', 'Outcome']
    return ','.join(output_list)


def prepare_line(line_list, outcomes_to_keep):
    outcome = '/'.join(line_list[:2])
    if outcome not in outcomes_to_keep:
        outcome = 'others'
    breed_separator = line_list[BREED_INDEX].find('/')
    if breed_separator == -1:
        breed1 = line_list[BREED_INDEX]
        breed2 = line_list[BREED_INDEX]
    else:
        breed1 = line_list[BREED_INDEX][:breed_separator]
        breed2 = line_list[BREED_INDEX][breed_separator + 1:]
    output_line_list = (
        line_list[2:4] + [str(TIME_TO_DAYS[line_list[4]]), breed1, breed2, line_list[6], outcome])
    return ','.join(output_line_list)


def main():
    outcomes_to_keep = set(CLASSES_DECREASING_FREQUENCY_ORDER[:NUM_CLASSES - 1])
    with open(OUTPUT_FILEPATH, 'w') as fout:
        with open(INPUT_FILEPATH) as fin:
            is_header = True
            for line in fin:
                if is_header:
                    print(header(), file=fout)
                    is_header = False
                    continue
                line_list = line.rstrip().split(',')
                if not line_list[AGE_INDEX]:
                    continue
                print(prepare_line(line_list, outcomes_to_keep), file=fout)



if __name__ == '__main__':
    main()
