#!/usr/bin/python3
# -*- coding: utf-8 -*-


import collections
import csv
import itertools
import math
# import os
import statistics

from scipy.stats import t

NUM_FOLDS = 10


def rec_dd():
    return collections.defaultdict(rec_dd)


def load_test_results(csv_filepath):
    csv_information = rec_dd()
    print('Starting to load CSV')
    with open(csv_filepath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        has_read_header = not HAS_HEADER
        header_start = None
        for row in reader:
            if not has_read_header:
                has_read_header = True
                header_start = row[0]
                continue
            if HAS_HEADER and MUST_CLEAN_HEADER_REPETITION and row[0] == header_start:
                continue

            if HEIGHT_COLUMN_INDEX is not None and ATTRIBUTE_COLUMN_INDEX is not None:
                # Rank height >= 2
                csv_information[row[DATASET_COLUMN_INDEX]][row[ATTRIBUTE_COLUMN_INDEX]][
                    row[HEIGHT_COLUMN_INDEX]][row[CRITERION_NAME_COLUMN_INDEX]][
                        row[SEED_INDEX_COLUMN_INDEX]][row[FOLD_NUMBER_COLUMN_INDEX]] = (
                            float(row[ACCURACY_W_MISSING_COLUMN_INDEX]),
                            int(row[NUM_VALIDATION_SAMPLES_W_MISSING_COLUMN_INDEX]),
                            float(row[ACCURACY_WO_MISSING_COLUMN_INDEX]),
                            int(row[NUM_VALIDATION_SAMPLES_WO_MISSING_COLUMN_INDEX]))
            elif HEIGHT_COLUMN_INDEX is not None and FOLD_NUMBER_COLUMN_INDEX is None:
                # Decision Tree
                csv_information[row[DATASET_COLUMN_INDEX]]['-'][row[HEIGHT_COLUMN_INDEX]][
                    row[CRITERION_NAME_COLUMN_INDEX]][row[SEED_INDEX_COLUMN_INDEX]]['-'] = (
                        float(row[ACCURACY_W_MISSING_COLUMN_INDEX]),
                        int(row[NUM_VALIDATION_SAMPLES_W_MISSING_COLUMN_INDEX]),
                        float(row[ACCURACY_WO_MISSING_COLUMN_INDEX]),
                        int(row[NUM_VALIDATION_SAMPLES_WO_MISSING_COLUMN_INDEX]))
            elif ATTRIBUTE_COLUMN_INDEX is not None:
                # Rank height = 1
                csv_information[row[DATASET_COLUMN_INDEX]][row[ATTRIBUTE_COLUMN_INDEX]]['-'][
                    row[CRITERION_NAME_COLUMN_INDEX]][row[SEED_INDEX_COLUMN_INDEX]][
                        row[FOLD_NUMBER_COLUMN_INDEX]] = (
                            float(row[ACCURACY_W_MISSING_COLUMN_INDEX]),
                            int(row[NUM_VALIDATION_SAMPLES_W_MISSING_COLUMN_INDEX]),
                            float(row[ACCURACY_WO_MISSING_COLUMN_INDEX]),
                            int(row[NUM_VALIDATION_SAMPLES_WO_MISSING_COLUMN_INDEX]))
            else:
                print('Unkown experiment type (check attribute, height and fold column indices.')
                exit(1)
    print('Done')
    print('Adjusting csv structural format')
    for dataset in csv_information:
        for attribute_name in csv_information[dataset]:
            for height in csv_information[dataset][attribute_name]:
                for criterion_name in csv_information[dataset][attribute_name][height]:
                    num_seeds = len(
                        csv_information[dataset][attribute_name][height][criterion_name])
                    list_of_folds_info_per_seed = [[] for _ in range(num_seeds)]
                    for seed_index in csv_information[dataset][
                            attribute_name][height][criterion_name]:
                        if FOLD_NUMBER_COLUMN_INDEX is not None:
                            folds_info = [[] for _ in range(NUM_FOLDS)]
                            for fold_number, fold_info in csv_information[dataset][
                                    attribute_name][height][criterion_name][seed_index].items():
                                folds_info[int(fold_number) - 1] = fold_info
                        else:
                            folds_info = [csv_information[dataset][
                                attribute_name][height][criterion_name][seed_index]['-']]
                        list_of_folds_info_per_seed[int(seed_index) - 1] = folds_info
                    csv_information[dataset][attribute_name][height][
                        criterion_name] = list_of_folds_info_per_seed
    print('Done')
    return csv_information


def aggregate_fold_info(csv_information):
    print('Aggregating folds information')
    aggregated_info = {}
    for dataset in csv_information:
        aggregated_info[dataset] = {}
        for attribute_name in csv_information[dataset]:
            aggregated_info[dataset][attribute_name] = {}
            for height in csv_information[dataset][attribute_name]:
                aggregated_info[dataset][attribute_name][height] = {}
                for criterion in csv_information[dataset][attribute_name][height]:
                    is_ok = True
                    aggregated_info_per_seed = []
                    for (seed_index,
                         folds_info) in enumerate(csv_information[dataset][attribute_name][height][
                             criterion]):
                        try:
                            (accuracy_w_missing_values,
                             list_of_folds_validation_set_size_w_missing,
                             accuracy_wo_missing_values,
                             list_of_folds_validation_set_size_wo_missing) = zip(*folds_info)
                        except ValueError:
                            if is_ok:
                                is_ok = False
                            missing_folds = []
                            for fold_number, fold_info in enumerate(folds_info):
                                if len(fold_info) == 0:
                                    missing_folds.append(fold_number + 1)
                            print('Experiment for dataset {}, attribute {}, height {}, seed {} and'
                                  ' fold(s) {} is missing.'.format(
                                      dataset, attribute_name, height, seed_index + 1,
                                      *missing_folds))
                            continue
                        total_accuracy_w_missing_values = calculate_total_accuracy(
                            accuracy_w_missing_values,
                            list_of_folds_validation_set_size_w_missing)
                        total_accuracy_wo_missing_values = calculate_total_accuracy(
                            accuracy_wo_missing_values,
                            list_of_folds_validation_set_size_wo_missing)
                        aggregated_info_per_seed.append((total_accuracy_w_missing_values,
                                                         total_accuracy_wo_missing_values))
                    if is_ok:
                        aggregated_info[dataset][attribute_name][height][
                            criterion] = aggregated_info_per_seed
                    else:
                        print('Skipping this configuration dataset/attribute/height/criterion.')
    print('Done')
    return aggregated_info


def calculate_total_accuracy(list_of_folds_accuracies, list_of_folds_validation_set_size):
    total_correct = 0
    total_num_samples = 0
    for curr_accuracy, curr_size in zip(list_of_folds_accuracies,
                                        list_of_folds_validation_set_size):
        total_num_samples += curr_size
        total_correct += round((curr_accuracy / 100.0) * curr_size)
    return 100.0 * total_correct / total_num_samples


def calculate_diff(first_samples, second_samples):
    return [first_sample - second_sample
            for (first_sample, second_sample) in zip(first_samples, second_samples)]


def calculate_t_statistic(samples_list):
    mean = statistics.mean(samples_list)
    variance = statistics.variance(samples_list)
    if variance == 0.0:
        return 0.0, 100.0
    num_samples = len(samples_list)
    t_statistic = mean / math.sqrt(variance / num_samples)
    degrees_of_freedom = num_samples - 1
    cdf = t.cdf(t_statistic, degrees_of_freedom)
    two_tailed_p_value = 2. * min(cdf, 1. - cdf)
    return t_statistic, two_tailed_p_value


def calculate_original_statistics(samples_list):
    mean = statistics.mean(samples_list)
    variance = statistics.variance(samples_list)
    return (mean, math.sqrt(variance))


def save_comparison_booleans(dataset, attribute_name, height, criterion_1, criterion_2,
                             t_stats_w_missing, two_tailed_p_value_w_missing, t_stats_wo_missing,
                             two_tailed_p_value_wo_missing, boolean_information):
    for two_sided_p_value_cutoff in TWO_TAILED_P_VALUE_CUTOFFS:
        if two_sided_p_value_cutoff >= two_tailed_p_value_w_missing:
            if t_stats_w_missing < 0.0:
                # second criterion is statistically better
                boolean_information[dataset][attribute_name][height][criterion_2][
                    two_sided_p_value_cutoff][0] += 1
            else:
                # first criterion is statistically better
                boolean_information[dataset][attribute_name][height][criterion_1][
                    two_sided_p_value_cutoff][0] += 1

        if two_sided_p_value_cutoff >= two_tailed_p_value_wo_missing:
            if t_stats_wo_missing < 0.0:
                # second criterion is statistically better
                boolean_information[dataset][attribute_name][height][criterion_2][
                    two_sided_p_value_cutoff][1] += 1
            else:
                # first criterion is statistically better
                boolean_information[dataset][attribute_name][height][criterion_1][
                    two_sided_p_value_cutoff][1] += 1


def write_stats(t_test_infos, statistics_csv_filepath):
    print('Saving statistics')
    with open(statistics_csv_filepath, 'a') as fout:
        print(','.join(['Dataset', 'Attribute Name', 'Height', 'Criteria Diff Name',
                        'paired t-statistic with missing values',
                        'paired two-tailed p-value with missing values',
                        'paired t-statistic without missing values',
                        'paired two-tailed p-value without missing values']), file=fout)
        for dataset in t_test_infos:
            for attribute_name in t_test_infos[dataset]:
                for height in t_test_infos[dataset][attribute_name]:
                    for (criteria_diff_name,
                         (t_stats_w_missing,
                          two_tailed_p_value_w_missing,
                          t_stats_wo_missing,
                          two_tailed_p_value_wo_missing)) in t_test_infos[
                              dataset][attribute_name][height].items():
                        line_list = [dataset,
                                     attribute_name,
                                     height,
                                     criteria_diff_name,
                                     str(t_stats_w_missing),
                                     str(two_tailed_p_value_w_missing),
                                     str(t_stats_wo_missing),
                                     str(two_tailed_p_value_wo_missing)]
                        print(','.join(line_list), file=fout)
    print('Done')


def write_aggreg_stats(boolean_information, original_statistics,
                       aggregated_statistics_csv_filepath_1, aggregated_statistics_csv_filepath_2,
                       aggregated_statistics_csv_filepath_3, aggregated_statistics_csv_filepath_4):
    def get_auxs(boolean_information):
        aux_without_attributes = collections.defaultdict(lambda: collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))))
        aux_without_attributes_and_heights = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0])))
        aux_without_datasets_and_attributes_and_heights = collections.defaultdict(
            lambda: collections.defaultdict(lambda: [0, 0]))

        for dataset in boolean_information:
            for attribute_name in boolean_information[dataset]:
                for height in boolean_information[dataset][attribute_name]:
                    for criterion in boolean_information[dataset][attribute_name][height]:
                        for p_value in boolean_information[dataset][attribute_name][height][
                                criterion]:
                            aux_without_attributes[dataset][height][criterion][p_value][0] += (
                                boolean_information[dataset][attribute_name][height][criterion][
                                    p_value][0])
                            aux_without_attributes[dataset][height][criterion][p_value][1] += (
                                boolean_information[dataset][attribute_name][height][criterion][
                                    p_value][1])

                            aux_without_attributes_and_heights[dataset][criterion][p_value][0] += (
                                boolean_information[dataset][attribute_name][height][criterion][
                                    p_value][0])
                            aux_without_attributes_and_heights[dataset][criterion][p_value][1] += (
                                boolean_information[dataset][attribute_name][height][criterion][
                                    p_value][1])

                            aux_without_datasets_and_attributes_and_heights[criterion][
                                p_value][0] += (boolean_information[dataset][attribute_name][
                                    height][criterion][p_value][0])
                            aux_without_datasets_and_attributes_and_heights[criterion][
                                p_value][1] += (boolean_information[dataset][attribute_name][
                                    height][criterion][p_value][1])
        return (aux_without_attributes,
                aux_without_attributes_and_heights,
                aux_without_datasets_and_attributes_and_heights)


    print('Saving aggregated statistics')
    with open(aggregated_statistics_csv_filepath_1, 'a') as fout:
        print(','.join(['Dataset', 'Attribute Name', 'Height', 'Criterion',
                        'Average with missing values', 'Standard Deviation with missing values',
                        'Average without missing values',
                        'Standard Deviation without missing values', 'Two-sided p-value cutoff',
                        'Number of times is statistically better with missing values',
                        'Is always better',
                        'Number of times is statistically better without missing values',
                        'Is always better']),
              file=fout)
        for dataset in boolean_information:
            for attribute_name in boolean_information[dataset]:
                for height in boolean_information[dataset][attribute_name]:
                    num_criteria = len(boolean_information[dataset][attribute_name][height])
                    for criterion in boolean_information[dataset][attribute_name][height]:
                        average_with_missing, std_dev_with_missing = original_statistics[
                            dataset][attribute_name][height][criterion][0]
                        average_without_missing, std_dev_without_missing = original_statistics[
                            dataset][attribute_name][height][criterion][1]
                        for p_value in boolean_information[dataset][attribute_name][height][
                                criterion]:
                            num_times_is_better_w_missing = boolean_information[dataset][
                                attribute_name][height][criterion][p_value][0]
                            is_always_better_w_missing = (
                                num_times_is_better_w_missing == num_criteria - 1)
                            num_times_is_better_wo_missing = boolean_information[dataset][
                                attribute_name][height][criterion][p_value][1]
                            is_always_better_wo_missing = (
                                num_times_is_better_wo_missing == num_criteria - 1)
                            print(','.join([dataset, attribute_name, height, criterion,
                                            str(average_with_missing),
                                            str(std_dev_with_missing),
                                            str(average_without_missing),
                                            str(std_dev_without_missing),
                                            str(p_value),
                                            str(num_times_is_better_w_missing),
                                            str(is_always_better_w_missing),
                                            str(num_times_is_better_wo_missing),
                                            str(is_always_better_wo_missing)]),
                                  file=fout)

        (aux_without_attributes,
         aux_without_attributes_and_heights,
         aux_without_datasets_and_attributes_and_heights) = get_auxs(boolean_information)

    with open(aggregated_statistics_csv_filepath_2, 'a') as fout:
        print(','.join(['Dataset', 'Height', 'Criterion', 'Two-sided p-value cutoff',
                        'Number of times is statistically better with missing values',
                        'Is always better',
                        'Number of times is statistically better without missing values',
                        'Is always better']),
              file=fout)
        for dataset in aux_without_attributes:
            for height in aux_without_attributes[dataset]:
                num_criteria = len(aux_without_attributes[dataset][height])
                for criterion in aux_without_attributes[dataset][height]:
                    for p_value in aux_without_attributes[dataset][height][
                            criterion]:
                        num_times_is_better_w_missing = aux_without_attributes[dataset][height][
                            criterion][p_value][0]
                        is_always_better_w_missing = (
                            num_times_is_better_w_missing == num_criteria - 1)
                        num_times_is_better_wo_missing = aux_without_attributes[dataset][height][
                            criterion][p_value][1]
                        is_always_better_wo_missing = (
                            num_times_is_better_wo_missing == num_criteria - 1)
                        print(','.join([dataset, height, criterion,
                                        str(p_value),
                                        str(num_times_is_better_w_missing),
                                        str(is_always_better_w_missing),
                                        str(num_times_is_better_wo_missing),
                                        str(is_always_better_wo_missing)]),
                              file=fout)

    with open(aggregated_statistics_csv_filepath_3, 'a') as fout:
        print(','.join(['Dataset', 'Criterion', 'Two-sided p-value cutoff',
                        'Number of times is statistically better with missing values',
                        'Is always better',
                        'Number of times is statistically better without missing values',
                        'Is always better']),
              file=fout)
        for dataset in aux_without_attributes_and_heights:
            num_criteria = len(aux_without_attributes_and_heights[dataset])
            for criterion in aux_without_attributes_and_heights[dataset]:
                for p_value in aux_without_attributes_and_heights[dataset][criterion]:
                    num_times_is_better_w_missing = aux_without_attributes_and_heights[dataset][
                        criterion][p_value][0]
                    is_always_better_w_missing = (
                        num_times_is_better_w_missing == num_criteria - 1)
                    num_times_is_better_wo_missing = aux_without_attributes_and_heights[dataset][
                        criterion][p_value][1]
                    is_always_better_wo_missing = (
                        num_times_is_better_wo_missing == num_criteria - 1)
                    print(','.join([dataset, criterion,
                                    str(p_value),
                                    str(num_times_is_better_w_missing),
                                    str(is_always_better_w_missing),
                                    str(num_times_is_better_wo_missing),
                                    str(is_always_better_wo_missing)]),
                          file=fout)


    with open(aggregated_statistics_csv_filepath_4, 'a') as fout:
        print(','.join(['Criterion', 'Two-sided p-value cutoff',
                        'Number of times is statistically better with missing values',
                        'Is always better',
                        'Number of times is statistically better without missing values',
                        'Is always better']),
              file=fout)
        num_criteria = len(aux_without_datasets_and_attributes_and_heights)
        for criterion in aux_without_datasets_and_attributes_and_heights:
            for p_value in aux_without_datasets_and_attributes_and_heights[criterion]:
                num_times_is_better_w_missing = aux_without_datasets_and_attributes_and_heights[
                    criterion][p_value][0]
                is_always_better_w_missing = (
                    num_times_is_better_w_missing == num_criteria - 1)
                num_times_is_better_wo_missing = aux_without_datasets_and_attributes_and_heights[
                    criterion][p_value][1]
                is_always_better_wo_missing = (
                    num_times_is_better_wo_missing == num_criteria - 1)
                print(','.join([criterion,
                                str(p_value),
                                str(num_times_is_better_w_missing),
                                str(is_always_better_w_missing),
                                str(num_times_is_better_wo_missing),
                                str(is_always_better_wo_missing)]),
                      file=fout)
    print('Done')


def init_t_test_infos(aggregated_csv_information):
    t_test_infos = {}
    for dataset in aggregated_csv_information:
        t_test_infos[dataset] = {}
        for attribute_name in aggregated_csv_information[dataset]:
            t_test_infos[dataset][attribute_name] = {}
            for height in aggregated_csv_information[dataset][attribute_name]:
                t_test_infos[dataset][attribute_name][height] = {}
    return t_test_infos


def init_original_statistics(aggregated_csv_information):
    original_statistics = {}
    for dataset in aggregated_csv_information:
        original_statistics[dataset] = {}
        for attribute_name in aggregated_csv_information[dataset]:
            original_statistics[dataset][attribute_name] = {}
            for height in aggregated_csv_information[dataset][attribute_name]:
                original_statistics[dataset][attribute_name][height] = {}
    return original_statistics


def init_boolean_information(aggregated_csv_information):
    boolean_information = {}
    for dataset in aggregated_csv_information:
        boolean_information[dataset] = {}
        for attribute_name in aggregated_csv_information[dataset]:
            boolean_information[dataset][attribute_name] = {}
            for height in aggregated_csv_information[dataset][attribute_name]:
                boolean_information[dataset][attribute_name][height] = {}
                for criterion in aggregated_csv_information[dataset][attribute_name][height]:
                    boolean_information[dataset][attribute_name][height][criterion] = {}
                    for p_value_cutoff in TWO_TAILED_P_VALUE_CUTOFFS:
                        boolean_information[dataset][attribute_name][height][criterion][
                            p_value_cutoff] = [0, 0]
    return boolean_information



def main():
    # csv_information[dataset][attribute_name][height][criterion_name][seed_index][fold_number] =
    # (accuracy_w_missing_values, num_validation_samples_w_missing,
    #  accuracy_wo_missing_values, num_validation_samples_wo_missing)
    csv_information = load_test_results(CSV_FILEPATH)
    # aggregated_csv_information[dataset][attribute_name][height][criterion_name][seed_index] =
    # (total_accuracy_w_missing_values, total_accuracy_wo_missing_values)
    aggregated_csv_information = aggregate_fold_info(csv_information)

    # t_test_infos[dataset][attribute_name][height][criteria_diff_name] =
    # (t_statistic_w_missing, two_sided_p_value_w_missing,
    #  t_statistic_wo_missing, two_sided_p_value_wo_missing)
    t_test_infos = init_t_test_infos(aggregated_csv_information)
    # boolean_information[dataset][attribute_name][height][criterion_name][p_value] = (
    # num_greater_w_missing, num_greater_wo_missing)
    boolean_information = init_boolean_information(aggregated_csv_information)
    # original_statistics[dataset][attribute_name][height][criterion_name] = (
    # (original_mean_w_missing, original_standard_dev_w_missing),
    # (original_mean_wo_missing, original_standard_dev_wo_missing))
    original_statistics = init_original_statistics(aggregated_csv_information)
    print('Calculating paired t-statistics')
    for dataset in aggregated_csv_information:
        for attribute_name in aggregated_csv_information[dataset]:
            for height in aggregated_csv_information[dataset][attribute_name]:

                for criterion in aggregated_csv_information[dataset][attribute_name][height]:
                    (original_samples_w_missing_values,
                     original_samples_wo_missing_values) = zip(*aggregated_csv_information[
                         dataset][attribute_name][height][criterion])
                    original_statistics[dataset][attribute_name][height][criterion] = (
                        calculate_original_statistics(original_samples_w_missing_values),
                        calculate_original_statistics(original_samples_wo_missing_values))

                for criteria_pair in itertools.combinations(
                        aggregated_csv_information[dataset][attribute_name][height], 2):
                    criterion_1, criterion_2 = criteria_pair
                    criteria_diff_name = ' - '.join(['(' + str(criterion_1) + ')',
                                                     '(' + str(criterion_2) + ')'])

                    (first_samples_w_missing_values,
                     first_samples_wo_missing_values) = zip(*aggregated_csv_information[
                         dataset][attribute_name][height][criterion_1])
                    (second_samples_w_missing_values,
                     second_samples_wo_missing_values) = zip(*aggregated_csv_information[
                         dataset][attribute_name][height][criterion_2])

                    diff_sample_w_missing_values = calculate_diff(first_samples_w_missing_values,
                                                                  second_samples_w_missing_values)
                    (t_stats_w_missing,
                     two_tailed_p_value_w_missing) = calculate_t_statistic(
                         diff_sample_w_missing_values)

                    diff_sample_wo_missing_values = calculate_diff(first_samples_wo_missing_values,
                                                                   second_samples_wo_missing_values)
                    (t_stats_wo_missing,
                     two_tailed_p_value_wo_missing) = calculate_t_statistic(
                         diff_sample_wo_missing_values)
                    # print('-'*80)
                    # print('t statistics w missing:', t_stats_w_missing)
                    # print('t statistics wo missing:', t_stats_wo_missing)
                    # print('2-tailed w missing:', two_tailed_p_value_w_missing)
                    # print('2-tailed p-value wo missing:', two_tailed_p_value_wo_missing)

                    t_test_infos[dataset][attribute_name][height][criteria_diff_name] = (
                        t_stats_w_missing,
                        two_tailed_p_value_w_missing,
                        t_stats_wo_missing,
                        two_tailed_p_value_wo_missing)
                    save_comparison_booleans(dataset, attribute_name, height, criterion_1,
                                             criterion_2, t_stats_w_missing,
                                             two_tailed_p_value_w_missing, t_stats_wo_missing,
                                             two_tailed_p_value_wo_missing, boolean_information)
    print('Done')
    write_stats(t_test_infos, STATISTICS_CSV_FILEPATH)
    write_aggreg_stats(boolean_information,
                       original_statistics,
                       AGGREG_STATISTICS_CSV_FILEPATH_1,
                       AGGREG_STATISTICS_CSV_FILEPATH_2,
                       AGGREG_STATISTICS_CSV_FILEPATH_3,
                       AGGREG_STATISTICS_CSV_FILEPATH_4)


if __name__ == '__main__':
    CSV_FILENAME = ('decision_tree_FULL_TEST_RESULTS_correct_seeds.csv')
    CSV_FILEPATH = CSV_FILENAME # os.path.join('.', 'outputs_from_datasets', CSV_FILENAME)
    STATISTICS_CSV_FILEPATH = 'statistics_output.csv'
    AGGREG_STATISTICS_CSV_FILEPATH_1 = 'aggregated_statistics_output_1.csv'
    AGGREG_STATISTICS_CSV_FILEPATH_2 = 'aggregated_statistics_output_2.csv'
    AGGREG_STATISTICS_CSV_FILEPATH_3 = 'aggregated_statistics_output_3.csv'
    AGGREG_STATISTICS_CSV_FILEPATH_4 = 'aggregated_statistics_output_4.csv'

    HAS_HEADER = True
    MUST_CLEAN_HEADER_REPETITION = False

    DATASET_COLUMN_INDEX = 0
    ATTRIBUTE_COLUMN_INDEX = None
    HEIGHT_COLUMN_INDEX = 4
    CRITERION_NAME_COLUMN_INDEX = 3
    SEED_INDEX_COLUMN_INDEX = 223
    FOLD_NUMBER_COLUMN_INDEX = None
    ACCURACY_W_MISSING_COLUMN_INDEX = 7
    NUM_VALIDATION_SAMPLES_W_MISSING_COLUMN_INDEX = 8
    ACCURACY_WO_MISSING_COLUMN_INDEX = 9
    NUM_VALIDATION_SAMPLES_WO_MISSING_COLUMN_INDEX = 10

    TWO_TAILED_P_VALUE_CUTOFFS = [0.2, 0.1, 0.02, 0.01]

    main()
