#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''Module used to calculate t-student statistics of experiments.
'''

import collections
import itertools
import json
import math
import os
import statistics
import sys

from scipy.stats import t as student_t


ColumnIndices = collections.namedtuple('ColumnIndices',
                                       ['dataset_col',
                                        'attribute_col',
                                        'num_values_col',
                                        'criterion_col',
                                        'trial_number_col',
                                        'fold_number_col',
                                        'accuracy_w_missing_col',
                                        'accuracy_wo_missing_col'])

#: Contain the column indices for a rank experiment
RANK_COLUMN_INDICES = ColumnIndices(dataset_col=1,
                                    attribute_col=3,
                                    num_values_col=5,
                                    criterion_col=8,
                                    trial_number_col=7,
                                    fold_number_col=11,
                                    accuracy_w_missing_col=29,
                                    accuracy_wo_missing_col=30)

#: Contain the column indices for a cross-validation experiment
CROSS_VALIDATION_COLUMN_INDICES = ColumnIndices(dataset_col=1,
                                                attribute_col=None,
                                                num_values_col=None,
                                                criterion_col=4,
                                                trial_number_col=3,
                                                fold_number_col=None,
                                                accuracy_w_missing_col=20,
                                                accuracy_wo_missing_col=21)

#: Contain the column indices for a train-and-test experiment
TRAIN_AND_TEST_COLUMN_INDICES = ColumnIndices(dataset_col=1,
                                              attribute_col=None,
                                              num_values_col=None,
                                              criterion_col=7,
                                              trial_number_col=5,
                                              fold_number_col=None,
                                              accuracy_w_missing_col=20,
                                              accuracy_wo_missing_col=21)


def main(output_path):
    '''Calculates the t-student statistics of experiments contained in this folder.

    The `output_path` folder must contain the `raw_output.csv` file and the `experiment_config.json`
    file, otherwise the function will exit.
    '''

    raw_output_path = os.path.join(output_path, 'raw_output.csv')
    if (not os.path.exists(raw_output_path)
            or not os.path.isfile(raw_output_path)):
        print('This path does not contain the output of an experiment.')
        sys.exit(1)

    experiment_config_filepath = os.path.join(output_path, 'experiment_config.json')
    if (not os.path.exists(experiment_config_filepath)
            or not os.path.isfile(experiment_config_filepath)):
        print('This path does not contain the output of an experiment.')
        sys.exit(1)
    with open(experiment_config_filepath, 'r') as experiment_config_json:
        experiment_config = json.load(experiment_config_json)
    if "min num values to compare" in experiment_config:
        min_num_values_to_compare = experiment_config["min num values to compare"]
    else:
        min_num_values_to_compare = 2

    if experiment_config["rank attributes"]:
        is_rank = True
        column_indices = RANK_COLUMN_INDICES
    elif experiment_config["use cross-validation"]:
        column_indices = CROSS_VALIDATION_COLUMN_INDICES
    else:
        column_indices = TRAIN_AND_TEST_COLUMN_INDICES

    single_sided_p_value_threshold = experiment_config["t-test single-sided p-value"]

    raw_data = _load_raw_data(raw_output_path, column_indices, is_rank, min_num_values_to_compare)
    _save_raw_stats(raw_data, output_path, is_rank)
    _save_aggreg_stats(raw_data, single_sided_p_value_threshold)


def _load_raw_data(raw_output_path, column_indices, is_rank, min_num_values_to_compare=2):
    def _init_raw_data():
        # This function creates (in a lazy way) an infinitely-nested default dict. This is
        # useful when creating a default dict highly nested.
        return collections.defaultdict(_init_raw_data)

    raw_data = _init_raw_data()
    has_read_header = False
    with open(raw_output_path, 'r') as fin:
        for line in fin:
            if not has_read_header:
                has_read_header = True
                continue
            line_list = line.split(',')

            dataset_name = line_list[column_indices.dataset_col]
            criterion_name = line_list[column_indices.criterion_col]
            trial_number = line_list[column_indices.trial_number_col]

            accuracy_w_missing = line_list[column_indices.accuracy_w_missing_col]
            accuracy_wo_missing = line_list[column_indices.accuracy_wo_missing_col]
            if is_rank:
                num_values = line_list[column_indices.num_values_col]
                if num_values < min_num_values_to_compare:
                    continue
                attribute_name = line_list[column_indices.attribute_col]
                fold_number = line_list[column_indices.fold_number_col]
                raw_data[dataset_name][attribute_name][criterion_name][trial_number][
                    fold_number] = (accuracy_w_missing,
                                    accuracy_wo_missing)
            else:
                raw_data[dataset_name][criterion_name][trial_number] = (accuracy_w_missing,
                                                                        accuracy_wo_missing)
    return raw_data


def _save_raw_stats(raw_data, output_path, is_rank):
    raw_stats_output_file = os.path.join(output_path, 'raw_t_student_stats.csv')
    with open(raw_stats_output_file, 'w') as fout:
        header = ['Dataset',
                  'Attribute',
                  'Criterion Difference Name',
                  'Paired t-statistics on Accuracy with Missing Values',
                  'Degrees of Freedom of Accuracy with Missing Values',
                  'P-value t-statistics on Accuracy with Missing Values',
                  'Paired t-statistics on Accuracy without Missing Values',
                  'P-value t-statistics on Accuracy without Missing Values',
                  'Degrees of Freedom of Accuracy without Missing Values']
        print(','.join(header), file=fout)
        if is_rank:
            for dataset_name in raw_data:
                for attribute_name in raw_data[dataset_name]:
                    for (criterion_name_1,
                         criterion_name_2) in itertools.combinations(
                             raw_data[dataset_name][attribute_name], 2):

                        criterion_diff_name = ' - '.join((criterion_name_1, criterion_name_2))
                        accuracy_w_missing_diff = []
                        accuracy_wo_missing_diff = []

                        for trial_number in raw_data[dataset_name][criterion_name_1]:
                            for fold_number in raw_data[dataset_name][criterion_name_1][
                                    trial_number]:
                                criterion_1_accuracies = raw_data[dataset_name][criterion_name_1][
                                    trial_number][fold_number]
                                criterion_2_accuracies = raw_data[dataset_name][criterion_name_2][
                                    trial_number][fold_number]

                                accuracy_w_missing_diff.append(
                                    criterion_1_accuracies[0] - criterion_2_accuracies[0])
                                if (criterion_1_accuracies[1] is not None
                                        and criterion_2_accuracies[1] is not None):
                                    accuracy_wo_missing_diff.append(
                                        criterion_1_accuracies[1] - criterion_2_accuracies[1])

                        (t_statistic_w_missing,
                         p_value_w_missing) = _calculate_t_statistic(accuracy_w_missing_diff)
                        (t_statistic_wo_missing,
                         p_value_wo_missing) = _calculate_t_statistic(accuracy_wo_missing_diff)
                        print(','.join([dataset_name,
                                        attribute_name,
                                        criterion_diff_name,
                                        str(t_statistic_w_missing),
                                        str(len(accuracy_w_missing_diff) - 1),
                                        str(p_value_w_missing),
                                        str(t_statistic_wo_missing),
                                        str(len(accuracy_wo_missing_diff) - 1),
                                        str(p_value_wo_missing)]),
                              file=fout)
        else:
            for dataset_name in raw_data:
                for (criterion_name_1,
                     criterion_name_2) in itertools.combinations(raw_data[dataset_name], 2):

                    criterion_diff_name = ' - '.join((criterion_name_1, criterion_name_2))
                    accuracy_w_missing_diff = []
                    accuracy_wo_missing_diff = []

                    for trial_number in raw_data[dataset_name][criterion_name_1]:
                        criterion_1_accuracies = raw_data[dataset_name][criterion_name_1][
                            trial_number]
                        criterion_2_accuracies = raw_data[dataset_name][criterion_name_2][
                            trial_number]

                        accuracy_w_missing_diff.append(
                            criterion_1_accuracies[0] - criterion_2_accuracies[0])
                        if (criterion_1_accuracies[1] is not None
                                and criterion_2_accuracies[1] is not None):
                            accuracy_wo_missing_diff.append(
                                criterion_1_accuracies[1] - criterion_2_accuracies[1])

                    (t_statistic_w_missing,
                     p_value_w_missing) = _calculate_t_statistic(accuracy_w_missing_diff)
                    (t_statistic_wo_missing,
                     p_value_wo_missing) = _calculate_t_statistic(accuracy_wo_missing_diff)
                    print(','.join([dataset_name,
                                    str(None),
                                    criterion_diff_name,
                                    str(t_statistic_w_missing),
                                    str(len(accuracy_w_missing_diff) - 1),
                                    str(p_value_w_missing),
                                    str(t_statistic_wo_missing),
                                    str(len(accuracy_wo_missing_diff) - 1),
                                    str(p_value_wo_missing)]),
                          file=fout)


def _save_aggreg_stats(output_path, single_sided_p_value_threshold):
    # aggreg_data[(dataset, attribute, criterion)] = [num_times_stat_better_w_missing,
    #                                                 num_times_stat_better_wo_missing]
    aggreg_data = collections.defaultdict(lambda: [0, 0])

    raw_stats_output_file = os.path.join(output_path, 'raw_t_student_stats.csv')
    has_read_header = False
    with open(raw_stats_output_file, 'r') as fin:
        for line in fin:
            if not has_read_header:
                has_read_header = True
                continue
            line_list = line.split(',')

            dataset_name = line_list[0]
            attribute = line_list[1]
            criterion_diff_name = line_list[2]
            criterion_name_1, criterion_name_2 = criterion_diff_name.split(' - ')

            p_value_w_missing = line_list[5]
            if p_value_w_missing <= single_sided_p_value_threshold:
                aggreg_data[(dataset_name, attribute, criterion_name_1)][0] += 1
            elif p_value_w_missing >= 1. - single_sided_p_value_threshold:
                aggreg_data[(dataset_name, attribute, criterion_name_2)][0] += 1

            p_value_wo_missing = line_list[8]
            if p_value_wo_missing <= single_sided_p_value_threshold:
                aggreg_data[(dataset_name, attribute, criterion_name_1)][1] += 1
            elif p_value_wo_missing >= 1. - single_sided_p_value_threshold:
                aggreg_data[(dataset_name, attribute, criterion_name_2)][1] += 1

    aggreg_stats_output_file = os.path.join(output_path, 'aggreg_t_student_stats.csv')
    with open(aggreg_stats_output_file, 'w') as fout:
        header = ['Dataset',
                  'Attribute',
                  'Criterion',
                  'Number of times is statistically better with missing values',
                  'Number of times is statistically better without missing values']
        print(','.join(header), file=fout)
        for key, value in aggreg_data.items():
            print(','.join([*key, *value]), file=fout)

def _calculate_t_statistic(samples_list):
    mean = statistics.mean(samples_list)
    variance = statistics.variance(samples_list)
    if variance == 0.0:
        return 0.0, 100.0

    num_samples = len(samples_list)
    t_statistic = mean / math.sqrt(variance / num_samples)
    degrees_of_freedom = num_samples - 1
    p_value = 1. - student_t.cdf(t_statistic, degrees_of_freedom)
    return t_statistic, p_value


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please include a path to an experiment output folder.')
        sys.exit(1)

    main(sys.argv[1])
