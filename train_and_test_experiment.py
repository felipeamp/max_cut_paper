#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''Module used to run train and test tests with decision trees, usually with maximum depth == 1.
'''


import datetime
import itertools
import os
import random
import sys
import timeit

import criteria
import dataset
import decision_tree

import numpy as np


#: Initial seeds used in `random` and `numpy.random` modules, in order of `trial_number`.
RANDOM_SEEDS = [65537, 986112772, 580170418, 897083807, 1286664107, 899169460, 1728505703,
                423232363, 1576030107, 1102706565, 756372267, 1041481669, 500571641, 1196189230,
                49471178, 827006267, 1581871235, 1249719834, 1281093615, 603059048, 1217122226,
                1784259850, 1636348000, 169498007, 1644610044, 1001000160, 884435702, 759171700,
                1729486164, 735355316, 590274769, 1685315218, 1811339189, 1436392076, 966320783,
                332035403, 1477247432, 1277551165, 395864655, 1334785552, 1863196977, 420054870,
                691025606, 1670255402, 535409696, 1556940403, 1036018082, 1120042937, 2128605734,
                1359372989, 335126928, 2109877295, 2070420066, 276135419, 1966874990, 1599483352,
                509177296, 8165980, 95787270, 343114090, 1938652731, 487748814, 1904611130,
                828040489, 620410008, 1013438160, 1422307526, 140262428, 1885459289, 116052729,
                1232834979, 708634310, 1761972120, 1247444947, 1585555404, 1859131028, 455754736,
                286190308, 1082412114, 2050198702, 998783919, 1496754253, 1371389911, 1314822048,
                1157568092, 332882253, 1647703149, 2011051574, 1222161240, 1358795771, 927702031,
                760815609, 504204359, 1424661575, 1228406087, 1971630940, 1758874112, 1403628276,
                643422904, 1196432617]

#: Maximum number of random training sample generation before giving up.
MAX_RANDOM_TRIES = 10


def main(experiment_config):
    """Sets the configurations according to `experiment_config` and runs them.
    """
    raw_output_filepath = os.path.join(experiment_config["output folder"], 'raw_output.csv')
    with open(raw_output_filepath, 'w') as fout:
        init_raw_output_csv(fout, output_split_char=',')
        criteria_list = get_criteria(experiment_config["criteria"])

        if experiment_config["prunning parameters"]["use chi-sq test"]:
            max_p_value_chi_sq = experiment_config["prunning parameters"]["max chi-sq p-value"]
            decision_tree.MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE = experiment_config[
                "prunning parameters"]["second most freq value min samples"]
        else:
            max_p_value_chi_sq = None
            decision_tree.MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE = None

        decision_tree.USE_MIN_SAMPLES_SECOND_LARGEST_CLASS = experiment_config[
            "prunning parameters"]["use second most freq class min samples"]
        if decision_tree.USE_MIN_SAMPLES_SECOND_LARGEST_CLASS:
            decision_tree.MIN_SAMPLES_SECOND_LARGEST_CLASS = experiment_config[
                "prunning parameters"]["second most freq class min samples"]
        else:
            decision_tree.MIN_SAMPLES_SECOND_LARGEST_CLASS = None

        if experiment_config["use all datasets"]:
            datasets_configs = dataset.load_all_configs(experiment_config["datasets basepath"])
            datasets = dataset.load_all_datasets(datasets_configs)

            if not experiment_config["use numeric attributes"]:
                for _, curr_dataset in datasets:
                    curr_dataset.valid_numeric_attribute = [False] * len(
                        curr_dataset.valid_numeric_attribute)

            for ((dataset_name, curr_dataset),
                 min_num_samples_allowed) in itertools.product(
                     datasets,
                     experiment_config["prunning parameters"]["min num samples allowed"]):
                for criterion in criteria_list:
                    print('-'*100)
                    print(criterion.name)
                    print()
                    run(dataset_name,
                        curr_dataset,
                        experiment_config["num training samples"],
                        criterion,
                        min_num_samples_allowed=min_num_samples_allowed,
                        max_depth=experiment_config["max depth"],
                        num_trials=experiment_config["num trials"],
                        use_numeric_attributes=experiment_config["use numeric attributes"],
                        use_chi_sq_test=experiment_config["prunning parameters"]["use chi-sq test"],
                        max_p_value_chi_sq=max_p_value_chi_sq,
                        output_file_descriptor=fout,
                        output_split_char=',')
        else:
            datasets_folders = [os.path.join(experiment_config["datasets basepath"], folderpath)
                                for folderpath in experiment_config["datasets folders"]]
            datasets_configs = [dataset.load_config(folderpath)
                                for folderpath in datasets_folders]
            datasets_configs.sort(key=lambda x: x["dataset name"].lower())

            for (dataset_config,
                 min_num_samples_allowed) in itertools.product(
                     datasets,
                     experiment_config["min num samples allowed"]):
                curr_dataset = dataset.Dataset(dataset_config["filepath"],
                                               dataset_config["key attrib index"],
                                               dataset_config["class attrib index"],
                                               dataset_config["split char"],
                                               dataset_config["missing value string"])

                if not experiment_config["use numeric attributes"]:
                    curr_dataset.valid_numeric_attribute = [False] * len(
                        curr_dataset.valid_numeric_attribute)

                for criterion in criteria_list:
                    print('-'*100)
                    print(criterion.name)
                    print()
                    run(dataset_config["dataset name"],
                        curr_dataset,
                        experiment_config["num training samples"],
                        criterion,
                        min_num_samples_allowed=min_num_samples_allowed,
                        max_depth=experiment_config["max depth"],
                        num_trials=experiment_config["num trials"],
                        use_numeric_attributes=experiment_config["use numeric attributes"],
                        use_chi_sq_test=experiment_config["prunning parameters"]["use chi-sq test"],
                        max_p_value_chi_sq=max_p_value_chi_sq,
                        output_file_descriptor=fout,
                        output_split_char=',')


def init_raw_output_csv(raw_output_file_descriptor, output_split_char=','):
    """Writes the header to the raw output CSV file.
    """
    fields_list = ['Date Time',
                   'Dataset',
                   'Total Number of Samples',
                   'Number of Training Samples',
                   'Number of Test Samples',
                   'Trial Number',
                   'Use Numeric Attributes',
                   'Criterion',
                   'Maximum Depth Allowed',

                   'Number of Samples Forcing a Leaf',
                   'Use Min Samples in Second Largest Class?',
                   'Min Samples in Second Largest Class',

                   'Uses Chi-Square Test',
                   'Maximum p-value Allowed by Chi-Square Test [between 0 and 1]',
                   'Minimum Number in Second Most Frequent Value',

                   'Number of Valid Attributes in Root Node (m)',

                   'Total Time Taken [s]',
                   'Time Taken to Create Tree [s]',
                   'Time Taken Prunning Trivial Subtrees [s]',

                   'Accuracy Percentage on Trivial Tree (with no splits)',

                   'Accuracy Percentage (with missing values)',
                   'Accuracy Percentage (without missing values)',
                   'Number of Samples with Unkown Values for Accepted Attribute',
                   'Percentage of Samples with Unkown Values for Accepted Attribute',

                   'Number of Nodes (after prunning)',
                   'Tree Depth (after prunning)',
                   'Number of Nodes Pruned']
    print(output_split_char.join(fields_list), file=raw_output_file_descriptor)
    raw_output_file_descriptor.flush()


def get_criteria(criteria_names_list):
    """Given a list of criteria names, returns a list of all there criteria (as `Criterion`'s).
    If a criterion name is unkown, the system will exit the experiment.
    """
    criteria_list = []
    for criterion_name in criteria_names_list:
        if criterion_name == "Twoing":
            criteria_list.append(criteria.Twoing())
        elif criterion_name == "GW Squared Gini":
            criteria_list.append(criteria.GWSquaredGini())
        elif criterion_name == "GW Chi Square":
            criteria_list.append(criteria.GWChiSquare())
        elif criterion_name == "LS Squared Gini":
            criteria_list.append(criteria.LSSquaredGini())
        elif criterion_name == "LS Chi Square":
            criteria_list.append(criteria.LSChiSquare())
        else:
            print('Unkown criterion name:', criterion_name)
            print('Exiting.')
            sys.exit(1)
    return criteria_list


def run(dataset_name, train_dataset, num_training_samples, criterion, min_num_samples_allowed,
        max_depth, num_trials, use_numeric_attributes, use_chi_sq_test, max_p_value_chi_sq,
        output_file_descriptor, output_split_char=',', seed=None):
    """Runs `num_trials` experiments, each one randomly selecting `num_training_samples` valid
    samples to use for training and testing the tree in the rest of the dataset. Saves the training
    and classification information in the `output_file_descriptor` file.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    training_samples_indices = list(range(train_dataset.num_samples))
    for trial_number in range(num_trials):
        print('*'*80)
        print('STARTING TRIAL #{}'.format(trial_number + 1))
        print()

        if seed is None:
            random.seed(RANDOM_SEEDS[trial_number])
            np.random.seed(RANDOM_SEEDS[trial_number])
        random.shuffle(training_samples_indices)
        curr_training_samples_indices = training_samples_indices[:num_training_samples]
        curr_test_samples_indices = training_samples_indices[num_training_samples:]

        tree = decision_tree.DecisionTree(criterion=criterion)
        # First let's train the tree and save the training information
        start_time = timeit.default_timer()
        (time_taken_prunning,
         num_nodes_prunned) = tree.train(dataset=train_dataset,
                                         training_samples_indices=curr_training_samples_indices,
                                         max_depth=max_depth,
                                         min_samples_per_node=min_num_samples_allowed,
                                         use_stop_conditions=use_chi_sq_test,
                                         max_p_value_chi_sq=max_p_value_chi_sq)
        total_time_taken = timeit.default_timer() - start_time

        num_random_tries = 1
        while (sorted(tree.get_root_node().class_index_num_samples)[-2] == 0
               or sum(tree.get_root_node().valid_nominal_attribute) == 0):
            num_random_tries += 1
            if num_random_tries == MAX_RANDOM_TRIES:
                print('Already did {} random generation, none worked (only one class or no valid'
                      ' attribute).'.format(MAX_RANDOM_TRIES))
                print('Will skip to the next test.')
                return None

            random.shuffle(training_samples_indices)
            curr_training_samples_indices = training_samples_indices[:num_training_samples]
            curr_test_samples_indices = training_samples_indices[
                num_training_samples: 2 * num_training_samples]

            start_time = timeit.default_timer()
            (time_taken_prunning,
             num_nodes_prunned) = tree.train(dataset=train_dataset,
                                             training_samples_indices=curr_training_samples_indices,
                                             max_depth=max_depth,
                                             min_samples_per_node=min_num_samples_allowed,
                                             use_stop_conditions=use_chi_sq_test,
                                             max_p_value_chi_sq=max_p_value_chi_sq)
            total_time_taken = timeit.default_timer() - start_time

        num_valid_nominal_attributes = sum(tree.get_root_node().valid_nominal_attribute)

        time_taken_tree = total_time_taken - time_taken_prunning

        # Time to test this tree's classification and save the classification information
        trivial_accuracy = tree.get_trivial_accuracy(curr_test_samples_indices)
        (_,
         num_correct_classifications_w_unkown,
         num_correct_classifications_wo_unkown,
         _,
         _,
         _,
         num_unkown,
         _) = tree.test(curr_test_samples_indices)

        accuracy_with_missing_values = (100.0 * num_correct_classifications_w_unkown
                                        / len(curr_test_samples_indices))
        try:
            accuracy_without_missing_values = (100.0 * num_correct_classifications_wo_unkown
                                               / (len(curr_test_samples_indices) - num_unkown))
        except ZeroDivisionError:
            accuracy_without_missing_values = None
        percentage_unkown = 100.0 * num_unkown / len(curr_test_samples_indices)

        num_nodes_found = tree.get_root_node().get_num_nodes()
        max_depth_found = tree.get_root_node().get_max_depth()

        save_trial_info(dataset_name, train_dataset.num_samples, num_training_samples,
                        trial_number + 1, use_numeric_attributes, criterion.name, max_depth,
                        min_num_samples_allowed, decision_tree.USE_MIN_SAMPLES_SECOND_LARGEST_CLASS,
                        decision_tree.MIN_SAMPLES_SECOND_LARGEST_CLASS,
                        use_chi_sq_test, max_p_value_chi_sq,
                        decision_tree.MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE,
                        num_valid_nominal_attributes, total_time_taken, time_taken_tree,
                        time_taken_prunning, trivial_accuracy, accuracy_with_missing_values,
                        accuracy_without_missing_values, num_unkown, percentage_unkown,
                        num_nodes_found, max_depth_found, num_nodes_prunned, output_split_char,
                        output_file_descriptor)


def save_trial_info(dataset_name, num_total_samples, num_training_samples, trial_number,
                    use_numeric_attributes, criterion_name, max_depth, min_num_samples_allowed,
                    use_min_samples_second_largest_class, min_samples_second_largest_class,
                    use_chi_sq_test, max_p_value_chi_sq, min_num_second_most_freq_value,
                    num_valid_nominal_attributes, total_time_taken, time_taken_tree,
                    time_taken_prunning, trivial_accuracy_percentage, accuracy_with_missing_values,
                    accuracy_without_missing_values, num_unkown, percentage_unkown, num_nodes_found,
                    max_depth_found, num_nodes_prunned, output_split_char, output_file_descriptor):
    """Saves the experiment's trial information in the CSV file.
    """
    line_list = [str(datetime.datetime.now()),
                 dataset_name,
                 str(num_total_samples),
                 str(num_training_samples),
                 str(num_total_samples - num_training_samples),
                 str(trial_number),
                 str(use_numeric_attributes),
                 criterion_name,
                 str(max_depth),

                 str(min_num_samples_allowed),
                 str(use_min_samples_second_largest_class),
                 str(min_samples_second_largest_class),

                 str(use_chi_sq_test),
                 str(max_p_value_chi_sq),
                 str(min_num_second_most_freq_value),

                 str(num_valid_nominal_attributes),

                 str(total_time_taken),
                 str(time_taken_tree),
                 str(time_taken_prunning),

                 str(trivial_accuracy_percentage),

                 str(accuracy_with_missing_values),
                 str(accuracy_without_missing_values),
                 str(num_unkown),
                 str(percentage_unkown),

                 str(num_nodes_found),
                 str(max_depth_found),
                 str(num_nodes_prunned)]
    print(output_split_char.join(line_list), file=output_file_descriptor)
    output_file_descriptor.flush()
