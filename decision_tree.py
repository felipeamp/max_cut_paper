#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing the DecisionTree, TreeNode and NodeSplit classes.
"""

import math
import random
import sys
import timeit

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import chi2


#: Minimum number of samples needed in the two most frequent values of an attribute such that it is
#: considered valid.
MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE = 40

#: Use the minimum number of samples in the two most frequent classes as a criteria to allow a split
#: or not.
USE_MIN_SAMPLES_SECOND_LARGEST_CLASS = True
#: Minimum number of samples needed in the two most frequent classes such that this node can be
#: split.
MIN_SAMPLES_SECOND_LARGEST_CLASS = 40


class DecisionTree(object):
    """Data structure containing basic information pertaining to the whole tree.

        This class' state should be accessed only indirectly, through its methods.
    """
    def __init__(self, criterion):
        """Initializes a DecisionTree instance with the given arguments.

        Args:
            criterion (Criterion): criterion which will be used to generate the tree nodes/splits.
        """
        self._criterion = criterion
        self._dataset = None
        self._root_node = None

    def get_root_node(self):
        """Returns the TreeNode at the root of the tree. Might be None.
        """
        return self._root_node

    def get_trivial_accuracy(self, test_samples_indices):
        """Returns the accuracy obtained by classifying all test samples in the most common class
        among training samples. Must be called after training the tree.
        """
        num_correct = sum(
            self._dataset.sample_class[curr_sample_index] == self._root_node.most_common_int_class
            for curr_sample_index in test_samples_indices)
        return 100.0 * num_correct / len(test_samples_indices)

    def _classify_sample(self, sample, sample_key):
        if self._root_node is None:
            print('Cannot classify in untrained tree!')
            sys.exit(1)
        curr_node = self._root_node
        classified_with_unkown_value = False
        unkown_value_attrib_index = None
        while not curr_node.is_leaf:
            split_attrib_index = curr_node.node_split.separation_attrib_index
            sample_value = sample[split_attrib_index]
            if self._dataset.valid_numeric_attribute[split_attrib_index]:
                if sample_value is None:
                    print('\tSample {} has value unkown to split'
                          ' (value = {} in attrib #{}).'.format(
                              sample_key,
                              sample_value,
                              curr_node.node_split.separation_attrib_index))
                    classified_with_unkown_value = True
                    unkown_value_attrib_index = curr_node.node_split.separation_attrib_index
                    break
                mid_point = curr_node.node_split.mid_point
                if sample_value <= mid_point:
                    curr_node = curr_node.nodes[0]
                else:
                    curr_node = curr_node.nodes[1]
            else:
                try:
                    split_index = curr_node.node_split.values_to_split[sample_value]
                    curr_node = curr_node.nodes[split_index]
                except KeyError:
                    print('\tSample {} has value unkown to split'
                          ' (value = {} in attrib #{}).'.format(
                              sample_key,
                              sample_value,
                              curr_node.node_split.separation_attrib_index))
                    classified_with_unkown_value = True
                    unkown_value_attrib_index = curr_node.node_split.separation_attrib_index
                    break
        return (curr_node.most_common_int_class,
                classified_with_unkown_value,
                unkown_value_attrib_index)

    def _classify_samples(self, test_dataset_sample, test_dataset_sample_class,
                          test_dataset_sample_costs, test_samples_indices,
                          test_dataset_sample_keys):
        print('Starting classifications...')
        classifications = []
        classified_with_unkown_value_array = []
        unkown_value_attrib_index_array = []

        num_correct_classifications = 0
        num_correct_classifications_wo_unkown = 0
        num_unkown = 0

        total_cost = 0.0
        total_cost_wo_unkown = 0.0

        for test_sample_index in test_samples_indices:
            (predicted_class,
             classified_with_unkown_value,
             unkown_value_attrib_index) = self._classify_sample(
                 test_dataset_sample[test_sample_index],
                 test_dataset_sample_keys[test_sample_index])
            classifications.append(predicted_class)
            classified_with_unkown_value_array.append(classified_with_unkown_value)
            unkown_value_attrib_index_array.append(unkown_value_attrib_index)
            if predicted_class == test_dataset_sample_class[test_sample_index]:
                num_correct_classifications += 1
                if not classified_with_unkown_value:
                    num_correct_classifications_wo_unkown += 1
            total_cost += test_dataset_sample_costs[test_sample_index][predicted_class]
            if not classified_with_unkown_value:
                total_cost_wo_unkown += test_dataset_sample_costs[
                    test_sample_index][predicted_class]
            else:
                num_unkown += 1
        print('Done!')
        return (classifications,
                num_correct_classifications,
                num_correct_classifications_wo_unkown,
                total_cost,
                total_cost_wo_unkown,
                classified_with_unkown_value_array,
                num_unkown,
                unkown_value_attrib_index_array)

    def train(self, dataset, training_samples_indices, max_depth, min_samples_per_node,
              use_stop_conditions=False, max_p_value_chi_sq=0.1):
        """Trains the tree in a recursive fashion, starting at the root's TreeNode. Afterwards,
        prunes the trivial subtrees.

        Args:
            dataset (Dataset): dataset containing the samples used for training.
            training_samples_indices (:obj:'list' of 'int'): list containing the indices of samples
                of `dataset` used for training.
            max_depth (int): maximum tree depth allowed. Zero means the root is a leaf.
            min_samples_per_node (int): if a node has less than this number of training samples, it
                will necessarily be a leaf.
            use_stop_conditions (bool, optional): informs wether we should use prunning techniques
                to avoid using attributes with small number of samples (and, thus, avoiding
                statistical anomalies). An attribute will be considered invalid if it contains less
                than `MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE` samples in the second most frequent
                value (this way at least two values have this number of samples) or if a chi-square
                test, applied on the attributes' contingency table has a p-value greater or equal to
                `max_p_value_chi_sq`. When an attribute is considered invalid for the number of
                samples in the second most frequent value, it will be considered invalid in every
                child node of the current TreeNode. If it was considered invalid because of the
                chi-square test, it can be considered valid in a descendant node. Note that numeric
                attributes are never tested in this way. Defaults to `False`.
            max_p_value_chi_sq (float, optional): is the maximum p-value allowed for an attribute to
                be accepted when doing chi-square tests (that is, when `use_stop_conditions` is
                `True`). A p-value of 1.0 is equal to 100%. Defaults to `0.1`.
        Returns:
            tuple containing, in order:
                - time_taken_prunning (float): time spent prunning the trained tree.
                - nodes_prunned (int): number of nodes prunned.
        """
        self._dataset = dataset
        print('Starting tree training...')
        self._root_node = TreeNode(dataset,
                                   training_samples_indices,
                                   dataset.valid_nominal_attribute[:],
                                   dataset.valid_numeric_attribute[:],
                                   max_depth,
                                   min_samples_per_node,
                                   use_stop_conditions,
                                   max_p_value_chi_sq)
        self._root_node.create_subtree(self._criterion)
        print('Starting prunning trivial subtrees...')
        start_time = timeit.default_timer()
        num_nodes_prunned = self._root_node.prune_trivial_subtrees()
        time_taken_prunning = timeit.default_timer() - start_time
        print('Done!')
        return time_taken_prunning, num_nodes_prunned

    def train_and_test(self, dataset, training_samples_indices, validation_sample_indices,
                       max_depth, min_samples_per_node, use_stop_conditions=False,
                       max_p_value_chi_sq=0.1):
        """Trains a tree with part of the dataset (training samples) and tests the tree
        classification in another part (validation samples).

        Note that although the training and test samples are part of the same Dataset class, they
        usually shouldn't intersect.

        Args:
            dataset (Dataset): dataset containing the samples used for training.
            training_samples_indices (:obj:'list' of 'int'): list containing the indices of samples
                of `dataset` used for training.
            validation_sample_indices (:obj:'list' of 'int'): list containing the indices of samples
                of `dataset` used to test the tree classification.
            max_depth (int): maximum tree depth allowed. Zero means the root is a leaf.
            min_samples_per_node (int): if a node has less than this number of training samples, it
                will necessarily be a leaf.
            use_stop_conditions (bool, optional): informs wether we should use prunning techniques
                to avoid using attributes with small number of samples (and, thus, avoiding
                statistical anomalies). An attribute will be considered invalid if it contains less
                than `MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE` samples in the second most frequent
                value (this way at least two values have this number of samples) or if a chi-square
                test, applied on the attributes' contingency table has a p-value greater or equal to
                `max_p_value_chi_sq`. When an attribute is considered invalid for the number of
                samples in the second most frequent value, it will be considered invalid in every
                child node of the current TreeNode. If it was considered invalid because of the
                chi-square test, it can be considered valid in a descendant node. Note that numeric
                attributes are never tested in this way. Defaults to `False`.
            max_p_value_chi_sq (float, optional): is the maximum p-value allowed for an attribute to
                be accepted when doing chi-square tests (that is, when `use_stop_conditions` is
                `True`). A p-value of 1.0 is equal to 100%. Defaults to `0.1`.

        Returns:
            A tuple containing the tree's max depth in the second entry, the time taken prunning
            in the third entry and the number of nodes prunned in the fourth entry. In the first
            entry it returns another tuple containing, in order:
                - a list of predicted class for each validation sample;
                - the number of correct classifications;
                - the number of correct classifications done without validation samples with unkown
                    values (that is, values that are unkown at a TreeNode -- they are classified as
                    the most common class at that node);
                - the total cost of the classification errors (when errors costs are uniform, this
                    is equal to the total number of validation samples minus the number of correct
                    classifications);
                -  the total cost of the classification errors without considering validation
                    samples with unkown values;
                - a list of booleans indicating if the i-th validation sample was classified with an
                    unkown value;
                - the number of validation samples classified with unkown values;
                - list where the i-th entry has the attribute index used for classification of the
                    i-th sample when an unkown value occurred.
        """
        time_taken_prunning, num_nodes_prunned = self.train(dataset,
                                                            training_samples_indices,
                                                            max_depth,
                                                            min_samples_per_node,
                                                            use_stop_conditions,
                                                            max_p_value_chi_sq)
        max_depth = self.get_root_node().get_max_depth()
        return (self._classify_samples(self._dataset.samples,
                                       self._dataset.sample_class,
                                       self._dataset.sample_costs,
                                       validation_sample_indices,
                                       self._dataset.sample_index_to_key),
                max_depth,
                time_taken_prunning,
                num_nodes_prunned)

    def cross_validate(self, dataset, num_folds, max_depth, min_samples_per_node,
                       is_stratified=True, print_tree=False, seed=None, print_samples=False,
                       use_stop_conditions=False, max_p_value_chi_sq=0.1):
        """Does a cross-validation using a given dataset.

        It splits this dataset in `num_folds` folds and calls `train_and_test` on each. Might be
        given a seed for the dataset's random splitting and might be stratified.

        Args:
            dataset (Dataset): dataset containing the samples used for training.
            num_folds (int): number of folds used in the cross-validation.
            max_depth (int): maximum tree depth allowed. Zero means the root is a leaf.
            min_samples_per_node (int): if a node has less than this number of training samples, it
                will necessarily be a leaf.
            is_stratified (bool, optional): Indicates wheter the cross-validation should be
                stratified or just a simple k-fold cross-validation. Stratified means the samples'
                splitting will try to keep the classes' distribution the same across folds. Defaults
                to `True`.
            print_tree (bool, optional): Indicates wether the trees of every fold should be printed
                to stdout. Defaults to `False`.
            seed (int, optional): indicates the seed that should be used to generate the random
                samples' splitting in folds. If `None`, a random seed is used. Defaults to `None`.
            print_samples (bool, optional): if `True`, prints the samples indices used at each fold.
                Used for debugging. Defaults to `False`.
            use_stop_conditions (bool, optional): informs wether we should use prunning techniques
                to avoid using attributes with small number of samples (and, thus, avoiding
                statistical anomalies). An attribute will be considered invalid if it contains less
                than `MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE` samples in the second most frequent
                value (this way at least two values have this number of samples) or if a chi-square
                test, applied on the attributes' contingency table has a p-value greater or equal to
                `max_p_value_chi_sq`. When an attribute is considered invalid for the number of
                samples in the second most frequent value, it will be considered invalid in every
                child node of the current TreeNode. If it was considered invalid because of the
                chi-square test, it can be considered valid in a descendant node. Note that numeric
                attributes are never tested in this way. Defaults to `False`.
            max_p_value_chi_sq (float, optional): is the maximum p-value allowed for an attribute to
                be accepted when doing chi-square tests (that is, when `use_stop_conditions` is
                `True`). A p-value of 1.0 is equal to 100%. Defaults to `0.1`.

        Returns:
            A tuple containing, in order:
                - a list of predicted class for each sample;
                - the number of correct classifications;
                - the number of correct classifications done without samples with unkown values;
                    (that is, values that are unkown at a TreeNode -- they are classified as the
                    most common class at that node);
                - the total cost of the classification errors (when errors costs are uniform, this
                    is equal to the total number of samples minus the number of correct
                    classifications);
                - the total cost of the classification errors without considering samples with
                    unkown values;
                - a list of booleans indicating if the i-th sample was classified with an unkown
                    value;
                - the number of samples classified with unkown values;
                - list where the i-th entry has the attribute index used for classification of the
                    i-th sample when an unkown value occurred;
                - list containing the time spent prunning in each fold;
                - list containing the number of nodes prunned in each fold;
                - list containing the maximum tree depth for each fold;
                - list containing the number of nodes per fold, after prunning;
                - list containing the number of valid attributes in root node in each fold;
                - Accuracy percentage obtained by classifying, in each fold, the test samples in the
                most common class among training samples.
        """

        classifications = [0] * dataset.num_samples
        num_correct_classifications = 0
        num_correct_classifications_wo_unkown = 0
        total_cost = 0.0
        total_cost_wo_unkown = 0.0
        classified_with_unkown_value_array = [False] * dataset.num_samples
        num_unkown = 0
        unkown_value_attrib_index_array = [0] * dataset.num_samples
        max_depth_per_fold = []
        num_nodes_per_fold = []
        num_valid_attributes_in_root = []
        time_taken_prunning_per_fold = []
        num_nodes_prunned_per_fold = []
        num_correct_trivial_classifications = 0

        fold_count = 0

        sample_indices_and_classes = list(enumerate(dataset.sample_class))
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        random.shuffle(sample_indices_and_classes)
        shuffled_sample_indices, shuffled_sample_classes = zip(*sample_indices_and_classes)

        if is_stratified:
            for (training_randomized_indices,
                 validation_randomized_indices) in StratifiedKFold(n_splits=num_folds).split(
                     shuffled_sample_indices,
                     shuffled_sample_classes):

                training_samples_indices = [shuffled_sample_indices[index]
                                            for index in training_randomized_indices]
                validation_sample_indices = [shuffled_sample_indices[index]
                                             for index in validation_randomized_indices]

                if print_samples:
                    print('Samples used for validation in this fold:')
                    print(validation_sample_indices)
                    print()

                ((curr_classifications,
                  curr_num_correct_classifications,
                  curr_num_correct_classifications_wo_unkown,
                  curr_total_cost,
                  curr_total_cost_wo_unkown,
                  curr_classified_with_unkown_value_array,
                  curr_num_unkown,
                  curr_unkown_value_attrib_index_array),
                 curr_max_depth,
                 curr_time_taken_prunning,
                 curr_num_nodes_prunned) = self.train_and_test(dataset,
                                                               training_samples_indices,
                                                               validation_sample_indices,
                                                               max_depth,
                                                               min_samples_per_node,
                                                               use_stop_conditions,
                                                               max_p_value_chi_sq)
                max_depth_per_fold.append(curr_max_depth)
                num_nodes_per_fold.append(self.get_root_node().get_num_nodes())
                num_valid_attributes_in_root.append(
                    sum(self._root_node.valid_nominal_attribute))
                for curr_index, validation_sample_index in enumerate(validation_sample_indices):
                    classifications[validation_sample_index] = curr_classifications[curr_index]
                    classified_with_unkown_value_array[validation_sample_index] = (
                        curr_classified_with_unkown_value_array[curr_index])
                    unkown_value_attrib_index_array[validation_sample_index] = (
                        curr_unkown_value_attrib_index_array[curr_index])
                num_correct_classifications += curr_num_correct_classifications
                num_correct_classifications_wo_unkown += curr_num_correct_classifications_wo_unkown
                total_cost += curr_total_cost
                total_cost_wo_unkown += curr_total_cost_wo_unkown
                num_unkown += curr_num_unkown
                num_correct_trivial_classifications += round(
                    len(validation_sample_indices) *
                    (self.get_trivial_accuracy(validation_sample_indices) / 100.0))

                fold_count += 1
                time_taken_prunning_per_fold.append(curr_time_taken_prunning)
                num_nodes_prunned_per_fold.append(curr_num_nodes_prunned)

                if print_tree:
                    print()
                    print('-' * 50)
                    print('Fold:', fold_count)
                    self.save_tree()
        else:
            for (training_samples_indices,
                 validation_sample_indices) in KFold(n_splits=num_folds).split(
                     shuffled_sample_indices):

                ((curr_classifications,
                  curr_num_correct_classifications,
                  curr_num_correct_classifications_wo_unkown,
                  curr_total_cost,
                  curr_total_cost_wo_unkown,
                  curr_classified_with_unkown_value_array,
                  curr_num_unkown,
                  curr_unkown_value_attrib_index_array),
                 curr_max_depth,
                 curr_time_taken_prunning,
                 curr_num_nodes_prunned) = self.train_and_test(dataset,
                                                               training_samples_indices,
                                                               validation_sample_indices,
                                                               max_depth,
                                                               min_samples_per_node,
                                                               use_stop_conditions,
                                                               max_p_value_chi_sq)
                max_depth_per_fold.append(curr_max_depth)
                num_nodes_per_fold.append(self.get_root_node().get_num_nodes())
                num_valid_attributes_in_root.append(
                    sum(self._root_node.valid_nominal_attribute))
                for curr_index, validation_sample_index in enumerate(validation_sample_indices):
                    classifications[validation_sample_index] = curr_classifications[curr_index]
                    classified_with_unkown_value_array[validation_sample_index] = (
                        curr_classified_with_unkown_value_array[curr_index])
                    unkown_value_attrib_index_array[validation_sample_index] = (
                        curr_unkown_value_attrib_index_array[curr_index])
                num_correct_classifications += curr_num_correct_classifications
                num_correct_classifications_wo_unkown += curr_num_correct_classifications_wo_unkown
                total_cost += curr_total_cost
                total_cost_wo_unkown += curr_total_cost_wo_unkown
                num_unkown += curr_num_unkown
                num_correct_trivial_classifications += round(
                    len(validation_sample_indices) *
                    (self.get_trivial_accuracy(validation_sample_indices) / 100.0))

                fold_count += 1
                time_taken_prunning_per_fold.append(curr_time_taken_prunning)
                num_nodes_prunned_per_fold.append(curr_num_nodes_prunned)

                if print_tree:
                    print()
                    print('-' * 50)
                    print('Fold:', fold_count)
                    self.save_tree()

        return  (classifications,
                 num_correct_classifications,
                 num_correct_classifications_wo_unkown,
                 total_cost,
                 total_cost_wo_unkown,
                 classified_with_unkown_value_array,
                 num_unkown,
                 unkown_value_attrib_index_array,
                 time_taken_prunning_per_fold,
                 num_nodes_prunned_per_fold,
                 max_depth_per_fold,
                 num_nodes_per_fold,
                 num_valid_attributes_in_root,
                 100.0 * num_correct_trivial_classifications / dataset.num_samples)

    def test(self, test_sample_indices):
        """Tests the (already trained) tree over samples from the same dataset as the
            training set. If the tree hasn't been trained, the program will exit.

        Args:
            test_sample_indices (:obj:'list' of 'int'): list of the test set indices for samples
                from the same dataset used for training.

        Returns:
            A tuple containing, in order:
                - a list of predicted class for each test sample;
                - the number of correct classifications;
                - the number of correct classifications done without test samples with unkown
                    values (that is, values that are unkown at a TreeNode -- they are classified as
                    the most common class at that node);
                - the total cost of the classification errors (when errors costs are uniform, this
                    is equal to the total number of test samples minus the number of correct
                    classifications);
                -  the total cost of the classification errors without considering test
                    samples with unkown values;
                - a list of booleans indicating if the i-th test sample was classified with an
                    unkown value;
                - the number of test samples classified with unkown values;
                - list where the i-th entry has the attribute index used for classification of the
                    i-th sample when an unkown value occurred.
        """

        if self._root_node is None:
            print('Decision tree must be trained before testing.')
            sys.exit(1)
        return self._classify_samples(self._dataset.samples,
                                      self._dataset.sample_class,
                                      self._dataset.sample_costs,
                                      test_sample_indices,
                                      self._dataset.sample_index_to_key)

    def test_from_csv(self, test_dataset_csv_filepath, key_attrib_index, class_attrib_index,
                      split_char, missing_value_string):
        """Tests the (already trained) tree using all samples from a given csv file. If the tree
        hasn't been trained, the program will exit.

        Args:
            test_dataset_csv_filepath (str): path to the test dataset.
            key_attrib_index (int): column index of the samples' keys on the csv.
            class_attrib_index (int): column index of the samples' classes on the csv.
            split_char (str): char used to split columns in the csv.
            missing_value_string (str): string used to indicate that a sample does not have a value.

        Returns:
            A tuple containing, in order:
                - a list of predicted class for each test sample;
                - the number of correct classifications;
                - the number of correct classifications done without test samples with unkown
                    values (that is, values that are unkown at a TreeNode -- they are classified as
                    the most common class at that node);
                - the total cost of the classification errors (when errors costs are uniform, this
                    is equal to the total number of test samples minus the number of correct
                    classifications);
                -  the total cost of the classification errors without considering test
                    samples with unkown values;
                - a list of booleans indicating if the i-th test sample was classified with an
                    unkown value;
                - the number of test samples classified with unkown values;
                - list where the i-th entry has the attribute index used for classification of the
                    i-th sample when an unkown value occurred.
        """

        if self._root_node is None or self._dataset is None:
            print('Decision tree must be trained before testing.')
            sys.exit(1)
        self._dataset.load_test_set_from_csv(test_dataset_csv_filepath,
                                             key_attrib_index,
                                             class_attrib_index,
                                             split_char,
                                             missing_value_string)
        return self._classify_samples(self._dataset.test_samples,
                                      self._dataset.test_sample_class,
                                      self._dataset.test_sample_costs,
                                      list(range(len(self._dataset.test_sample_index_to_key))),
                                      self._dataset.test_sample_index_to_key)

    def save_tree(self, filepath=None):
        """Saves the tree information: nodes, attributes used to split each one, values to each
        side, etc.

        Args:
            filepath (str, optional): file in which to save the tree. If `None`, prints to stdout.
                Defaults to `None`.
        """
        # TESTED!
        # Saves in a txt file or something similar
        def _aux_print_nominal_string(attrib_name, string_values, curr_depth):
            # TESTED!
            ret_string = '|' * curr_depth
            ret_string += '{} in {}:'.format(attrib_name, string_values)
            return ret_string

        def _aux_print_numeric_string(attrib_name, mid_point, inequality, curr_depth):
            # TESTED!
            ret_string = '|' * curr_depth
            ret_string += '{} {} {}:'.format(attrib_name, inequality, mid_point)
            return ret_string

        def _aux_print_split(file_object, tree_node, curr_depth):
            # TESTED!
            if tree_node.is_leaf:
                leaf_class_string = self._dataset.class_int_to_name[
                    tree_node.most_common_int_class]
                string_leaf = '|' * curr_depth + 'CLASS: ' + leaf_class_string
                print(string_leaf, file=file_object)
            else:
                attrib_index = tree_node.node_split.separation_attrib_index
                attrib_name = self._dataset.attrib_names[attrib_index]
                mid_point = tree_node.node_split.mid_point
                if mid_point is not None:
                    # <= mid_point, go left
                    print(_aux_print_numeric_string(attrib_name, mid_point, '<=', curr_depth),
                          file=file_object)
                    _aux_print_split(file_object, tree_node.nodes[0], curr_depth + 1)
                    # > mid_point, go right
                    print(_aux_print_numeric_string(attrib_name, mid_point, '>', curr_depth),
                          file=file_object)
                    _aux_print_split(file_object, tree_node.nodes[1], curr_depth + 1)
                else:
                    for split_values, child_node in zip(tree_node.node_split.splits_values,
                                                        tree_node.nodes):
                        curr_string_values = sorted(
                            [self._dataset.attrib_int_to_value[attrib_index][int_value]
                             for int_value in split_values])
                        print(_aux_print_nominal_string(attrib_name,
                                                        curr_string_values,
                                                        curr_depth),
                              file=file_object)
                        _aux_print_split(file_object, child_node, curr_depth + 1)

        if filepath is None:
            _aux_print_split(sys.stdout, self._root_node, curr_depth=0)
        else:
            with open(filepath, 'w') as tree_output_file:
                _aux_print_split(tree_output_file, self._root_node, curr_depth=0)


class TreeNode(object):
    """Contains information of a certain node of a decision tree.

        It has information about the samples used during training at this node and also about it's
    NodeSplit.

    Attributes:
        is_leaf (bool): indicates if the current TreeNode is a tree leaf.
        max_depth_remaining (int): maximum depth that the subtree rooted at the current TreeNode can
            still grow. If zero, the current TreeNode will be a leaf.
        node_split (NodeSplit): Data structure containing information about which attribute and
            split values were obtained in this TreeNode with a certain criterion. Also contains the
            criterion value. It is None if the current TreeNode is a leaf.
        nodes (:obj:'list' of 'TreeNode'): list containing every child TreeNode from the current
            TreeNode.
        contingency_tables (:obj:'list' of 'tuple' of 'list' of 'np.array'): contains a list where
            the i-th entry is a tuple containing two pieces of information of the i-th attribute:
            the contingency table for that attribute (value index is row, class index is column) and
            a list of number of times each value is attained in the training set (i-th entry is the
            number of times a sample has value i in this attribute and training dataset). Used by
            many criteria when calculating the optimal split. Note that, for invalid attributes, the
            entry is a tuple with empty lists ([], []).
        dataset (Dataset): dataset containing the training samples.
        valid_samples_indices (:obj:'list' of 'int'): contains the indices of the valid training
            samples.
        valid_nominal_attribute (:obj:'list' of 'bool'): list where the i-th entry indicates wether
            the i-th attribute from the dataset is valid and nominal or not.
        valid_numeric_attribute (:obj:'list' of 'bool'): list where the i-th entry indicates wether
            the i-th attribute from the dataset is valid and numeric or not.
        num_valid_samples (int): number of training samples in this TreeNode.
        class_index_num_samples (:obj:'list' of 'int'): list where the i-th entry indicates the
            number of samples having class i.
        most_common_int_class (int): index of the most frequent class.
        number_non_empty_classes (int): number of classes having no sample in this TreeNode.
    """
    def __init__(self, dataset, valid_samples_indices, valid_nominal_attribute,
                 valid_numeric_attribute, max_depth_remaining, min_samples_per_node,
                 use_stop_conditions=False, max_p_value_chi_sq=0.1):
        """Initializes a TreeNode instance with the given arguments.

        Args:
            dataset (Dataset): dataset of samples used for training/split generation.
            valid_samples_indices (:obj:'list' of 'int'): indices of samples that should be used for
                training at this node.
            valid_nominal_attribute (:obj:'list' of 'bool'): the i-th entry informs wether the i-th
                attribute is a valid nominal one.
            valid_numeric_attribute (:obj:'list' of 'bool'): the i-th entry informs wether the i-th
                attribute is a valid numeric one.
            max_depth_remaining (int): maximum depth that the subtree rooted at this node can have.
                If zero, this node will be a leaf.
            min_samples_per_node (int): minimum number of samples that must be present in order to
                try to create a subtree rooted at this node. If less than this, this node will be a
                leaf.
            use_stop_conditions (bool, optional): informs wether we should use prunning techniques
                to avoid using attributes with small number of samples (and, thus, avoiding
                statistical anomalies). An attribute will be considered invalid if it contains less
                than `MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE` samples in the second most frequent
                value (this way at least two values have this number of samples) or if a chi-square
                test, applied on the attributes' contingency table has a p-value greater or equal to
                `max_p_value_chi_sq`. When an attribute is considered invalid for the number of
                samples in the second most frequent value, it will be considered invalid in every
                child node of the current TreeNode. If it was considered invalid because of the
                chi-square test, it can be considered valid in a descendant node. Note that numeric
                attributes are never tested in this way. Defaults to `False`.
            max_p_value_chi_sq (float, optional): is the maximum p-value allowed for an attribute to
                be accepted when doing chi-square tests (that is, when `use_stop_conditions` is
                `True`). A p-value of 1.0 is equal to 100%. Defaults to `0.1`.
        """
        self._use_stop_conditions = use_stop_conditions
        self._max_p_value_chi_sq = max_p_value_chi_sq

        self.max_depth_remaining = max_depth_remaining
        self._min_samples_per_node = min_samples_per_node

        self.is_leaf = True
        self.node_split = None
        self.nodes = []
        self.contingency_tables = None

        self.dataset = dataset
        self.valid_samples_indices = valid_samples_indices
        # Note that self.valid_nominal_attribute might be different from
        # self.dataset.valid_nominal_attribute when use_stop_conditions == True.
        self.valid_nominal_attribute = valid_nominal_attribute
        self.valid_numeric_attribute = valid_numeric_attribute

        self.num_valid_samples = len(valid_samples_indices)
        self.class_index_num_samples = [0] * dataset.num_classes
        self.most_common_int_class = None
        self.number_non_empty_classes = 0

        # Fill self.class_index_num_samples
        for sample_index in valid_samples_indices:
            self.class_index_num_samples[
                dataset.sample_class[sample_index]] += 1

        self.most_common_int_class = self.class_index_num_samples.index(
            max(self.class_index_num_samples))
        self._calculate_contingency_tables()


    def _calculate_contingency_tables(self):
        self.contingency_tables = [] # vector of pairs (attrib_contingency_table,
                                     #                  attrib_values_num_samples)
        for (attrib_index,
             is_valid_nominal_attribute) in enumerate(self.valid_nominal_attribute):
            if not is_valid_nominal_attribute:
                self.contingency_tables.append(([], []))
                continue

            attrib_num_values = len(self.dataset.attrib_int_to_value[attrib_index])
            curr_contingency_table = np.zeros((attrib_num_values, self.dataset.num_classes),
                                              dtype=int)
            curr_values_num_samples = np.zeros((attrib_num_values), dtype=int)

            for sample_index in self.valid_samples_indices:
                curr_sample_value = self.dataset.samples[sample_index][attrib_index]
                curr_sample_class = self.dataset.sample_class[sample_index]
                curr_contingency_table[curr_sample_value][curr_sample_class] += 1
                curr_values_num_samples[curr_sample_value] += 1

            self.contingency_tables.append((curr_contingency_table, curr_values_num_samples))

    def _is_attribute_valid(self, attrib_index, min_allowed_in_two_largest):
        """Returns a pair of booleans indicating:
            - wether the current attribute has more than `min_allowed_in_two_largest` samples in
                second most frequent value;
            - wether the condition above is `True` AND its chi-square test's p-value is smaller than
                `self._max_p_value_chi_sq`.
        """
        def _get_chi_square_test_p_value(contingency_table, values_num_samples):
            classes_seen = set()
            for value in range(contingency_table.shape[0]):
                for sample_class, num_samples in enumerate(contingency_table[value]):
                    if num_samples > 0 and sample_class not in classes_seen:
                        classes_seen.add(sample_class)
            num_classes = len(classes_seen)
            if num_classes == 1:
                return 0.0

            num_values = sum(num_samples > 0 for num_samples in values_num_samples)
            num_samples = sum(num_samples for num_samples in values_num_samples)
            curr_chi_square_value = 0.0
            for value, value_num_sample in enumerate(values_num_samples):
                if value_num_sample == 0:
                    continue
                for class_index in classes_seen:
                    expected_value_class = (
                        values_num_samples[value] * self.class_index_num_samples[class_index]
                        / num_samples)
                    diff = contingency_table[value][class_index] - expected_value_class
                    curr_chi_square_value += diff * (diff / expected_value_class)
            return 1. - chi2.cdf(x=curr_chi_square_value, df=((num_classes - 1) * (num_values - 1)))


        values_num_samples = self.contingency_tables[attrib_index][1]
        largest = 0
        second_largest = 0
        for num_samples in values_num_samples:
            if num_samples > largest:
                second_largest = largest
                largest = num_samples
            elif num_samples > second_largest:
                second_largest = num_samples
        if second_largest < min_allowed_in_two_largest:
            return (False, False)
        chi_square_test_p_value = _get_chi_square_test_p_value(
            self.contingency_tables[attrib_index][0],
            self.contingency_tables[attrib_index][1])
        return (True, chi_square_test_p_value < self._max_p_value_chi_sq)

    def create_subtree(self, criterion):
        """Given the splitting criterion, creates a tree rooted at the current TreeNode.

        Args:
            criterion (Criterion): splitting criterion used to create the tree recursively.
        """

        def _get_values_to_split(splits_values):
            values_to_split = {}
            for split_index, split_values in enumerate(splits_values):
                for value in split_values:
                    values_to_split[value] = split_index
            return values_to_split

        def _get_splits_samples_indices(num_splits, separation_attrib_index, values_to_split,
                                        valid_samples_indices, samples):
            splits_samples_indices = [[] for _ in range(num_splits)]
            for sample_index in valid_samples_indices:
                sample_value_in_split_attrib = samples[sample_index][separation_attrib_index]
                try:
                    splits_samples_indices[values_to_split[
                        sample_value_in_split_attrib]].append(sample_index)
                except KeyError:
                    print('Should not get here. Sample {} has value {} at attribute # {}, '
                          'but this value is unknown to the decision tree.'.format(
                              sample_index,
                              sample_value_in_split_attrib,
                              separation_attrib_index))
                    sys.exit(1)
            return splits_samples_indices

        def _get_numeric_splits_samples_indices(separation_attrib_index, mid_point,
                                                valid_samples_indices, samples):
            splits_samples_indices = [[], []]
            for sample_index in valid_samples_indices:
                sample_value_in_split_attrib = samples[sample_index][separation_attrib_index]
                if sample_value_in_split_attrib <= mid_point:
                    splits_samples_indices[0].append(sample_index)
                else:
                    splits_samples_indices[1].append(sample_index)
            return splits_samples_indices

        def _has_multiple_nominal_values(values_num_samples):
            return sum(num_samples > 0 for num_samples in values_num_samples) > 1

        def _has_multiple_numeric_values(valid_samples_indices, sample, attrib_index):
            values_seen = set()
            for sample_index in valid_samples_indices:
                sample_value = sample[sample_index][attrib_index]
                if sample_value not in values_seen:
                    if values_seen:
                        return True
                    values_seen.add(sample_value)
            return False

        def _has_enough_samples_in_second_largest_class(class_index_num_samples,
                                                        most_common_int_class):
            second_largest = max(num_samples
                                 for class_index, num_samples in enumerate(class_index_num_samples)
                                 if class_index != most_common_int_class)
            return second_largest >= MIN_SAMPLES_SECOND_LARGEST_CLASS


        # Is it time to stop growing subtrees?
        if (self.max_depth_remaining <= 0
                or self.num_valid_samples < self._min_samples_per_node
                or self.number_non_empty_classes == 1
                or (USE_MIN_SAMPLES_SECOND_LARGEST_CLASS
                    and not _has_enough_samples_in_second_largest_class(
                        self.class_index_num_samples,
                        self.most_common_int_class))):
            return None

        # If a valid attribute has only one value, it should be marked as invalid from this node on.
        num_valid_nominal_attributes = 0
        for attrib_index in range(len(self.valid_nominal_attribute)):
            if not self.valid_nominal_attribute[attrib_index]:
                continue
            if not _has_multiple_nominal_values(self.contingency_tables[attrib_index][1]):
                self.valid_nominal_attribute[attrib_index] = False
            else:
                num_valid_nominal_attributes += 1

        num_valid_numeric_attributes = 0
        for attrib_index in range(len(self.valid_numeric_attribute)):
            if not self.valid_numeric_attribute[attrib_index]:
                continue
            if not _has_multiple_numeric_values(self.valid_samples_indices,
                                                self.dataset.samples,
                                                attrib_index):
                self.valid_numeric_attribute[attrib_index] = False
            else:
                num_valid_numeric_attributes += 1

        # If there are no valid attributes, this node should be a leaf.
        if not num_valid_nominal_attributes and not num_valid_numeric_attributes:
            return None

        if self._use_stop_conditions:
            num_valid_attributes = sum(self.dataset.valid_numeric_attribute)
            # Attributes which are valid (`True`) in `new_valid_nominal_attribute` and invalid
            # (`False`) in `new_valid_nominal_attribute_incl_chi_sq_test` should not be used to
            # split at this node, but could be used to split in descendant nodes.
            new_valid_nominal_attribute = self.valid_nominal_attribute[:]
            new_valid_nominal_attribute_incl_chi_sq_test = self.valid_nominal_attribute[:]
            for (attrib_index,
                 is_valid_nominal_attribute) in enumerate(self.valid_nominal_attribute):
                if is_valid_nominal_attribute:
                    (is_valid_num_samples,
                     is_valid_chi_sq_and_num_samples) = (self._is_attribute_valid(
                         attrib_index,
                         min_allowed_in_two_largest=MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE))
                    if is_valid_chi_sq_and_num_samples:
                        num_valid_attributes += 1
                    elif is_valid_num_samples:
                        new_valid_nominal_attribute_incl_chi_sq_test[attrib_index] = False
                    else:
                        new_valid_nominal_attribute[attrib_index] = False
                        new_valid_nominal_attribute_incl_chi_sq_test[attrib_index] = False
            self.valid_nominal_attribute = new_valid_nominal_attribute_incl_chi_sq_test
            if num_valid_attributes == 0:
                return None

        # Get best split. Note that self is the current TreeNode.
        (separation_attrib_index,
         splits_values,
         criterion_value) = criterion.select_best_attribute_and_split(self)

        if math.isinf(criterion_value):
            # Stop condition when there is no valid attribute with more than one value (then
            # criterion_value is default, which is +- inf).
            return None

        if self.dataset.valid_numeric_attribute[separation_attrib_index]:
            # NUMERIC ATTRIBUTE
            last_left_value = list(splits_values[0])[0]
            first_right_value = list(splits_values[1])[0]
            mid_point = 0.5 * (last_left_value + first_right_value)
            splits_samples_indices = _get_numeric_splits_samples_indices(
                separation_attrib_index,
                mid_point,
                self.valid_samples_indices,
                self.dataset.samples)
            # Save this node's split information.
            self.node_split = NodeSplit(separation_attrib_index,
                                        None,
                                        None,
                                        criterion_value,
                                        mid_point=mid_point)

        else:
            # NOMINAL ATTRIBUTE

            # Calculate a list containing the inverse information of splits_values: here, given a
            # value, we want to know to which split it belongs
            values_to_split = _get_values_to_split(splits_values)

            splits_samples_indices = _get_splits_samples_indices(len(splits_values),
                                                                 separation_attrib_index,
                                                                 values_to_split,
                                                                 self.valid_samples_indices,
                                                                 self.dataset.samples)
            # Save this node's split information.
            self.node_split = NodeSplit(separation_attrib_index,
                                        splits_values,
                                        values_to_split,
                                        criterion_value)

        # Create subtrees
        self.is_leaf = False
        if self._use_stop_conditions:
            # Any attribute that has enough samples in the second most frequent value could pass the
            # chi-square test in a descendant node, thus we don't send the information of chi-square
            # test to child nodes.
            self.valid_nominal_attribute = new_valid_nominal_attribute
        for curr_split_samples_indices in splits_samples_indices:
            self.nodes.append(TreeNode(self.dataset,
                                       curr_split_samples_indices,
                                       self.valid_nominal_attribute,
                                       self.valid_numeric_attribute,
                                       self.max_depth_remaining - 1,
                                       self._min_samples_per_node,
                                       self._use_stop_conditions,
                                       self._max_p_value_chi_sq))
            self.nodes[-1].create_subtree(criterion)

    def get_most_popular_subtree(self):
        """Returns the number of samples in the most popular subtree. If it is leaf, returns
        `self.num_valid_samples`."""
        if self.is_leaf:
            return self.num_valid_samples
        return max(subtree.num_valid_samples for subtree in self.nodes)

    def prune_trivial_subtrees(self):
        """Applies prunning to an already trained tree. Returns the number of prunned nodes.

        If a TreeNode is trivial, that is, every leaf in its subtree has the same
        `most_common_int_class`, then the current TreeNode becomes a leaf with this class, deleting
        every child node in this process. It is applied recursively.
        """
        num_prunned = 0
        if not self.is_leaf:
            children_classes = set()
            num_trivial_children = 0
            for child_node in self.nodes:
                num_prunned += child_node.prune_trivial_subtrees()
                if child_node.is_leaf:
                    num_trivial_children += 1
                    children_classes.add(child_node.most_common_int_class)
            if num_trivial_children == len(self.nodes) and len(children_classes) == 1:
                self.is_leaf = True
                num_prunned += num_trivial_children
                self.nodes = []
        return num_prunned

    def get_num_nodes(self):
        """Returns the number of nodes in the tree rooted at the current TreeNode (counting includes
        leaves and the current TreeNode)."""
        num_nodes = 1
        for child_node in self.nodes:
            num_nodes += child_node.get_num_nodes()
        return num_nodes

    def get_max_depth(self):
        """Returns the maximum depth of the tree rooted at the current TreeNode. If the current node
        is a leaf, it will return zero."""
        if self.is_leaf:
            return 0
        ret = 0
        for child_node in self.nodes:
            max_child_depth = child_node.get_max_depth()
            if max_child_depth > ret:
                ret = max_child_depth
        return ret + 1


class NodeSplit(object):
    """Data structure containing information about the best split found on a node.

    Used for debugging and for the classification of test samples.

    Attributes:
        separation_attrib_index (int): Index of the attribute used for splitting.
        splits_values (:obj:'list' of 'set' of 'int'): list containing a set of attribute values for
            each TreeNode child. Binary splits have two sets (left and right split values), multiway
            splits may have many more.
        values_to_split (:obj:'dict' of 'int'): reversed index for `splits_values`. Given a value,
            it returns the index of the split that this value belongs to.
        criterion_value (float): criterion value for this split.
        mid_point (float): cut point for numeric splits. Will be the average between the largest
            value on the left split and the smallest value on the right split.
    """
    def __init__(self, separation_attrib_index, splits_values, values_to_split, criterion_value,
                 mid_point=None):
        """Initializes a TreeNode instance with the given arguments.

        Args:
            separation_attrib_index (int): Index of the attribute used for splitting.
            splits_values (:obj:'list' of 'set' of 'int'): list containing a set of attribute values
                for each TreeNode child. Binary splits have two sets (left and right split values),
                multiway splits may have many more.
            values_to_split (:obj:'dict' of 'int'): reversed index for `splits_values`. Given a
                value, it returns the index of the split that this value belongs to.
            criterion_value (float): optimal criterion value obtained for this TreeNode.
            mid_point (float, optional): cut point for numeric splits. Will be the average between
                the largest value on the left split and the smallest value on the right split. Not
                used for splits that use nominal attributes. Defaults to `None`.
        """
        self.separation_attrib_index = separation_attrib_index
        self.splits_values = splits_values
        self.values_to_split = values_to_split
        self.criterion_value = criterion_value

        self.mid_point = mid_point
