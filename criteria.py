#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing all criteria available for tests.


"""

import abc
import itertools
import math
import random
import timeit

import chol
import cvxpy as cvx
import numpy as np


class Criterion(object):
    __metaclass__ = abc.ABCMeta

    name = ''

    @classmethod
    @abc.abstractmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the criterion.
        """
        # returns (separation_attrib_index, splits_values, criterion_value)
        pass


#################################################################################################
#################################################################################################
###                                                                                           ###
###                                       TWOING                                              ###
###                                                                                           ###
#################################################################################################
#################################################################################################

class Twoing(Criterion):
    name = 'Twoing'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the Twoing criterion.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.

        Returns:
            A tuple cointaining, in order:
                - the index of the accepted attribute;
                - a list of sets, each containing the values that should go to that split/subtree.
                -  Split value according to the criterion. If no attribute has a valid split, this
                value should be `float('-inf')`.
        """
        best_splits_per_attrib = []
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(zip(tree_node.valid_nominal_attribute,
                                                         tree_node.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                best_total_gini_gain = float('-inf')
                best_left_values = set()
                best_right_values = set()
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index][1])
                for (set_left_classes,
                     set_right_classes) in cls._generate_twoing(tree_node.class_index_num_samples):
                    (twoing_contingency_table,
                     superclass_index_num_samples) = cls._get_twoing_contingency_table(
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.contingency_tables[attrib_index][1],
                         set_left_classes,
                         set_right_classes)
                    original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
                                                              superclass_index_num_samples)
                    (curr_gini_gain,
                     left_values,
                     right_values) = cls._two_class_trick(
                         original_gini,
                         superclass_index_num_samples,
                         values_seen,
                         tree_node.contingency_tables[attrib_index][1],
                         twoing_contingency_table,
                         len(tree_node.valid_samples_indices))
                    if curr_gini_gain > best_total_gini_gain:
                        best_total_gini_gain = curr_gini_gain
                        best_left_values = left_values
                        best_right_values = right_values
                best_splits_per_attrib.append((attrib_index,
                                               [best_left_values, best_right_values],
                                               best_total_gini_gain))
            elif is_valid_numeric_attrib:
                values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                  tree_node.dataset.samples,
                                                                  tree_node.dataset.sample_class,
                                                                  attrib_index)
                values_and_classes.sort()
                (best_twoing,
                 last_left_value,
                 first_right_value) = cls._twoing_for_numeric(
                     values_and_classes,
                     tree_node.dataset.num_classes)
                best_splits_per_attrib.append((attrib_index,
                                               best_twoing,
                                               [{last_left_value}, {first_right_value}]))

        best_attribute_and_split = (None, [], float('-inf'))
        for curr_attrib_split in best_splits_per_attrib:
            if curr_attrib_split[2] > best_attribute_and_split[2]:
                best_attribute_and_split = curr_attrib_split
        return best_attribute_and_split

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
        return values_and_classes

    @staticmethod
    def _generate_twoing(class_index_num_samples):
        # We only need to look at superclasses of up to (len(class_index_num_samples)/2 + 1)
        # elements because of symmetry! The subsets we are not choosing are complements of the ones
        # chosen.
        non_empty_classes = set([])
        for class_index, class_num_samples in enumerate(class_index_num_samples):
            if class_num_samples > 0:
                non_empty_classes.add(class_index)
        number_non_empty_classes = len(non_empty_classes)

        for left_classes in itertools.chain.from_iterable(
                itertools.combinations(non_empty_classes, size_left_superclass)
                for size_left_superclass in range(1, number_non_empty_classes // 2 + 1)):
            set_left_classes = set(left_classes)
            set_right_classes = non_empty_classes - set_left_classes
            if not set_left_classes or not set_right_classes:
                # A valid split must have at least one sample in each side
                continue
            yield set_left_classes, set_right_classes

    @staticmethod
    def _get_twoing_contingency_table(contingency_table, values_num_samples, set_left_classes,
                                      set_right_classes):
        twoing_contingency_table = np.zeros((contingency_table.shape[0], 2), dtype=float)
        superclass_index_num_samples = [0, 0]
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            for class_index in set_left_classes:
                superclass_index_num_samples[0] += contingency_table[value][class_index]
                twoing_contingency_table[value][0] += contingency_table[value][class_index]
            for class_index in set_right_classes:
                superclass_index_num_samples[1] += contingency_table[value][class_index]
                twoing_contingency_table[value][1] += contingency_table[value][class_index]
        return twoing_contingency_table, superclass_index_num_samples

    @classmethod
    def _twoing_for_numeric(cls, sorted_values_and_classes, num_classes):
        last_left_value = sorted_values_and_classes[0][0]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1

        class_num_left = [0] * num_classes
        class_num_left[sorted_values_and_classes[0][1]] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        best_twoing = float('-inf')
        best_last_left_value = None
        best_first_right_value = None

        for first_right_index in range(1, len(sorted_values_and_classes)):
            first_right_value = sorted_values_and_classes[first_right_index][0]
            if first_right_value != last_left_value:
                twoing_value = cls._get_twoing_value(class_num_left,
                                                     class_num_right,
                                                     num_left_samples,
                                                     num_right_samples)
                if twoing_value > best_twoing:
                    best_twoing = twoing_value
                    best_last_left_value = last_left_value
                    best_first_right_value = first_right_value

                last_left_value = first_right_value

            num_left_samples += 1
            num_right_samples -= 1
            first_right_class = sorted_values_and_classes[first_right_index][1]
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1
        return (best_twoing, best_last_left_value, best_first_right_value)

    @staticmethod
    def _get_twoing_value(class_num_left, class_num_right, num_left_samples,
                          num_right_samples):
        sum_dif = 0.0
        for left_num, right_num in zip(class_num_left, class_num_right):
            class_num_tot = class_num_left + class_num_right
            if class_num_tot == 0:
                continue
            sum_dif += abs(left_num / num_left_samples - right_num / num_right_samples)

        num_total_samples = num_left_samples + num_right_samples
        frequency_left = num_left_samples / num_total_samples
        frequency_right = num_right_samples / num_total_samples

        twoing_value = (frequency_left * frequency_right / 4.0) * sum_dif ** 2
        return twoing_value

    @staticmethod
    def _two_class_trick(original_gini, class_index_num_samples, values_seen, values_num_samples,
                         contingency_table, num_total_valid_samples):
        # TESTED!
        def _get_non_empty_class_indices(class_index_num_samples):
            # TESTED!
            first_non_empty_class = None
            second_non_empty_class = None
            for class_index, class_num_samples in enumerate(class_index_num_samples):
                if class_num_samples > 0:
                    if first_non_empty_class is None:
                        first_non_empty_class = class_index
                    else:
                        second_non_empty_class = class_index
                        break
            return first_non_empty_class, second_non_empty_class

        def _calculate_value_class_ratio(values_seen, values_num_samples, contingency_table,
                                         non_empty_class_indices):
            # TESTED!
            value_number_ratio = [] # [(value, number_on_second_class, ratio_on_second_class)]
            second_class_index = non_empty_class_indices[1]
            for curr_value in values_seen:
                number_second_non_empty = contingency_table[curr_value][second_class_index]
                value_number_ratio.append((curr_value,
                                           number_second_non_empty,
                                           number_second_non_empty/values_num_samples[curr_value]))
            value_number_ratio.sort(key=lambda tup: tup[2])
            return value_number_ratio

        def _calculate_children_gini_index(num_left_first, num_left_second, num_right_first,
                                           num_right_second, num_left_samples, num_right_samples):
            # TESTED!
            if num_left_samples != 0:
                left_first_class_freq_ratio = float(num_left_first)/float(num_left_samples)
                left_second_class_freq_ratio = float(num_left_second)/float(num_left_samples)
                left_split_gini_index = (1.0
                                         - left_first_class_freq_ratio**2
                                         - left_second_class_freq_ratio**2)
            else:
                # We can set left_split_gini_index to any value here, since it will be multiplied
                # by zero in curr_children_gini_index
                left_split_gini_index = 1.0

            if num_right_samples != 0:
                right_first_class_freq_ratio = float(num_right_first)/float(num_right_samples)
                right_second_class_freq_ratio = float(num_right_second)/float(num_right_samples)
                right_split_gini_index = (1.0
                                          - right_first_class_freq_ratio**2
                                          - right_second_class_freq_ratio**2)
            else:
                # We can set right_split_gini_index to any value here, since it will be multiplied
                # by zero in curr_children_gini_index
                right_split_gini_index = 1.0

            curr_children_gini_index = ((num_left_samples * left_split_gini_index
                                         + num_right_samples * right_split_gini_index)
                                        / (num_left_samples + num_right_samples))
            return curr_children_gini_index

        # We only need to sort values by the percentage of samples in second non-empty class with
        # this value. The best split will be given by choosing an index to split this list of
        # values in two.
        (first_non_empty_class,
         second_non_empty_class) = _get_non_empty_class_indices(class_index_num_samples)
        if first_non_empty_class is None or second_non_empty_class is None:
            return (float('-inf'), {0}, set())

        value_number_ratio = _calculate_value_class_ratio(values_seen,
                                                          values_num_samples,
                                                          contingency_table,
                                                          (first_non_empty_class,
                                                           second_non_empty_class))

        best_split_total_gini_gain = float('-inf')
        best_last_left_index = 0

        num_left_first = 0
        num_left_second = 0
        num_left_samples = 0
        num_right_first = class_index_num_samples[first_non_empty_class]
        num_right_second = class_index_num_samples[second_non_empty_class]
        num_right_samples = num_total_valid_samples

        for last_left_index, (last_left_value, last_left_num_second, _) in enumerate(
                value_number_ratio[:-1]):
            num_samples_last_left_value = values_num_samples[last_left_value]
            # num_samples_last_left_value > 0 always, since the values without samples were not
            # added to the values_seen when created by cls._generate_value_to_index

            last_left_num_first = num_samples_last_left_value - last_left_num_second

            num_left_samples += num_samples_last_left_value
            num_left_first += last_left_num_first
            num_left_second += last_left_num_second
            num_right_samples -= num_samples_last_left_value
            num_right_first -= last_left_num_first
            num_right_second -= last_left_num_second

            curr_children_gini_index = _calculate_children_gini_index(num_left_first,
                                                                      num_left_second,
                                                                      num_right_first,
                                                                      num_right_second,
                                                                      num_left_samples,
                                                                      num_right_samples)
            curr_gini_gain = original_gini - curr_children_gini_index
            if curr_gini_gain > best_split_total_gini_gain:
                best_split_total_gini_gain = curr_gini_gain
                best_last_left_index = last_left_index

        # Let's get the values and split the indices corresponding to the best split found.
        set_left_values = set([tup[0] for tup in value_number_ratio[:best_last_left_index + 1]])
        set_right_values = set(values_seen) - set_left_values

        return (best_split_total_gini_gain, set_left_values, set_right_values)

    @staticmethod
    def _calculate_gini_index(side_num, class_num_side):
        gini_index = 1.0
        for curr_class_num_side in class_num_side:
            if curr_class_num_side > 0:
                gini_index -= (curr_class_num_side/side_num)**2
        return gini_index

    @classmethod
    def _calculate_children_gini_index(cls, left_num, class_num_left, right_num, class_num_right):
        left_split_gini_index = cls._calculate_gini_index(left_num, class_num_left)
        right_split_gini_index = cls._calculate_gini_index(right_num, class_num_right)
        children_gini_index = ((left_num * left_split_gini_index
                                + right_num * right_split_gini_index)
                               / (left_num + right_num))
        return children_gini_index



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                      GW SQUARED GINI                                      ###
###                                                                                           ###
#################################################################################################
#################################################################################################

class GWSquaredGini(Criterion):
    name = 'GW Squared Gini'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the GWSG criterion.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.

        Returns:
            A tuple cointaining, in order:
                - the index of the accepted attribute;
                - a list of sets, each containing the values that should go to that split/subtree.
                -  Split value according to the criterion. If no attribute has a valid split, this
                value should be `float('-inf')`.
        """
        best_splits_per_attrib = []
        for attrib_index, is_valid_nominal_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_nominal_attrib:
                (new_to_orig_value_int,
                 new_contingency_table,
                 new_values_num_seen) = cls._remove_empty_values(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])

                (curr_cut_value,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(new_to_orig_value_int,
                                                              new_contingency_table,
                                                              new_values_num_seen)
                best_splits_per_attrib.append((attrib_index,
                                               [left_int_values, right_int_values],
                                               curr_cut_value))

        best_attribute_and_split = (None, [], float('-inf'))
        for curr_attrib_split in best_splits_per_attrib:
            if curr_attrib_split[2] > best_attribute_and_split[2]:
                best_attribute_and_split = curr_attrib_split
        return best_attribute_and_split


    @staticmethod
    def _remove_empty_values(contingency_table, values_num_samples):
        # Define conversion from original values to new values
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for orig_value, curr_num_samples in enumerate(values_num_samples):
            if curr_num_samples > 0:
                orig_to_new_value_int[orig_value] = len(new_to_orig_value_int)
                new_to_orig_value_int.append(orig_value)

        # Generate the new contingency tables
        new_contingency_table = np.zeros((len(new_to_orig_value_int), contingency_table.shape[1]),
                                         dtype=int)
        new_value_num_seen = np.zeros((len(new_to_orig_value_int)), dtype=int)
        for orig_value, curr_num_samples in enumerate(values_num_samples):
            if curr_num_samples > 0:
                curr_new_value = orig_to_new_value_int[orig_value]
                new_value_num_seen[curr_new_value] = curr_num_samples
                np.copyto(dst=new_contingency_table[curr_new_value, :],
                          src=contingency_table[orig_value, :])

        return (new_to_orig_value_int,
                new_contingency_table,
                new_value_num_seen)

    @classmethod
    def _generate_best_split(cls, new_to_orig_value_int, new_contingency_table,
                             new_values_num_seen):
        def _init_values_weights(new_contingency_table, new_values_num_seen):
            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut).
            weights = np.zeros((new_values_num_seen.shape[0], new_values_num_seen.shape[0]),
                               dtype=np.float64)
            for value_index_i in range(new_values_num_seen.shape[0]):
                for value_index_j in range(new_values_num_seen.shape[0]):
                    if value_index_i == value_index_j:
                        continue
                    for class_index in range(new_contingency_table.shape[1]):
                        num_elems_value_j_diff_class = (
                            new_values_num_seen[value_index_j]
                            - new_contingency_table[value_index_j, class_index])
                        weights[value_index_i, value_index_j] += (
                            new_contingency_table[value_index_i, class_index]
                            * num_elems_value_j_diff_class)
            return weights

        weights = _init_values_weights(new_contingency_table, new_values_num_seen)
        frac_split_cholesky = cls._solve_max_cut(weights)
        left_new_values, right_new_values = cls._generate_random_partition(frac_split_cholesky)

        left_orig_values, right_orig_values = cls._get_split_in_orig_values(new_to_orig_value_int,
                                                                            left_new_values,
                                                                            right_new_values)
        cut_val = cls._calculate_split_value(left_new_values, right_new_values, weights)
        return cut_val, left_orig_values, right_orig_values


    @staticmethod
    def _solve_max_cut(weights):
        def _solve_sdp(weights):
            # See Max Cut approximation given by Goemans and Williamson, 1995.
            var = cvx.Semidef(weights.shape[0])
            obj = cvx.Minimize(0.25 * cvx.trace(weights.T * var))

            constraints = [var == var.T, var >> 0]
            for i in range(weights.shape[0]):
                constraints.append(var[i, i] == 1)

            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.SCS, verbose=False)
            return var.value

        fractional_split_squared = _solve_sdp(weights)
        # The solution should already be symmetric, but let's just make sure the approximations
        # didn't change that.
        sym_fractional_split_squared = 0.5 * (fractional_split_squared
                                              + fractional_split_squared.T)
        # We are interested in the Cholesky decomposition of the above matrix to finally choose a
        # random partition based on it. Detail: the above matrix may be singular, so not every
        # method works.
        temp_P, temp_L, _ = chol.chol_higham(sym_fractional_split_squared)

        # Note that temp_L.T is upper triangular, but
        # frac_split_cholesky = np.dot(temp.L.T, temp_P)
        # is not necessarily upper triangular. Since we are only interested in decomposing
        # sym_fractional_split_squared = np.dot(frac_split_cholesky.T, frac_split_cholesky)
        # that is not a problem.
        return np.dot(temp_L.T, temp_P)

    @staticmethod
    def _generate_random_partition(frac_split_cholesky):
        random_vector = np.random.randn(frac_split_cholesky.shape[1])
        values_split = np.zeros((frac_split_cholesky.shape[1]), dtype=np.float64)
        for column_index in range(frac_split_cholesky.shape[1]):
            column = frac_split_cholesky[:, column_index]
            values_split[column_index] = np.dot(random_vector, column)
        values_split_bool = np.apply_along_axis(lambda x: x > 0.0, axis=0, arr=values_split)

        left_new_values = set()
        right_new_values = set()
        for new_value in range(frac_split_cholesky.shape[1]):
            if values_split_bool[new_value]:
                left_new_values.add(new_value)
            else:
                right_new_values.add(new_value)
        return left_new_values, right_new_values

    @staticmethod
    def _get_split_in_orig_values(new_to_orig_value_int, left_new_values, right_new_values):
        # Let's get the original values on each side of this partition
        left_orig_values = set(new_to_orig_value_int[left_new_value]
                               for left_new_value in left_new_values)
        right_orig_values = set(new_to_orig_value_int[right_new_value]
                                for right_new_value in right_new_values)
        return left_orig_values, right_orig_values

    @staticmethod
    def _calculate_split_value(left_new_values, right_new_values, weights):
        cut_val = 0.0
        for value_left, value_right in itertools.product(left_new_values, right_new_values):
            cut_val += weights[value_left, value_right]
        return cut_val



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                     LS Squared Gini                                       ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class LSSquaredGini(Criterion):
    name = 'LS Squared Gini'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!
        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)

        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(
                  zip(tree_node.valid_nominal_attribute,
                      tree_node.dataset.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                start_time = timeit.default_timer()
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

            elif is_valid_numeric_attrib:
                start_time = timeit.default_timer()
                values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                  tree_node.dataset.samples,
                                                                  tree_node.dataset.sample_class,
                                                                  attrib_index)
                values_and_classes.sort()
                (best_cut_value,
                 last_left_value,
                 first_right_value) = cls._best_cut_for_numeric(
                     values_and_classes,
                     tree_node.dataset.num_classes)
                ret.append((attrib_index,
                            best_cut_value,
                            [{last_left_value}, {first_right_value}],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(
                  zip(tree_node.valid_nominal_attribute,
                      tree_node.dataset.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class)

                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values

            elif is_valid_numeric_attrib:
                values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                  tree_node.dataset.samples,
                                                                  tree_node.dataset.sample_class,
                                                                  attrib_index)
                values_and_classes.sort()
                (curr_gain,
                 last_left_value,
                 first_right_value) = cls._best_cut_for_numeric(
                     values_and_classes,
                     tree_node.dataset.num_classes)
                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = {last_left_value}
                    best_split_right_values = {first_right_value}

        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_attrib_valid_values(attrib_index, samples, valid_samples_indices):
        #TESTED!
        seen_values = set([])
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for sample_index in valid_samples_indices:
            value_int = samples[sample_index][attrib_index]
            if value_int not in seen_values:
                orig_to_new_value_int[value_int] = len(seen_values)
                new_to_orig_value_int.append(value_int)
                seen_values.add(value_int)
        return len(seen_values), orig_to_new_value_int, new_to_orig_value_int

    @staticmethod
    def _calculate_diff(valid_samples_indices, sample_costs):
        #TESTED!
        def _max_min_diff(list_of_values):
            max_val = list_of_values[0]
            min_val = max_val
            for value in list_of_values[1:]:
                if value > max_val:
                    max_val = value
                elif value < min_val:
                    min_val = value
            return abs(max_val - min_val)

        diff_keys = []
        diff_values = []
        for sample_index in valid_samples_indices:
            curr_costs = sample_costs[sample_index]
            diff_values.append(_max_min_diff(curr_costs))
            diff_keys.append(sample_index)
        diff_keys_values = sorted(list(zip(diff_keys, diff_values)),
                                  key=lambda key_value: key_value[1])
        diff_keys, diff_values = zip(*diff_keys_values)
        return diff_keys, diff_values

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
        return values_and_classes

    @classmethod
    def _generate_best_split(cls, attrib_index, num_classes, attrib_num_valid_values,
                             orig_to_new_value_int, new_to_orig_value_int, valid_samples_indices,
                             samples, sample_class):
        #TESTED!
        def _init_values_histograms(attrib_index, num_classes, attrib_num_valid_values,
                                    valid_samples_indices):
            #TESTED!
            values_histogram = np.zeros((attrib_num_valid_values), dtype=np.int64)
            values_histogram_with_classes = np.zeros((attrib_num_valid_values, num_classes),
                                                     dtype=np.int64)
            for sample_index in valid_samples_indices:
                orig_value = samples[sample_index][attrib_index]
                new_value = orig_to_new_value_int[orig_value]
                values_histogram[new_value] += 1
                values_histogram_with_classes[new_value][sample_class[sample_index]] += 1
            return values_histogram, values_histogram_with_classes

        def _init_values_weights(num_classes, values_histogram, values_histogram_with_classes):
            # TESTED!
            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((values_histogram.shape[0], values_histogram.shape[0]),
                               dtype=np.float64)
            for value_index_i in range(values_histogram.shape[0]):
                for value_index_j in range(values_histogram.shape[0]):
                    if value_index_i == value_index_j:
                        continue
                    for class_index in range(num_classes):
                        num_elems_value_j_diff_class = (
                            values_histogram[value_index_j]
                            - values_histogram_with_classes[value_index_j, class_index])
                        weights[value_index_i, value_index_j] += (
                            values_histogram_with_classes[value_index_i, class_index]
                            * num_elems_value_j_diff_class)
            return weights

        (values_histogram,
         values_histogram_with_classes) = _init_values_histograms(attrib_index,
                                                                  num_classes,
                                                                  attrib_num_valid_values,
                                                                  valid_samples_indices)
        weights = _init_values_weights(num_classes,
                                       values_histogram,
                                       values_histogram_with_classes)
        values_seen = set(range(attrib_num_valid_values))

        gain, new_left_values, new_right_values = cls._generate_initial_partition(values_seen,
                                                                                  weights)
        # Look for a better solution locally
        (gain_switched,
         new_left_values_switched,
         new_right_values_switched) = cls._switch_while_increase(gain,
                                                                 new_left_values,
                                                                 new_right_values,
                                                                 weights)
        if gain_switched > gain:
            gain = gain_switched
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values_switched)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values_switched)
        else:
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values)
        return gain, values_histogram, left_values, right_values

    @classmethod
    def _generate_initial_partition(cls, values_seen, weights):
        set_left_values = set()
        set_right_values = set()
        cut_val = 0.0

        # calculating initial solution for max cut
        for value in values_seen:
            if not set_left_values and not set_right_values:
                set_left_values.add(value)
                continue
            sum_with_left = sum(weights[value][left_value] for left_value in set_left_values)
            sum_with_right = sum(weights[value][right_value] for right_value in set_right_values)
            if sum_with_left >= sum_with_right:
                set_right_values.add(value)
                cut_val += sum_with_left
            else:
                set_left_values.add(value)
                cut_val += sum_with_right
        return cut_val, set_left_values, set_right_values

    @classmethod
    def _switch_while_increase(cls, cut_val, set_left_values, set_right_values, weights):
        curr_cut_val = cut_val
        values_seen = set_left_values | set_right_values

        improvement = True
        while improvement:
            improvement = False
            for value in values_seen:
                new_cut_val = cls._calculate_split_gain_for_single_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          value,
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value in set_left_values:
                        set_left_values.remove(value)
                        set_right_values.add(value)
                    else:
                        set_left_values.add(value)
                        set_right_values.remove(value)
                    improvement = True
                    break
            if improvement:
                continue
            for value1, value2 in itertools.combinations(values_seen, 2):
                if ((value1 in set_left_values and value2 in set_left_values) or
                        (value1 in set_right_values and value2 in set_right_values)):
                    continue
                new_cut_val = cls._calculate_split_gain_for_double_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          (value1, value2),
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value1 in set_left_values:
                        set_left_values.remove(value1)
                        set_right_values.add(value1)
                        set_right_values.remove(value2)
                        set_left_values.add(value2)
                    else:
                        set_left_values.remove(value2)
                        set_right_values.add(value2)
                        set_right_values.remove(value1)
                        set_left_values.add(value1)
                    improvement = True
                    break

        return curr_cut_val, set_left_values, set_right_values

    @staticmethod
    def _calculate_split_gain_for_single_switch(curr_gain, new_left_values, new_right_values,
                                                new_value_to_change_sides, weights):
        new_gain = curr_gain
        if new_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
            for value in new_right_values:
                new_gain -= weights[value][new_value_to_change_sides]
        else:
            for value in new_left_values:
                new_gain -= weights[value][new_value_to_change_sides]
            for value in new_right_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain_for_double_switch(curr_gain, new_left_values, new_right_values,
                                                new_values_to_change_sides, weights):
        assert len(new_values_to_change_sides) == 2
        new_gain = curr_gain
        first_value_to_change_sides = new_values_to_change_sides[0]
        second_value_to_change_sides = new_values_to_change_sides[1]

        if first_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
        else:
            for value in new_left_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
        return new_gain

    @classmethod
    def _best_cut_for_numeric(cls, sorted_values_and_classes, num_classes):
        last_left_value = sorted_values_and_classes[0][0]
        last_left_class = sorted_values_and_classes[0][1]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1

        class_num_left = [0] * num_classes
        class_num_left[last_left_class] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        best_cut_value = float('-inf')
        best_last_left_value = None
        best_first_right_value = None

        curr_cut_value = num_right_samples - class_num_right[last_left_class]

        for first_right_index in range(1, len(sorted_values_and_classes)):
            first_right_value = sorted_values_and_classes[first_right_index][0]
            first_right_class = sorted_values_and_classes[first_right_index][1]

            if first_right_value != last_left_value and curr_cut_value > best_cut_value:
                best_cut_value = curr_cut_value
                best_last_left_value = last_left_value
                best_first_right_value = first_right_value

            curr_cut_value -= num_left_samples - class_num_left[first_right_class]
            num_left_samples += 1
            num_right_samples -= 1
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1
            curr_cut_value += num_right_samples - class_num_right[first_right_class]
            if first_right_value != last_left_value:
                last_left_value = first_right_value

        return (best_cut_value, best_last_left_value, best_first_right_value)





#################################################################################################
#################################################################################################
###                                                                                           ###
###                                       GW CHI SQUARE                                       ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class GWChiSquare(Criterion):
    name = 'GW Chi Square'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!
        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class)

                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_attrib_valid_values(attrib_index, samples, valid_samples_indices):
        #TESTED!
        seen_values = set([])
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for sample_index in valid_samples_indices:
            value_int = samples[sample_index][attrib_index]
            if value_int not in seen_values:
                orig_to_new_value_int[value_int] = len(seen_values)
                new_to_orig_value_int.append(value_int)
                seen_values.add(value_int)
        return len(seen_values), orig_to_new_value_int, new_to_orig_value_int

    @staticmethod
    def _calculate_diff(valid_samples_indices, sample_costs):
        #TESTED!
        def _max_min_diff(list_of_values):
            max_val = list_of_values[0]
            min_val = max_val
            for value in list_of_values[1:]:
                if value > max_val:
                    max_val = value
                elif value < min_val:
                    min_val = value
            return abs(max_val - min_val)

        diff_keys = []
        diff_values = []
        for sample_index in valid_samples_indices:
            curr_costs = sample_costs[sample_index]
            diff_values.append(_max_min_diff(curr_costs))
            diff_keys.append(sample_index)
        diff_keys_values = sorted(list(zip(diff_keys, diff_values)),
                                  key=lambda key_value: key_value[1])
        diff_keys, diff_values = zip(*diff_keys_values)
        return diff_keys, diff_values

    @classmethod
    def _generate_best_split(cls, attrib_index, num_classes, attrib_num_valid_values,
                             orig_to_new_value_int, new_to_orig_value_int, valid_samples_indices,
                             samples, sample_class):
        #TESTED!
        def _init_values_histograms(attrib_index, num_classes, attrib_num_valid_values,
                                    valid_samples_indices):
            #TESTED!
            values_histogram = np.zeros((attrib_num_valid_values), dtype=np.int64)
            values_histogram_with_classes = np.zeros((attrib_num_valid_values, num_classes),
                                                     dtype=np.int64)
            for sample_index in valid_samples_indices:
                orig_value = samples[sample_index][attrib_index]
                new_value = orig_to_new_value_int[orig_value]
                values_histogram[new_value] += 1
                values_histogram_with_classes[new_value][sample_class[sample_index]] += 1
            return values_histogram, values_histogram_with_classes

        def _init_values_weights(num_classes, values_histogram, values_histogram_with_classes):
            # TESTED!

            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((values_histogram.shape[0], values_histogram.shape[0]),
                               dtype=np.float64)
            num_values = sum(num_samples > 0 for num_samples in values_histogram)
            for value_index_i in range(values_histogram.shape[0]):
                if values_histogram[value_index_i] == 0:
                    continue
                for value_index_j in range(values_histogram.shape[0]):
                    if value_index_i == value_index_j or values_histogram[value_index_j] == 0:
                        continue

                    num_samples_both_values = (values_histogram[value_index_i]
                                               + values_histogram[value_index_j])
                    for class_index in range(num_classes):
                        num_samples_both_values_this_class = (
                            values_histogram_with_classes[value_index_i, class_index]
                            + values_histogram_with_classes[value_index_j, class_index])
                        if num_samples_both_values_this_class == 0:
                            continue
                        expected_value_index_i_class = (
                            values_histogram[value_index_i] * num_samples_both_values_this_class
                            / num_samples_both_values)
                        expected_value_index_j_class = (
                            values_histogram[value_index_j] * num_samples_both_values_this_class
                            / num_samples_both_values)
                        diff_index_i = (
                            values_histogram_with_classes[value_index_i, class_index]
                            - expected_value_index_i_class)
                        diff_index_j = (
                            values_histogram_with_classes[value_index_j, class_index]
                            - expected_value_index_j_class)

                        edge_weight_curr_class = (
                            diff_index_i * (diff_index_i / expected_value_index_i_class)
                            + diff_index_j * (diff_index_j / expected_value_index_j_class))

                        weights[value_index_i, value_index_j] += edge_weight_curr_class

                        if edge_weight_curr_class < 0.0:
                            print('='*90)
                            print('VALOR DE CHI SQUARE DA ARESTA {}{} COM CLASSE {}: {} < 0'.format(
                                value_index_i,
                                value_index_j,
                                class_index,
                                edge_weight_curr_class))
                            print('='*90)
                    if num_values > 2:
                        weights[value_index_i, value_index_j] /= (num_values - 1.)

            return weights


        (values_histogram,
         values_histogram_with_classes) = _init_values_histograms(attrib_index,
                                                                  num_classes,
                                                                  attrib_num_valid_values,
                                                                  valid_samples_indices)
        weights = _init_values_weights(num_classes,
                                       values_histogram,
                                       values_histogram_with_classes)

        frac_split_cholesky = cls._solve_max_cut(attrib_num_valid_values, weights)
        (left_values,
         right_values,
         new_left_values,
         new_right_values) = cls._generate_random_partition(frac_split_cholesky,
                                                            new_to_orig_value_int)
        gain = cls._calculate_split_gain(new_left_values,
                                         new_right_values,
                                         weights)
        return gain, values_histogram, left_values, right_values

    @staticmethod
    def _calculate_split_gain(new_left_values, new_right_values, weights):
        gain = 0.0
        for value_left, value_right in itertools.product(new_left_values, new_right_values):
            gain += weights[value_left, value_right]
        return gain

    @staticmethod
    def _solve_max_cut(attrib_num_valid_values, weights):
        #TESTED!
        def _solve_sdp(size, weights):
            #TESTED!
            # See Max Cut approximate given by Goemans and Williamson, 1995.
            var = cvx.Semidef(size)
            obj = cvx.Minimize(0.25 * cvx.trace(weights.T * var))

            constraints = [var == var.T, var >> 0]
            for i in range(size):
                constraints.append(var[i, i] == 1)

            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.SCS, verbose=False)
            return var.value

        fractional_split_squared = _solve_sdp(attrib_num_valid_values, weights)
        # The solution should be symmetric, but let's just make sure the approximations didn't
        # change that.
        sym_fractional_split_squared = 0.5 * (fractional_split_squared
                                              + fractional_split_squared.T)
        # We are interested in the Cholesky decomposition of the above matrix to finally choose a
        # random partition based on it. Detail: the above matrix may be singular, so not every
        # method works.
        temp_P, temp_L, _ = chol.chol_higham(sym_fractional_split_squared)

        # Note that temp_L.T is upper triangular, but
        # frac_split_cholesky = np.dot(temp.L.T, temp_P)
        # is not necessarily upper triangular. Since we are only interested in decomposing
        # sym_fractional_split_squared = np.dot(frac_split_cholesky.T, frac_split_cholesky)
        # that is not a problem.
        return np.dot(temp_L.T, temp_P)

    @staticmethod
    def _generate_random_partition(frac_split_cholesky,
                                   new_to_orig_value_int):
        #TESTED!
        random_vector = np.random.randn(frac_split_cholesky.shape[1])
        values_split = np.zeros((frac_split_cholesky.shape[1]), dtype=np.float64)
        for column_index in range(frac_split_cholesky.shape[1]):
            column = frac_split_cholesky[:, column_index]
            values_split[column_index] = np.dot(random_vector, column)
        values_split_bool = np.apply_along_axis(lambda x: x > 0.0, axis=0, arr=values_split)
        # Let's get the values on each side of this partition
        left_values = set()
        right_values = set()
        new_left_values = set()
        new_right_values = set()
        for new_value in range(frac_split_cholesky.shape[1]):
            if values_split_bool[new_value]:
                left_values.add(new_to_orig_value_int[new_value])
                new_left_values.add(new_value)
            else:
                right_values.add(new_to_orig_value_int[new_value])
                new_right_values.add(new_value)

        return left_values, right_values, new_left_values, new_right_values




#################################################################################################
#################################################################################################
###                                                                                           ###
###                                       LS Chi Square                                       ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class LSChiSquare(Criterion):
    name = 'LS Chi Square'

    @classmethod
    def evaluate_all_attributes_2(cls, tree_node, num_tests, num_fails_allowed):
        # contains (attrib_index, gain, split_values, p_value, time_taken)
        best_split_per_attrib = []

        num_valid_attrib = 0
        smaller_contingency_tables = {}
        criterion_start_time = timeit.default_timer()

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (orig_to_new_value_int,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples) = cls._get_smaller_contingency_table(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])
                if len(new_to_orig_value_int) <= 1:
                    print("Attribute {} ({}) is valid but only has {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(new_to_orig_value_int)))
                    continue
                smaller_contingency_tables[attrib_index] = (orig_to_new_value_int,
                                                            new_to_orig_value_int,
                                                            smaller_contingency_table,
                                                            smaller_values_num_samples)
                num_valid_attrib += 1
                (curr_gain,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     new_to_orig_value_int,
                     smaller_contingency_table,
                     smaller_values_num_samples)
                best_split_per_attrib.append((attrib_index,
                                              curr_gain,
                                              [left_int_values, right_int_values],
                                              None,
                                              timeit.default_timer() - start_time))
        criterion_total_time = timeit.default_timer() - criterion_start_time


        ordered_start_time = timeit.default_timer()
        preference_rank_full = sorted(best_split_per_attrib, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []#(1,
        #                     float('-inf'),
        #                     None,
        #                     None,
        #                     None)]
        # bad_attrib_indices = {3, 5, 6, 10, 11, 12, 13, 17, 18, 20, 21, 22, 25, 56, 57, 52, 55, 59,
        #                       60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 476, 478}
        # preference_rank_mailcode_first = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
            # if pref_elem[0] in bad_attrib_indices:
            #     preference_rank_mailcode_first.append(pref_elem)

        # preference_rank_mailcode_first = sorted(preference_rank_mailcode_first,
        #                                         key=lambda x: x[1])
        # for pref_elem in preference_rank:
        #     if pref_elem[0] in bad_attrib_indices:
        #         continue
        #     preference_rank_mailcode_first.append(pref_elem)

        tests_done_ordered = 0
        accepted_attribute_ordered = None
        ordered_accepted_rank = None
        for (rank_index,
             (attrib_index, best_gain, _, _, _)) in enumerate(preference_rank):
            if math.isinf(best_gain):
                continue
            (_,
             new_to_orig_value_int,
             smaller_contingency_table,
             smaller_values_num_samples) = smaller_contingency_tables[attrib_index]
            (should_accept,
             num_tests_needed) = cls.accept_attribute(
                 best_gain,
                 num_tests,
                 len(tree_node.valid_samples_indices),
                 num_fails_allowed,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples)
            if not should_accept:
                tests_done_ordered += num_tests_needed
            else:
                accepted_attribute_ordered = tree_node.dataset.attrib_names[attrib_index]
                print('Accepted attribute:', accepted_attribute_ordered)
                ordered_accepted_rank = rank_index + 1
                tests_done_ordered += num_tests
                break
        ordered_total_time = timeit.default_timer() - ordered_start_time


        rev_start_time = timeit.default_timer()
        # Reversed ordered
        rev_preference_rank = reversed(preference_rank)

        tests_done_rev = 0
        accepted_attribute_rev = None
        for (attrib_index, best_gain, _, _, _) in rev_preference_rank:
            if math.isinf(best_gain):
                continue
            (_,
             new_to_orig_value_int,
             smaller_contingency_table,
             smaller_values_num_samples) = smaller_contingency_tables[attrib_index]
            (should_accept,
             num_tests_needed) = cls.accept_attribute(
                 best_gain,
                 num_tests,
                 len(tree_node.valid_samples_indices),
                 num_fails_allowed,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples)
            if not should_accept:
                tests_done_rev += num_tests_needed
            else:
                accepted_attribute_rev = tree_node.dataset.attrib_names[attrib_index]
                print('Accepted attribute:', accepted_attribute_rev)
                tests_done_rev += num_tests
                break
        rev_total_time = timeit.default_timer() - rev_start_time


        # Order splits randomly
        random_start_time = timeit.default_timer()
        random_order_rank = preference_rank[:]
        random.shuffle(random_order_rank)

        tests_done_random_order = 0
        accepted_attribute_random = None
        for (attrib_index, best_gain, _, _, _) in random_order_rank:
            if math.isinf(best_gain):
                continue
            (_,
             new_to_orig_value_int,
             smaller_contingency_table,
             smaller_values_num_samples) = smaller_contingency_tables[attrib_index]
            (should_accept,
             num_tests_needed) = cls.accept_attribute(
                 best_gain,
                 num_tests,
                 len(tree_node.valid_samples_indices),
                 num_fails_allowed,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples)
            if not should_accept:
                tests_done_random_order += num_tests_needed
            else:
                accepted_attribute_random = tree_node.dataset.attrib_names[attrib_index]
                print('Accepted attribute:', accepted_attribute_random)
                tests_done_random_order += num_tests
                break
        random_total_time = timeit.default_timer() - random_start_time

        if ordered_accepted_rank is None:
            return (tests_done_ordered,
                    accepted_attribute_ordered,
                    tests_done_rev,
                    accepted_attribute_rev,
                    tests_done_random_order,
                    accepted_attribute_random,
                    num_valid_attrib,
                    ordered_accepted_rank,
                    criterion_total_time,
                    ordered_total_time,
                    rev_total_time,
                    random_total_time,
                    preference_rank[0],
                    None)
        return (tests_done_ordered,
                accepted_attribute_ordered,
                tests_done_rev,
                accepted_attribute_rev,
                tests_done_random_order,
                accepted_attribute_random,
                num_valid_attrib,
                ordered_accepted_rank,
                criterion_total_time,
                ordered_total_time,
                rev_total_time,
                random_total_time,
                preference_rank[0],
                preference_rank[ordered_accepted_rank - 1])

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!
        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)

        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(
                  zip(tree_node.valid_nominal_attribute,
                      tree_node.dataset.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                start_time = timeit.default_timer()
                (_,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples) = cls._get_smaller_contingency_table(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])
                if len(new_to_orig_value_int) <= 1:
                    print("Attribute {} ({}) is valid but only has {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(new_to_orig_value_int)))
                    continue
                (curr_gain,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     new_to_orig_value_int,
                     smaller_contingency_table,
                     smaller_values_num_samples)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

            elif is_valid_numeric_attrib:
                start_time = timeit.default_timer()
                values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                  tree_node.dataset.samples,
                                                                  tree_node.dataset.sample_class,
                                                                  attrib_index)
                values_and_classes.sort()
                (best_cut_value,
                 last_left_value,
                 first_right_value) = cls._best_cut_for_numeric_chi_square(
                     values_and_classes,
                     tree_node.dataset.num_classes,
                     tree_node.class_index_num_samples)
                ret.append((attrib_index,
                            best_cut_value,
                            [{last_left_value}, {first_right_value}],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])

        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(
                  zip(tree_node.valid_nominal_attribute,
                      tree_node.dataset.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                (_,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples) = cls._get_smaller_contingency_table(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])
                if len(new_to_orig_value_int) <= 1:
                    print("Attribute {} ({}) is valid but only has {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(new_to_orig_value_int)))
                    continue

                (curr_gain,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     new_to_orig_value_int,
                     smaller_contingency_table,
                     smaller_values_num_samples)
                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values

            elif is_valid_numeric_attrib:
                values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                  tree_node.dataset.samples,
                                                                  tree_node.dataset.sample_class,
                                                                  attrib_index)
                values_and_classes.sort()
                (curr_gain,
                 last_left_value,
                 first_right_value) = cls._best_cut_for_numeric_chi_square(
                     values_and_classes,
                     tree_node.dataset.num_classes,
                     tree_node.class_index_num_samples)

                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = {last_left_value}
                    best_split_right_values = {first_right_value}


        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
        return values_and_classes

    @classmethod
    def _best_cut_for_numeric_chi_square(cls, sorted_values_and_classes, num_classes,
                                         class_index_num_samples):
        last_left_value = sorted_values_and_classes[0][0]
        last_left_class = sorted_values_and_classes[0][1]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1
        num_samples = len(sorted_values_and_classes)

        class_num_left = [0] * num_classes
        class_num_left[last_left_class] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        best_cut_value = float('-inf')
        best_last_left_value = None
        best_first_right_value = None

        for first_right_index in range(1, len(sorted_values_and_classes)):
            first_right_value = sorted_values_and_classes[first_right_index][0]
            first_right_class = sorted_values_and_classes[first_right_index][1]

            curr_cut_value = 0.0
            for class_index in range(num_classes):
                if class_index_num_samples[class_index] != 0:
                    expected_value_left_class = (
                        num_left_samples * class_index_num_samples[class_index] / num_samples)
                    diff_left = class_num_left[class_index] - expected_value_left_class
                    curr_cut_value += diff_left * (diff_left / expected_value_left_class)

                    expected_value_right_class = (
                        num_right_samples * class_index_num_samples[class_index] / num_samples)
                    diff_right = class_num_right[class_index] - expected_value_right_class
                    curr_cut_value += diff_right * (diff_right / expected_value_right_class)

            if first_right_value != last_left_value and curr_cut_value > best_cut_value:
                best_cut_value = curr_cut_value
                best_last_left_value = last_left_value
                best_first_right_value = first_right_value
                last_left_value = first_right_value

            num_left_samples += 1
            num_right_samples -= 1
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1

        return (best_cut_value, best_last_left_value, best_first_right_value)


    @staticmethod
    def _get_smaller_contingency_table(contingency_table, values_num_samples):
        seen_values = set(value
                          for value, num_samples in enumerate(values_num_samples)
                          if num_samples > 0)
        num_classes = contingency_table.shape[1]
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        smaller_contingency_table = np.zeros((len(seen_values), num_classes),
                                             dtype=float)
        smaller_values_num_samples = np.zeros((len(seen_values)), dtype=float)
        for orig_value, num_samples in enumerate(values_num_samples):
            if num_samples == 0:
                continue
            new_value = len(new_to_orig_value_int)
            smaller_values_num_samples[new_value] = num_samples
            orig_to_new_value_int[orig_value] = new_value
            new_to_orig_value_int.append(orig_value)
            smaller_values_num_samples[new_value] = num_samples
            for curr_class, num_samples_curr_class in enumerate(contingency_table[orig_value, :]):
                if num_samples_curr_class > 0:
                    smaller_contingency_table[new_value, curr_class] = num_samples_curr_class

        return (orig_to_new_value_int,
                new_to_orig_value_int,
                smaller_contingency_table,
                smaller_values_num_samples)

    @classmethod
    def _generate_best_split(cls, new_to_orig_value_int, smaller_contingency_table,
                             smaller_values_num_samples):

        def _init_values_weights(contingency_table, values_num_samples):
            # TESTED!
            def _get_chi_square_value(contingency_table_row_1, contingency_table_row_2,
                                      num_samples_first_value, num_samples_second_value):
                ret = 0.0
                num_samples_both_values = num_samples_first_value + num_samples_second_value
                num_classes = contingency_table_row_1.shape[0]
                curr_values_num_classes = 0
                for class_index in range(num_classes):
                    num_samples_both_values_this_class = (
                        contingency_table_row_1[class_index]
                        + contingency_table_row_2[class_index])
                    if num_samples_both_values_this_class == 0:
                        continue
                    curr_values_num_classes += 1

                    expected_value_first_class = (
                        num_samples_first_value * num_samples_both_values_this_class
                        / num_samples_both_values)

                    expected_value_second_class = (
                        num_samples_second_value * num_samples_both_values_this_class
                        / num_samples_both_values)

                    diff_first = (
                        contingency_table_row_1[class_index]
                        - expected_value_first_class)
                    diff_second = (
                        contingency_table_row_2[class_index]
                        - expected_value_second_class)

                    chi_sq_curr_class = (
                        diff_first * (diff_first / expected_value_first_class)
                        + diff_second * (diff_second / expected_value_second_class))

                    ret += chi_sq_curr_class

                    if chi_sq_curr_class < 0.0:
                        print('='*90)
                        print('VALOR DE CHI SQUARE DE UMA ARESTA COM CLASSE {}: {} < 0'.format(
                            class_index,
                            chi_sq_curr_class))
                        print('='*90)
                return ret, curr_values_num_classes

            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((smaller_values_num_samples.shape[0], values_num_samples.shape[0]),
                               dtype=np.float64)
            num_values = len(smaller_values_num_samples)
            for value_index_i in range(values_num_samples.shape[0]):
                if values_num_samples[value_index_i] == 0:
                    continue
                for value_index_j in range(values_num_samples.shape[0]):
                    if value_index_i >= value_index_j or values_num_samples[value_index_j] == 0:
                        continue
                    (edge_weight,
                     curr_values_num_classes) = _get_chi_square_value(
                         contingency_table[value_index_i, :],
                         contingency_table[value_index_j, :],
                         values_num_samples[value_index_i],
                         values_num_samples[value_index_j])

                    if curr_values_num_classes == 1:
                        weights[value_index_i, value_index_j] = 0.0
                    else:
                        weights[value_index_i, value_index_j] = edge_weight
                        weights[value_index_j, value_index_i] = edge_weight

                    if num_values > 2:
                        weights[value_index_i, value_index_j] /= (num_values - 1.)
                        weights[value_index_j, value_index_i] = (
                            weights[value_index_i, value_index_j])
            return weights


        weights = _init_values_weights(smaller_contingency_table, smaller_values_num_samples)
        values_seen = set(range(len(new_to_orig_value_int)))

        gain, new_left_values, new_right_values = cls._generate_initial_partition(values_seen,
                                                                                  weights)
        # Look for a better solution locally
        (gain_switched,
         new_left_values_switched,
         new_right_values_switched) = cls._switch_while_increase(gain,
                                                                 new_left_values,
                                                                 new_right_values,
                                                                 weights)
        if gain_switched > gain:
            gain = gain_switched
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values_switched)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values_switched)
        else:
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values)
        return gain, left_values, right_values

    @classmethod
    def _generate_initial_partition(cls, values_seen, weights):
        set_left_values = set()
        set_right_values = set()
        cut_val = 0.0

        # calculating initial solution for max cut
        for value in values_seen:
            if not set_left_values and not set_right_values:
                set_left_values.add(value)
                continue
            sum_with_left = sum(weights[value][left_value] for left_value in set_left_values)
            sum_with_right = sum(weights[value][right_value] for right_value in set_right_values)
            if sum_with_left >= sum_with_right:
                set_right_values.add(value)
                cut_val += sum_with_left
            else:
                set_left_values.add(value)
                cut_val += sum_with_right
        return cut_val, set_left_values, set_right_values

    @classmethod
    def _switch_while_increase(cls, cut_val, set_left_values, set_right_values, weights):
        curr_cut_val = cut_val
        values_seen = set_left_values | set_right_values

        improvement = True
        while improvement:
            improvement = False
            for value in values_seen:
                new_cut_val = cls._calculate_split_gain_for_single_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          value,
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value in set_left_values:
                        set_left_values.remove(value)
                        set_right_values.add(value)
                    else:
                        set_left_values.add(value)
                        set_right_values.remove(value)
                    improvement = True
                    break
            if improvement:
                continue
            for value1, value2 in itertools.combinations(values_seen, 2):
                if ((value1 in set_left_values and value2 in set_left_values) or
                        (value1 in set_right_values and value2 in set_right_values)):
                    continue
                new_cut_val = cls._calculate_split_gain_for_double_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          (value1, value2),
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value1 in set_left_values:
                        set_left_values.remove(value1)
                        set_right_values.add(value1)
                        set_right_values.remove(value2)
                        set_left_values.add(value2)
                    else:
                        set_left_values.remove(value2)
                        set_right_values.add(value2)
                        set_right_values.remove(value1)
                        set_left_values.add(value1)
                    improvement = True
                    break

        return curr_cut_val, set_left_values, set_right_values

    @staticmethod
    def _calculate_split_gain_for_single_switch(curr_gain, new_left_values, new_right_values,
                                                new_value_to_change_sides, weights):
        new_gain = curr_gain
        if new_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
            for value in new_right_values:
                new_gain -= weights[value][new_value_to_change_sides]
        else:
            for value in new_left_values:
                new_gain -= weights[value][new_value_to_change_sides]
            for value in new_right_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain_for_double_switch(curr_gain, new_left_values, new_right_values,
                                                new_values_to_change_sides, weights):
        assert len(new_values_to_change_sides) == 2
        new_gain = curr_gain
        first_value_to_change_sides = new_values_to_change_sides[0]
        second_value_to_change_sides = new_values_to_change_sides[1]

        if first_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
        else:
            for value in new_left_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain(new_left_values, new_right_values, weights):
        gain = 0.0
        for value_left, value_right in itertools.product(new_left_values, new_right_values):
            gain += weights[value_left, value_right]
        return gain

    @staticmethod
    def get_classes_dist(contingency_table, values_num_samples, num_valid_samples):
        num_classes = contingency_table.shape[1]
        classes_dist = [0] * num_classes
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            for class_index, num_samples in enumerate(contingency_table[value, :]):
                if num_samples > 0:
                    classes_dist[class_index] += num_samples
        for class_index in range(num_classes):
            classes_dist[class_index] /= float(num_valid_samples)
        return classes_dist

    @staticmethod
    def generate_random_contingency_table(classes_dist, num_valid_samples, values_num_samples):
        # TESTED!
        random_classes = np.random.choice(len(classes_dist),
                                          num_valid_samples,
                                          replace=True,
                                          p=classes_dist)
        random_contingency_table = np.zeros((values_num_samples.shape[0], len(classes_dist)),
                                            dtype=float)
        samples_done = 0
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples > 0:
                for class_index in random_classes[samples_done: samples_done + value_num_samples]:
                    random_contingency_table[value, class_index] += 1
                samples_done += value_num_samples
        return random_contingency_table

    @classmethod
    def accept_attribute(cls, real_gain, num_tests, num_valid_samples, num_fails_allowed,
                         new_to_orig_value_int, smaller_contingency_table,
                         smaller_values_num_samples):
        classes_dist = cls.get_classes_dist(smaller_contingency_table,
                                            smaller_values_num_samples,
                                            num_valid_samples)
        num_fails_seen = 0
        for test_number in range(1, num_tests + 1):
            random_contingency_table = cls.generate_random_contingency_table(
                classes_dist,
                num_valid_samples,
                smaller_values_num_samples)
            (gain,
             _,
             _) = cls._generate_best_split(new_to_orig_value_int,
                                           random_contingency_table,
                                           smaller_values_num_samples)
            if gain > real_gain:
                num_fails_seen += 1
                if num_fails_seen > num_fails_allowed:
                    return False, test_number
            if num_tests - test_number <= num_fails_allowed - num_fails_seen:
                return True, None
        return True, None
