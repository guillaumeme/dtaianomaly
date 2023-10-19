#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dtaianomaly.evaluation.affiliation_util.generics import (
        infer_Trange,
        has_point_anomalies, 
        _len_wo_nan, 
        _sum_wo_nan)
from dtaianomaly.evaluation.affiliation_util._affiliation_zone import (
        get_all_E_gt_func, 
        affiliation_partition)
from dtaianomaly.evaluation.affiliation_util._single_ground_truth_event import (
        affiliation_precision_proba,
        affiliation_recall_proba)


def check_events(events):
    """
    Verify the validity of the input events
    :param events: list of events, each represented by a couple (start, stop)
    :return: None. Raise an error for incorrect formed or non ordered events
    """
    if type(events) is not list:
        raise TypeError('Input `events` should be a list of couples')
    if not all([type(x) is tuple for x in events]):
        raise TypeError('Input `events` should be a list of tuples')
    if not all([len(x) == 2 for x in events]):
        raise ValueError('Input `events` should be a list of couples (start, stop)')
    if not all([x[0] <= x[1] for x in events]):
        raise ValueError('Input `events` should be a list of couples (start, stop) with start <= stop')
    if not all([events[i][1] < events[i+1][0] for i in range(len(events) - 1)]):
        raise ValueError('Couples of input `events` should be disjoint and ordered')


def other_checks(events_pred, events_gt, Trange):
    minimal_Trange = infer_Trange(events_pred, events_gt)
    if not Trange[0] <= minimal_Trange[0]:
        raise ValueError('`Trange` should include all the events')
    if not minimal_Trange[1] <= Trange[1]:
        raise ValueError('`Trange` should include all the events')

    if len(events_gt) == 0:
        raise ValueError('Input `events_gt` should have at least one event')

    if has_point_anomalies(events_pred) or has_point_anomalies(events_gt):
        raise ValueError('Cannot manage point anomalies currently')

    if Trange is None:
        # Set as default, but Trange should be indicated if probabilities are used
        raise ValueError('Trange should be indicated (or inferred with the `infer_Trange` function')


def precision_from_events(events_pred, events_gt, Trange):
    # testing the inputs
    check_events(events_pred)
    check_events(events_gt)
    other_checks(events_pred, events_gt, Trange)

    E_gt = get_all_E_gt_func(events_gt, Trange)
    aff_partition = affiliation_partition(events_pred, E_gt)

    p_precision = [affiliation_precision_proba(Is, J, E) for Is, J, E in zip(aff_partition, events_gt, E_gt)]
    if _len_wo_nan(p_precision) > 0:
        p_precision_average = _sum_wo_nan(p_precision) / _len_wo_nan(p_precision)
    else:
        p_precision_average = p_precision[0]  # math.nan

    return p_precision_average


def recall_from_events(events_pred, events_gt, Trange):
    # testing the inputs
    check_events(events_pred)
    check_events(events_gt)
    other_checks(events_pred, events_gt, Trange)

    E_gt = get_all_E_gt_func(events_gt, Trange)
    aff_partition = affiliation_partition(events_pred, E_gt)

    p_recall = [affiliation_recall_proba(Is, J, E) for Is, J, E in zip(aff_partition, events_gt, E_gt)]
    p_recall_average = sum(p_recall) / len(p_recall)

    return p_recall_average
