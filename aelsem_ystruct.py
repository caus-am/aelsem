from __future__ import print_function

import numpy as np
import rpy2.robjects as robjects
import collections
import warnings
import pickle
import time
import sys

import aelsem

def analyse_quadruple(classify_data, S_full, N, other_results_line, mB,
                      results, verbose=1):
    more_results_line = np.zeros(6+6, dtype=int)
    more_results_line[0:6] = other_results_line

    vars = other_results_line[0:4]
    S_sub = S_full[vars,:][:,vars]

    # Determine the LSEM equivalence classes with the best BIC scores
    best_class_by_type = aelsem.classify(classify_data, S=S_sub, N=N,
                                         epsilon=1e-6,
                                         get_scores_by_type=True,
                                         smooth=False)
    ##(best_DAG, best_MAG, best_BAP, best_ADMG, best_PD)
    best_class = best_class_by_type[-1]
    more_results_line[7:12] = best_class_by_type

    p = mB.shape[0]
    mO = np.eye(p, dtype=bool)
    vars_mask = np.zeros(p, dtype=bool)
    vars_mask[vars] = True
    (mB_marg, mO_marg) = aelsem.marginalized_model((mB, mO), vars_mask)
    #aelsem.print_model((mB_marg, mO_marg))
    mB_marg = mB_marg[vars,:][:,vars]
    mO_marg = mO_marg[vars,:][:,vars]
    #aelsem.print_model((mB_marg, mO_marg))
    true_class = classify_data['model_index'][aelsem.hash_model((mB_marg,
                                                                 mO_marg))]
    if verbose >= 3:
        print("True class:", true_class)
    more_results_line[6] = true_class

    return more_results_line

def analyse_file_sanity_check(more_results, results):
    num_lines = more_results.shape[0]
    agree = 0
    for i in range(num_lines):
        m_true = more_results[i,4]
        c_true2 = more_results[i,6]
        m_true2 = 0
        if c_true2 == 90:
            m_true2 = 4
        elif c_true2 == 246:
            m_true2 = 1
        elif c_true2 == 238:
            m_true2 = -1
        if m_true2 != m_true:
            print("Disagreement: {0} -> {1} vs. {2}"
                  .format(c_true2, m_true2, m_true))
        else:
            agree += 1
    print("Agreements:", agree)

def analyse_file_aggregate(more_results, results, use_column):
    num_lines = more_results.shape[0]
    for i in range(num_lines):
        best_class = more_results[i,use_column]
        results[more_results[i,4]][more_results[i,5]][best_class] += 1


def analyse_file_compute(base_filename, classify_data, results, verbose=1):
    R_filename = base_filename + ".Rdata"
    csv_filename = base_filename + "_struc.csv"
    csv2_filename = base_filename + "_sel.csv"

    robjects.r['load'](R_filename)
    #data_p = int(robjects.r['data'][robjects.r['data'].names.index('p')][0])
    data_nObs = int(robjects.r['data'][robjects.r['data'].names.index('nObs')][0])
    #data_nInt = int(robjects.r['data'][robjects.r['data'].names.index('nInt')][0])
    data_obs = np.array(robjects.r['data'][robjects.r['data'].names.index('obs')])
    #data_int = np.array(robjects.r['data'][robjects.r['data'].names.index('int')])
    #data_intpos = np.array(robjects.r['data'][robjects.r['data'].names.index('intpos')],dtype='int') - 1
    data_Lambda = np.array(robjects.r['data'][robjects.r['data'].names.index('B')])

    #print(data_p, "variables,", data_nObs, "data points")
    S_full = aelsem.sample_covariance_matrix(data_obs)
    #np.set_printoptions(threshold=np.inf)
    #print(S_full)

    mB = (np.fabs(data_Lambda) > 1e-12).T

    with warnings.catch_warnings():
        # ignore warnings about empty files
        warnings.filterwarnings("ignore")#, message="empty input file")
        other_results = np.loadtxt(csv_filename, delimiter=',', dtype=int,
                                   ndmin=2)
    #print(other_results.shape) # look at prev_results.shape[0]

    num_lines = other_results.shape[0]

    # To just report the file sizes without doing anything:
    #print(base_filename, "-", num_lines, "lines")
    #return

    more_results = np.zeros((num_lines, 6+6), dtype=int)
    for line_no in range(num_lines):
        if verbose >= 2 or (verbose and line_no % 100 == 0):
            print("File", base_filename, "- line", line_no+1,
                  "of", other_results.shape[0])
        more_results[line_no,:] = analyse_quadruple(classify_data,
                                                    S_full, data_nObs,
                                                    other_results[line_no,:],
                                                    mB, results,
                                                    verbose=verbose)
    # store more_results in file csv2_filename
    np.savetxt(csv2_filename, more_results, fmt='%1i', delimiter=',')
    print("File", base_filename, "completed")

    return more_results

def analyse_file(base_filename, classify_data, results, use_column, verbose=1):
    csv2_filename = base_filename + "_sel.csv"

    try:
        with warnings.catch_warnings():
            # ignore warnings about empty files
            warnings.filterwarnings("ignore")#, message="empty input file")
            more_results = np.loadtxt(csv2_filename, delimiter=',', dtype=int,
                                      ndmin=2)
    except IOError:
        more_results = analyse_file_compute(base_filename, classify_data,
                                            results, verbose)

    analyse_file_aggregate(more_results, results, use_column)


def main(argv):
    # Usage: run with just one command-line argument (e.g. p10_final) to
    # analyse all data in that experiment (consisting of 100 runs). To
    # parallelize, add an additional argument 0-99 to specify which of these
    # runs to analyse. The results for each run will be stored in a file, so
    # that running the command without the numeric argument afterwards will
    # just load those results and summarize them.

    #exp_name = "p10_final"
    #exp_name = "p30_final"
    #exp_name = "p50_final"
    exp_name = argv[1]

    if len(argv) >= 3:
        file_numbers = [int(argv[2])]
        report_aggregates = False
    else:
        file_numbers = range(100)
        report_aggregates = True

    use_column = 11 # Change this to 8 to get results for MAGs instead
    columns = {6: "truth", 7: "DAGs", 8: "MAGs", 9: "BAPs", 10: "ADMGs",
               11: "all path diagrams"}
    report_details = True #False

    store_aggregates = False

    verbose = 1
    # Meaning of verbose:
    # 0: only output notification when file is done
    # 1: current line number (every hundred lines) only
    # 2: above for every line, plus current model class being tested
    # 3: also some intermediate results

    base_filename = "ystruct/experiments_uai2015/exp_{0}/simulData".format(exp_name)

    classify_data = aelsem.classify_prepare(4, allow_cycles=False)

    results = {}
    for i in [0,-1,1,4]:
        results[i] = {}
        for j in range(-1,5):
            results[i][j] = collections.Counter()

    starting_time = time.clock()
    for file_no in file_numbers:
        analyse_file(base_filename + str(file_no), classify_data, results,
                     use_column, verbose=verbose)
    time_used = time.clock() - starting_time
    num_selections = 0
    for i in [0,-1,1,4]:
        for j in range(-1,5):
            num_selections += sum(results[i][j].values())
    print("Total time:", time_used)
    print("Number of model selections:", num_selections)
    if num_selections > 0:
        print("Average time per selection:", time_used / num_selections)

    if store_aggregates:
        pickle.dump(results, open("ystr_results_{0}.p".format(exp_name), "wb"))
        # To load: results = pickle.load(open("ystr-results.p", "rb"))

    if not report_aggregates:
        return

    print("*** RESULTS FOR: {0}, {1} ***".format(argv[1], columns[use_column]))

    # numbering used in csv files
    m_name = {0: "nothing", 1: "extended Y", -1: "extended Y (refl)",
              2: "Y [0]", 3: "Y [1]", 4: "Y [2]"}
    # numbering used by aelsem.py
    model_name = {90: "Y", 105: "upside-down Y",
                  246: "extended Y", 238: "extended Y (refl)",
                  243: "Verma Y", 233: "Verma Y (refl)",
                  247: "tetrad + indep"}
    test_TP = np.zeros(5, dtype=int)
    test_FP = np.zeros(5, dtype=int)
    filter_TP = np.zeros(5, dtype=int)
    filter_FP = np.zeros(5, dtype=int)
    for m_true in [0,1,-1,4]:
        for m_indep in [0,1,-1,2,3,4]:
            c = results[m_true][m_indep]
            if len(c) == 0:
                continue
            total = sum(c.values())
            if report_details:
                print("True structure:", m_name[m_true],
                      "\tStructure found by tests:", m_name[m_indep])

            # If 4 nodes test positively as a Y-structure, then so would the 4
            # nodes with a and b swapped, but the second 4-tuple isn't reported
            # in the input file
            mult = 2 if m_indep >= 2 else 1
            if m_indep != 0:
                if m_true == m_indep or m_true == 4:
                    test_TP[abs(m_indep)] += mult * total
                else:
                    test_FP[abs(m_indep)] += mult * total

            for model, number in c.most_common():
                if report_details:
                    print("    {0}/{1} times model class {2}"
                          .format(number, total, model), end="")
                    if model in model_name:
                        print(" (", model_name[model], ")", sep="")
                    else:
                        print("")
                    aelsem.print_model_class(classify_data['model_classes'][model], indent=2)
                m_sel = 0
                if model in model_name:
                    if model_name[model] == "Y":
                        m_sel = 4
                    elif model_name[model] == "extended Y":
                        m_sel = 1
                    elif model_name[model] == "extended Y (refl)":
                        m_sel = -1

                m_filter = 0
                if m_indep >= 2 and m_sel == 4:
                    m_filter = m_indep
                elif m_indep != 0 and m_sel != 0:
                    if m_indep * m_sel == -1:
                        print("indep and sel disagreed on orientation of extY ({0} times here)".format(number))
                        # filter these out as well
                    elif m_indep == 1 or m_sel == 1:
                        m_filter = 1
                    else:
                        m_filter = -1
                mult = 2 if m_filter >= 2 else 1
                if m_filter != 0:
                    if m_true == m_filter or m_true == 4:
                        filter_TP[abs(m_indep)] += mult * number
                    else:
                        filter_FP[abs(m_indep)] += mult * number

    for m_indep in [3,2,1]:
        test_TP[m_indep] += test_TP[m_indep + 1]
        test_FP[m_indep] += test_FP[m_indep + 1]
        filter_TP[m_indep] += filter_TP[m_indep + 1]
        filter_FP[m_indep] += filter_FP[m_indep + 1]

    print("Results for independence test method:")
    for m_indep in [1,2,3,4]:
        print("{0}:\nTP = {1}\tFP = {2}\tprec = {3}"
              .format(m_name[m_indep], test_TP[m_indep], test_FP[m_indep],
                      test_TP[m_indep] * 1.0 / (test_TP[m_indep] + test_FP[m_indep])))
    print("Results for combined (filtering) method:")
    for m_indep in [1,2,3,4]:
        print("{0}:\nTP = {1}\tFP = {2}\tprec = {3}"
              .format(m_name[m_indep], filter_TP[m_indep], filter_FP[m_indep],
                      filter_TP[m_indep] * 1.0 / (filter_TP[m_indep] + filter_FP[m_indep])))
    print("Relative recall of filtering compared to tests:")
    for m_indep in [1,2,3,4]:
        print("{0}:\nTPfilter = {1}\tTPtests = {2}\trelative recall = {3}"
              .format(m_name[m_indep], filter_TP[m_indep], test_TP[m_indep],
                      "n/a" if test_TP[m_indep] == 0 else
                      1.0 * filter_TP[m_indep] / test_TP[m_indep]))


    print("*** END OF RESULTS FOR: {0}, {1} ***".format(argv[1], columns[use_column]))


    # Print all model classes with a known arrow X -> Y:
    #for m, model in enumerate(model_classes):
    #    marks = aelsem.model_class_marks(model)
    #    if marks[0,1] == "-->":
    #        print(m)
    #        aelsem.print_model_class(model)

if __name__ == "__main__":
    main(sys.argv)
