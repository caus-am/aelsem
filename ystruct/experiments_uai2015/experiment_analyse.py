import sys
sys.path.append('../')

import scipy as SP
import scipy.io
import numpy as NP
import math
import rpy2.robjects as robjects
import random

import ystruct
from commondefs import *

def recall(TP, FN, UP):
    if (TP + FN + UP) != 0:
        recall = (1.0 * TP) / (TP + FN + UP)
    else:
        recall = float('nan')
    return recall

def precision(TP, FP):
    if (TP + FP) != 0:
        precision = (1.0*TP) / (TP + FP)
    else:
        precision = float('nan')
    return precision

def avgL1L2Err(count, sumErrL1, sumErrL2):
    if count > 0:
        avgL1Err = sumErrL1 / count
        avgL2Err = math.sqrt(sumErrL2 / count)
    else:
        avgL1Err = float('nan')
        avgL2Err = float('nan')
    return (avgL1Err, avgL2Err)

def printstats(csv_file,stats):
    f = open(csv_file, 'w')

    # print summary statistics
    for i in range(60/6):
        # TP, FP, TN, FN, UP, UN -> TP, FP, TN, FN, UP, UN, recall = TP / (TP + FN + UP), precision = TP / (TP + FP)
        (TP,FP,TN,FN,UP,UN) = (stats[i*6+0,0], stats[i*6+1,0], stats[i*6+2,0], stats[i*6+3,0], stats[i*6+4,0], stats[i*6+5,0])
        print >>f, TP, FP, TN, FN, UP, UN, recall(TP,FN,UP), precision(TP,FP), 

    for i in range(60/6,84/6):
        # TP, FP, TN, FN, UP, UN -> TP, FP, TN, FN, recall = TP / (TP + FN + UP), precision = TP / (TP + FP)
        (TP,FP,TN,FN,UP,UN) = (stats[i*6+0,0], stats[i*6+1,0], stats[i*6+2,0], stats[i*6+3,0], stats[i*6+4,0], stats[i*6+5,0])
        print >>f, TP, FP, TN, FN, recall(TP,FN,UP), precision(TP,FP), 

    p = 84
    for i in range(0,36/9):
        # ErrCount ErrL1 ErrL2 ErrCount_TP ErrL1_TP ErrL2_TP ErrCount_FP ErrL1_FP ErrL2_FP -> Y_avgL1Err, Y_avgL2Err, Y_avgL1ErrTP, Y_avgL2ErrTP, Y_avgL1ErrFP, Y_avgL2ErrFP
        (ErrCount, ErrL1, ErrL2, ErrCount_TP, ErrL1_TP, ErrL2_TP, ErrCount_FP, ErrL1_FP, ErrL2_FP) = (stats[p+i*9+0,0], stats[p+i*9+1,0], stats[p+i*9+2,0], stats[p+i*9+3,0], stats[p+i*9+4,0], stats[p+i*9+5,0], stats[p+i*9+6,0], stats[p+i*9+7,0], stats[p+i*9+8,0])
        (avgL1Err, avgL2Err) = avgL1L2Err(ErrCount, ErrL1, ErrL2)
        (avgL1ErrTP, avgL2ErrTP) = avgL1L2Err(ErrCount_TP, ErrL1_TP, ErrL2_TP)
        (avgL1ErrFP, avgL2ErrFP) = avgL1L2Err(ErrCount_FP, ErrL1_FP, ErrL2_FP)
        print >>f, avgL1Err, avgL2Err, avgL1ErrTP, avgL2ErrTP, avgL1ErrFP, avgL2ErrFP, 

    p = 84 + 36
    for i in range(2):
        # ErrCount, ErrL1, ErrL2 -> ErrL1 / ErrCount, math.sqrt(ErrL2 / ErrCount)
        (ErrCount, ErrL1, ErrL2) = (stats[p+i*3+0,0], stats[p+i*3+1,0], stats[p+i*3+2,0])
        print >>f, ErrL1 / ErrCount, math.sqrt(ErrL2 / ErrCount), 

    print >>f

    f.close()

    return

def formatstats(tex_file,stats,concise):
    f = open(tex_file, 'w')

    # format summary statistics as .tex file

    print >>f, '\\documentclass{article}'
    print >>f, '\\usepackage[landscape]{geometry}'
    print >>f, '\\newcommand\\indep{{\\,\\perp\\mkern-12mu\\perp\\,}}'
    print >>f, '\\newcommand\\notindep{{\\,\\not\\mkern-1mu\\perp\\mkern-12mu\\perp\\,}}'
    print >>f, '\\newcommand\\given{\\,|\\,}'
    print >>f, '\\begin{document}'
    print >>f, '\\tiny'
    if concise:
        print >>f, '\\begin{tabular}{llll}'
        print >>f, 'Pattern & Total \\# & Recall & Precision \\\\'
    else:
        print >>f, '\\begin{tabular}{llllllllll}'
        print >>f, 'Pattern & Total \\# & TP & FP & TN & FN & UP & UN & Recall & Precision \\\\'
    print >>f, '\\hline'
        
    names = ['$X \\indep Y$','$X \\notindep Y$','$X \\indep Y \\given Z$','$X \\notindep Y \\given Z$','\\mbox{$X \\indep Y \\given [Z]$}','\\mbox{$X \\notindep Y \\given [Z]$}','\\texttt{extY}','\\texttt{Y}','\\texttt{Y1}','\\texttt{Y2}']
    stride = 6
    for i in range(len(names)):
        print >>f, names[i], '&',
        pos = i*stride
        (TP,FP,TN,FN,UP,UN) = (stats[pos+0,0], stats[pos+1,0], stats[pos+2,0], stats[pos+3,0], stats[pos+4,0], stats[pos+5,0])
        #print >>f, TP, FP, TN, FN, UP, UN, recall(TP,FN,UP), precision(TP,FP), 
        total = TP + FP + TN + FN + UP + UN
        gt_pos = TP + FN + UP
        print >>f, '%d' % total, '&',
        if not concise:
            print >>f, '%d' % TP, '&',
            print >>f, '%d' % FP, '&',
            print >>f, '%d' % TN, '&',
            print >>f, '%d' % FN, '&',
            print >>f, '%d' % UP, '&',
            print >>f, '%d' % UN, '&',
        print >>f, '%.4f' % recall(TP,FN,UP), '&',
        print >>f, '%.2f' % precision(TP,FP), '(', '%.5f' % (gt_pos/total), ') \\\\'
    print >>f, '\\hline'
    print >>f, '\\end{tabular}'
    print >>f
    lastpos = len(names)*stride

    names = ['\\texttt{extY}','\\texttt{Y}','\\texttt{Y1}','\\texttt{Y2}']
    if concise:
        print >>f, '\\begin{tabular}{llll}'
        print >>f, 'Test pattern & Total \\# & Recall & Precision \\\\'
    else:
        print >>f, '\\begin{tabular}{llllllll}'
        print >>f, 'Test pattern & Total \\# & TP & FP & TN & FN & Recall & Precision \\\\'
    print >>f, '\\hline'
    stride = 6
    for i in range(len(names)):
        print >>f, names[i], ' & ',
        pos = lastpos + i*stride
        (TP,FP,TN,FN,UP,UN) = (stats[pos+0,0], stats[pos+1,0], stats[pos+2,0], stats[pos+3,0], stats[pos+4,0], stats[pos+5,0])
        #print >>f, TP, FP, TN, FN, recall(TP,FN,UP), precision(TP,FP), 
        total = TP + FP + TN + FN + UP + UN
        gt_pos = TP + FN + UP
        print >>f, '%d' % total, '&',
        if not concise:
            print >>f, '%d' % TP, '&',
            print >>f, '%d' % FP, '&',
            print >>f, '%d' % TN, '&',
            print >>f, '%d' % FN, '&',
        print >>f, '%.4f' % recall(TP,FN,UP), '&',
        print >>f, '%.2f' % precision(TP,FP), '(', '%.5f' % (gt_pos/total), ') \\\\'
    print >>f, '\\hline'
    print >>f, '\\end{tabular}'
    print >>f
    lastpos = lastpos + len(names)*stride

    #names = ['\\texttt{extY}','extY TP','extY FP','\\texttt{Y}','Y TP','Y FP','\\texttt{Y1}','Y1 TP','Y1 FP','\\texttt{Y2}','Y1 TP','Y2 FP']
    names = ['\\texttt{extY}','\\texttt{Y}','\\texttt{Y1}','\\texttt{Y2}']
    print >>f, '\\begin{tabular}{l|lll|lll}'
    print >>f, '       & \\multicolumn{3}{c|}{$\\ell_1$ error} & \\multicolumn{3}{c}{$\\ell_2$ error} \\\\'
    print >>f, 'Method & all & only TP & only FP & all & only TP & only FP \\\\'
    print >>f, '\\hline'
    stride = 9
    for i in range(len(names)):
        print >>f, names[i],
        pos = lastpos + i*stride
        (ErrCount, ErrL1, ErrL2) = (stats[pos+0,0], stats[pos+1,0], stats[pos+2,0])
        (avgL1Err, avgL2Err) = avgL1L2Err(ErrCount, ErrL1, ErrL2)
        (ErrCount, ErrL1, ErrL2) = (stats[pos+3,0], stats[pos+4,0], stats[pos+5,0])
        (avgL1ErrTP, avgL2ErrTP) = avgL1L2Err(ErrCount, ErrL1, ErrL2)
        (ErrCount, ErrL1, ErrL2) = (stats[pos+6,0], stats[pos+7,0], stats[pos+8,0])
        (avgL1ErrFP, avgL2ErrFP) = avgL1L2Err(ErrCount, ErrL1, ErrL2)
        #print >>f, avgL1Err, avgL2Err, avgL1ErrTP, avgL2ErrTP, avgL1ErrFP, avgL2ErrFP, 
        print >>f, '& %.2f' % avgL1Err,
        print >>f, '& %.2f' % avgL1ErrTP,
        print >>f, '& %.2f' % avgL1ErrFP,
        print >>f, '& %.2f' % avgL2Err,
        print >>f, '& %.2f' % avgL2ErrTP,
        print >>f, '& %.2f' % avgL2ErrFP, '\\\\'
    print >>f, '\\hline'
    lastpos = lastpos + len(names)*stride

    names = ['$p(Y \\given \\mathrm{do}(X)) = p(Y)$','$p(Y \\given \\mathrm{do}(X)) = p(Y \\given X)$']
    stride = 3
    for i in range(len(names)):
        print >>f, names[i], '&',
        pos = lastpos + i*stride
        (ErrCount, ErrL1, ErrL2) = (stats[pos+0,0], stats[pos+1,0], stats[pos+2,0])
        #print >>f, ErrL1 / ErrCount, math.sqrt(ErrL2 / ErrCount), 
        avgL1Err = ErrL1 / ErrCount
        avgL2Err = math.sqrt(ErrL2 / ErrCount)
        print >>f, '%.2f' % avgL1Err, '& - & - &',
        print >>f, '%.2f' % avgL2Err, '& - & - \\\\'
    print >>f, '\\hline'
    print >>f, '\\end{tabular}'
    print >>f, '\\end{document}'

    f.close()

    return

def main():
    # Set to >0 for debugging
    # WARNING: If verbose >= 2, then table 2 data not written to stats!
    verbose = 0 # 5

    # Read data from R file
    if len(sys.argv) < 4:
        sys.exit('Syntax: ipython csvfile texfile experiment_analyse.py datafilename1.RData [datafilename2.RData ...]')
    csv_file = sys.argv[1]
    tex_file = sys.argv[2]

    # Initialize statistics
    totstats = NP.zeros((84+4*9+2*3,1))

    # For all files
    for it in range(3,len(sys.argv)):
        filename = sys.argv[it]
        if verbose:
            print 'Reading ', filename
        robjects.r['load'](filename)
        data_p = int(robjects.r['data'][robjects.r['data'].names.index('p')][0])
        if verbose:
            print "Number of features:", data_p
        data_nObs = int(robjects.r['data'][robjects.r['data'].names.index('nObs')][0])
        if verbose:
            print "Number of passive observations:", data_nObs
        data_nInt = int(robjects.r['data'][robjects.r['data'].names.index('nInt')][0])
        if verbose:
            print "Number of interventions:", data_nInt
        data_obs = SP.array(robjects.r['data'][robjects.r['data'].names.index('obs')])
        data_int = SP.array(robjects.r['data'][robjects.r['data'].names.index('int')])
        data_intpos = SP.array(robjects.r['data'][robjects.r['data'].names.index('intpos')],dtype='int') - 1
        data_B = SP.array(robjects.r['data'][robjects.r['data'].names.index('B')])
        #data_noiseCov = SP.array(robjects.r['data'][robjects.r['data'].names.index('noiseCov')])
        assert data_obs.shape[0] == data_nObs
        assert data_obs.shape[1] == data_p
        assert data_int.shape[0] == data_nInt
        assert data_int.shape[1] == data_p
        assert data_intpos.shape[0] == data_nInt
        intvars = NP.unique(data_intpos) # intervened variables
        if verbose >= 9:
            print data_intpos[0:9]
            raw_input("Press enter...")

        # calculate basic statistics of gene expressions
        C = NP.corrcoef(x=data_obs,rowvar=0)
        data_obs_cov = NP.cov(m=data_obs,rowvar=0)
        data_obs_mean = NP.mean(a=data_obs,axis=0)

        # choose thresholds

        # Clo = 0.05 # lower threshold
        # Chi = 0.5  # upper threshold

        # Clo = pth2Cth(0.5,data_nObs,1)
        # Chi = pth2Cth(1e-5,data_nObs,1)

        # print pth2Cth(0.5,data_nObs,1)
        # print pth2Cth(0.5,data_nObs,2)
        # print pth2Cth(1e-5,data_nObs,1)
        # print pth2Cth(1e-5,data_nObs,2)

        # Clo = 0.03
        # Chi = 0.6

        Clo = pth2Cth(1e-1,data_nObs,1)
        Chi = pth2Cth(1e-4,data_nObs,1)

        if verbose:
            print "Using thresholds Clo = ", Clo, ", Chi = ", Chi

        # analyse stuff
        stats = ystruct.analyse_Y(data_obs,data_int,NP.array(data_intpos,ndmin=2),data_B,Clo,Chi,verbose,filename.replace('.Rdata','_struc.csv'))
        #stats = ystruct.analyse_Y(data_obs,data_int,NP.array(data_intpos,ndmin=2,dtype='i8'),data_B,Clo,Chi,verbose,filename.replace('.Rdata','_struc.csv')) # THIJS: added dtype i8 to data_intpos, because in the C++ code, the 32 bit ints were being interpreted as 64 bit ints (this was on Windows)
        #stats = ystruct.analyse_Y(data_obs,data_int,NP.array(data_intpos,ndmin=2),data_B,Clo,Chi,verbose,filename.replace('.Rdata','_struc.csv'))

        # print file-specific statistics
        # printstats(stats)

        # calculate total statistics
        totstats += stats

    printstats(csv_file,totstats)
    formatstats(tex_file,totstats,0)
    #formatstats(tex_file,totstats,1)

if __name__ =='__main__':
    main()
