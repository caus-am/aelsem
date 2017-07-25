import sys
sys.path.append('../')

#import h5py
import sys
import scipy as SP
import scipy.io
import pylab as PL
import numpy as NP
from sklearn.metrics import roc_curve, auc
import heapq
import itertools
import math
from datetime import datetime
import ystruct
import rpy2.robjects as robjects
from scipy.stats import t
import random

from commondefs import *

def permuteIndexes(array, perm):
    return array[NP.ix_(perm, perm)]

if __name__ =='__main__':
    verbose = 1

    # Read data from R file
    #filename = 'simulNEW.RData'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'Jerome_datalist10.RData'
    if verbose:
        print 'Reading ', filename
#    filename = 'Jerome_unrealisticSetting2.RData'
#    filename = '/home/jorism/vcs/hughes/Hughes/data/Hughes.RData'
    #filename = '/home/jorism/vcs/hughes/Kemmeren/data/KemmerenFull.RData'
#    filename = '/tmp/simul.RData'
    robjects.r['load'](filename)

    Ngenes = int(robjects.r['data'][robjects.r['data'].names.index('p')][0])
    if verbose:
        print "Available number of genes:", Ngenes
    Nsamples = int(robjects.r['data'][robjects.r['data'].names.index('nObs')][0])
    if verbose:
        print "Available number of observations:", Nsamples
    Nint = int(robjects.r['data'][robjects.r['data'].names.index('nInt')][0])
    if verbose:
        print "Number of interventions:", Nint
    obs_data = SP.array(robjects.r['data'][robjects.r['data'].names.index('obs')])
    int_data = SP.array(robjects.r['data'][robjects.r['data'].names.index('int')])
    intpos = SP.array(robjects.r['data'][robjects.r['data'].names.index('intpos')],dtype='int') - 1

    simulated = 1
    if simulated:
        B = SP.array(robjects.r['data'][robjects.r['data'].names.index('B')])
        noiseCov = SP.array(robjects.r['data'][robjects.r['data'].names.index('noiseCov')])

    # construct (inverse) permutation
    perm = intpos
    for i in range(0,Ngenes):
        if not(i in intpos):
          perm = SP.append(perm,i)
    iperm = -SP.ones(Ngenes,int)
    for i in range(len(perm)):
        iperm[perm[i]] = i

    assert obs_data.shape[0] == Nsamples
    assert obs_data.shape[1] == Ngenes
    assert int_data.shape[0] == Nint
    assert int_data.shape[1] == Ngenes

    intvars = perm[0:Nint] # intervened variables

    # build ground truth
    X = (int_data - SP.tile(NP.mean(obs_data,axis=0),(Nint,1))) / SP.tile(NP.std(obs_data,axis=0,ddof=1),(Nint,1))
    CE3 = NP.absolute(X)

    # calculate correlations of gene expressions
    C = NP.corrcoef(x=obs_data,rowvar=0)

    # plot correlations
    if 0:
        PL.figure()
        plt = PL.imshow(C,aspect='auto')
        fig = PL.gcf()
        PL.clim()   # clamp the color limits
        PL.title("C")
        PL.pause(1)

    # plot ground truth causal effects
    if 0:
        PL.figure()
        plt = PL.imshow(CE3[:,perm[0:Ngenes]],aspect='auto')
        fig = PL.gcf()
        PL.clim()   # clamp the color limits
        PL.title("CE3")
        PL.pause(1)

#    Clo = 0.05 # lower threshold
#    Chi = 0.5  # upper threshold

#    Clo = pth2Cth(0.5,Nsamples,1)
#    Chi = pth2Cth(1e-5,Nsamples,1)

#    print pth2Cth(0.5,Nsamples,1)
#    print pth2Cth(0.5,Nsamples,2)
#    print pth2Cth(1e-5,Nsamples,1)
#    print pth2Cth(1e-5,Nsamples,2)

#    Clo = 0.03
#    Chi = 0.6

    Clo = pth2Cth(1e-1,Nsamples,1)
    Chi = pth2Cth(1e-4,Nsamples,1)

    if verbose:
        print "Using thresholds Clo = ", Clo, ", Chi = ", Chi
        print "Please be patient..."
        print "Calling C++ code..."

    Ypattern = 3
    print "Searching pattern ", Ypattern
    CextYs = ystruct.searchPattern(C,B,Clo,Chi,0,0)
    CYs = ystruct.searchPattern(C,B,Clo,Chi,0,Ypattern)
    extYs = []
    extY_xys = set([])
    for i in range(CextYs.shape[0]):
        extYs.append((CextYs[i,0],CextYs[i,1],CextYs[i,2],CextYs[i,3]))
        extY_xys.add((CextYs[i,0],CextYs[i,1]))
    Ys = []
    Y_xys = set([])
    for i in range(CYs.shape[0]):
        Ys.append((CYs[i,0],CYs[i,1],CYs[i,2],CYs[i,3]))
        Y_xys.add((CYs[i,0],CYs[i,1]))

#    (trueCextYs,trueCYs) = ystruct.searchYs(C,B,Clo,Chi,1)  OBSOLETE
    trueCextYs = ystruct.searchPattern(C,B,Clo,Chi,1,0)
    trueCYs = ystruct.searchPattern(C,B,Clo,Chi,1,1)
    true_extYs = []
    true_extY_xys = set([])
    for i in range(trueCextYs.shape[0]):
        true_extYs.append((trueCextYs[i,0],trueCextYs[i,1],trueCextYs[i,2],trueCextYs[i,3]))
        true_extY_xys.add((trueCextYs[i,0],trueCextYs[i,1]))
    true_Ys = []
    true_Y_xys = set([])
    for i in range(trueCYs.shape[0]):
        true_Ys.append((trueCYs[i,0],trueCYs[i,1],trueCYs[i,2],trueCYs[i,3]))
        true_Y_xys.add((trueCYs[i,0],trueCYs[i,1]))
    if verbose:
        print "Continuing with python code..."

    if 1:
        xy_pos = SP.zeros((Ngenes,Ngenes))
        true_xy_pos = SP.zeros((Ngenes,Ngenes))
        for xyuz in Ys:
            x,y,u,z = xyuz
            xy_pos[x,y] += 1
            xy_pos[y,x] -= 1
        for xyuz in true_Ys:
            x,y,u,z = xyuz
            true_xy_pos[x,y] += 1
            true_xy_pos[y,x] -= 1
        # find top 2% true effects
        topgt = heapq.nlargest(int(0.02 * Ngenes * Ngenes), zip(xy_pos.flatten(1), itertools.count()))
        for (val,index) in topgt:
            print (index,val,index % Ngenes,index / Ngenes,xy_pos[index % Ngenes,index / Ngenes],true_xy_pos[index % Ngenes,index / Ngenes])
        # threshold = topgt[int(0.10 * Nint * Ngenes) - 1][0]

    PL.ion()
    print B
    print noiseCov
    orgCov = NP.linalg.inv(NP.identity(Ngenes) - B).transpose().dot(noiseCov).dot(NP.linalg.inv(NP.identity(Ngenes) - B))
    print orgCov
    print NP.cov(m=obs_data,rowvar=0)
    PL.figure(1)
    visualizeADMG(B,noiseCov,NP.array(range(Ngenes),ndmin=2),abs(B).max(),abs(noiseCov).max(),0)
    for xyuz in Ys:
        x,y,u,z = xyuz
        pi = invert_permutation(NP.argsort(xyuz))
        print xyuz, NP.argsort(xyuz), pi
        if u < z:
            obs = NP.array(sorted(xyuz),ndmin=2)
            (projB, projS) = ystruct.projectADMG(B,noiseCov,obs,0)
            print xyuz, ': projecting onto ', obs
            print 'Indeed truely Y?', xyuz in true_Ys
            print 'projB: ', projB
            print 'projS: ', projS
            projCov = NP.linalg.inv((NP.identity(4) - projB)).transpose().dot(projS).dot(NP.linalg.inv((NP.identity(4) - projB)))
            print 'projCov: ', projCov
            print 'orgCov restricted to X,Y,U,Z: ', orgCov[sorted(xyuz),:][:,sorted(xyuz)]
            projCorr = (NP.diag(NP.power(NP.diag(projCov),-0.5))).dot(projCov).dot(NP.diag(NP.power(NP.diag(projCov),-0.5)))
            print 'projCorr: ', projCorr
            # pi[0] = x  pi[1] = y  pi[2] = u  pi[3] = z
            print 'extY tests:'
            print 'C(z,y) = C(',z,',',y,') = ', projCorr[pi[3],pi[1]], '(should be large)'
            print 'C(z,y|x) = C(',z,',',y,'|',x,') = ', partialcorr3(projCorr,pi[3],pi[1],pi[0]), ' (should be zero)'
            print 'C(z,u) = C(',z,',',u,') = ', projCorr[pi[3],pi[2]], '(should be zero)'
            print 'C(z,u|x) = C(',z,',',u,'|',x,') = ', partialcorr3(projCorr,pi[3],pi[2],pi[0]), ' (should be large)'
            print 'Extra Y tests:'
            print 'C(u,y) = C(',u,',',y,') = ', projCorr[pi[2],pi[1]], '(should be large)'
            print 'C(u,y|x) = C(',u,',',y,'|',x,') = ', partialcorr3(projCorr,pi[2],pi[1],pi[0]), ' (should be zero)'
            print 'Extra Y1 tests:'
            print 'C(z,x) = C(',z,',',x,') = ', projCorr[pi[3],pi[0]], '(should be large)'
            print 'C(x,y) = C(',x,',',y,') = ', projCorr[pi[0],pi[1]], '(should be large)'
            print 'C(u,x) = C(',u,',',x,') = ', projCorr[pi[2],pi[0]], '(should be large)'
            print 'C(u,y) = C(',u,',',y,') = ', projCorr[pi[2],pi[1]], '(should be large)'
            print 'C(x,u|y) = ', partialcorr3(projCorr,pi[0],pi[2],pi[1]), '(should be large)'
            print 'C(x,z|y) = ', partialcorr3(projCorr,pi[0],pi[3],pi[1]), '(should be large)'
            print 'C(u,z|y) = ', partialcorr3(projCorr,pi[2],pi[3],pi[1]), '(should be large)'
            print 'C(x,y|u) = ', partialcorr3(projCorr,pi[0],pi[1],pi[2]), '(should be large)'
            print 'C(x,z|u) = ', partialcorr3(projCorr,pi[0],pi[3],pi[2]), '(should be large)'
            print 'C(y,z|u) = ', partialcorr3(projCorr,pi[1],pi[3],pi[2]), '(should be large)'
            print 'C(x,y|z) = ', partialcorr3(projCorr,pi[0],pi[1],pi[3]), '(should be large)'
            print 'C(x,u|z) = ', partialcorr3(projCorr,pi[0],pi[2],pi[3]), '(should be large)'
            print 'C(y,u|z) = ', partialcorr3(projCorr,pi[1],pi[2],pi[3]), '(should be large)'
            print 'Extra Y2 tests:'
            print 'C(z,y|x,u) = ', partialcorr4(projCorr,pi[3],pi[1],pi[0],pi[2]), '(should be zero)'
            print 'C(u,y|x,z) = ', partialcorr4(projCorr,pi[2],pi[1],pi[0],pi[3]), '(should be zero)'
            print 'C(u,z|x,y) = ', partialcorr4(projCorr,pi[2],pi[3],pi[0],pi[1]), '(should be large)'
            print 'C(u,x|z,y) = ', partialcorr4(projCorr,pi[2],pi[0],pi[3],pi[1]), '(should be large)'
            print 'C(z,x|u,y) = ', partialcorr4(projCorr,pi[3],pi[0],pi[2],pi[1]), '(should be large)'
            print 'C(x,y|u,z) = ', partialcorr4(projCorr,pi[0],pi[1],pi[2],pi[3]), '(should be large)'
            if x in intvars:
                slope, intercept, r_value, p_value, std_err = SP.stats.linregress(obs_data[:,x],obs_data[:,y])
                maxe = 0.0
                avge = 0.0
                nre = 0
                for ipos in range(Nint):
                    if perm[ipos] == x:
                        e = abs(int_data[ipos,y] - (int_data[ipos,x]*slope+intercept))
                        if e > maxe:
                            maxe = e
                        avge = avge + e
                        nre += 1
                avge = avge / nre
                print 'Pair (', x, ', ', y, '): max Y_Error: ', maxe, ', avg Y_Error: ', avge
                print 'OUTPUT Deterministicness: ', projCorr[pi[0], pi[1]], 'Y_Error: ', avge, 'Correct: ', (xyuz in true_Ys)*1
#                if (avge > 1.0) and (abs(projCorr[pi[0], pi[1]]) < 0.5):
#                if (avge > 1.0):
                if 1:
                    PL.figure(2)
                    projBpi = permuteIndexes(projB,pi)
                    projSpi = permuteIndexes(projS,pi)
                    obspi = NP.array(obs[0,pi],ndmin=2)
                    visualizeADMG(projBpi,projSpi,obspi,abs(projBpi).max(),abs(projSpi).max(),1)
                    PL.waitforbuttonpress()

    if 0:
        for x in range(Ngenes):
            for y in range(Ngenes):
                if xy_pos[x,y] > 30:
                    slope, intercept, r_value, p_value, std_err = SP.stats.linregress(obs_data[:,x],obs_data[:,y])
                    if verbose:
                        print 'Pair (', x, ', ', y, '): xy_pos=', xy_pos[x,y], ', Error: ', abs(int_data[iperm[x],y] - (int_data[iperm[x],x]*slope+intercept))**2
                    PL.figure()
                    PL.plot(obs_data[:,x],obs_data[:,y],'b.',int_data[iperm[x],x],int_data[iperm[x],y],'rx',NP.arange(2)*4-2,(NP.arange(2)*4-2)*slope+intercept,'g-',int_data[iperm[x],x],int_data[iperm[x],x]*slope+intercept,'gx')
                    PL.xlabel('Cause')
                    PL.ylabel('Effect')
                    PL.title(str(x) + str(y))
                    PL.show()
                    PL.pause(1)
