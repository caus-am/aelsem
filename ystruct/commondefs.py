import math
import scipy as SP
from scipy.stats import t
import pylab as PL
import pygraphviz as PGV
import numpy as NP

def invert_permutation(p):
    '''Returns an array s, where s[i] gives the index of i in p.
    The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1. 
    '''
    n = len(p)
    s = NP.zeros(n, dtype = NP.int32)
    i = NP.arange(n, dtype = NP.int32)
    NP.put(s, p, i) # s[p[i]] = i 
    return s

def partialcorr(Cxy, Cxz, Cyz, N):
    """Calculates partial correlation rho_{xy|z} from Pearson's correlations rho_{xy}, rho_{xz} and rho_{yz},
    and the p-value (given the number of data points N).
    The formula for the partial correlation was taken from the wikipedia page on partial correlations.
    The calculation of the p-value (assuming a Student's t distribution) is taken from the partialcorr.m file in the MatLab Stats toolbox.
    """
    Cxy_z = (Cxy - Cxz * Cyz) / math.sqrt((1.0  - Cxz**2) * (1.0 - Cyz**2))   # partial correlation rho_{xy|z}
    dz = 1  # dimension of conditioning variable
    df = max(N - dz - 2,0)  # degrees of freedom
    tstat = Cxy_z / math.sqrt(1.0 - Cxy_z ** 2)  # calculate t statistic
    tstat = math.sqrt(df) * tstat
    pval = 2 * t.cdf(-abs(tstat), df, loc=0, scale=1)  # calculate p value
    return (Cxy_z,pval)

def pcor(X,Y,Z):
    Cxy = SP.stats.pearsonr(X,Y)[0]
    Cxz = SP.stats.pearsonr(X,Z)[0]
    Cyz = SP.stats.pearsonr(Y,Z)[0]
    return partialcorr(Cxy,Cxz,Cyz,len(X))[1]

def partialcorr3(C, x, y, z):
    Cxy = C[x,y]
    Cxz = C[x,z]
    Cyz = C[y,z]
    Cxy_z = (Cxy - Cxz * Cyz) / math.sqrt((1.0  - Cxz**2) * (1.0 - Cyz**2))
    return Cxy_z

def partialcorr4(C, x, y, z0, z1):
    Cxy_z1 = partialcorr3(C, x, y, z1)
    Cxz0_z1 = partialcorr3(C, x, z0, z1)
    Cyz0_z1 = partialcorr3(C, y, z0, z1)
    Cxy_z0z1 = (Cxy_z1 - Cxz0_z1 * Cyz0_z1) / math.sqrt((1.0  - Cxz0_z1**2) * (1.0 - Cyz0_z1**2))
    return Cxy_z0z1

def pth2Cth(pth,N,dz):
    """Convert threshold on partial correlation to equivalent threshold on its p-value"""
    #dz = 1  # dimension of conditioning variable
    df = max(N - dz - 2,0)  # degrees of freedom
    y = -t.isf(1.0 - pth / 2.0, df, loc=0, scale=1) / math.sqrt(df) 
    Cth = abs(y / math.sqrt(1.0 + y ** 2))
    return Cth

def Cth2pth(Cth,N,dz):
    """Convert threshold on p-value to equivalent threshold of partial correlation"""
    #dz = 1  # dimension of conditioning variable
    df = max(N - dz - 2,0)  # degrees of freedom
    tstat = Cth / math.sqrt(1.0 - Cth ** 2)  # calculate t statistic
    tstat = math.sqrt(df) * tstat
    pth = 2 * t.cdf(-abs(tstat), df, loc=0, scale=1)  # calculate p value
    return pth

def visualizeADMG(B,S,obs,maxB,maxS,Y):
    G = PGV.AGraph(strict=False,directed=True)
    N = B.shape[0]
    assert( N == B.shape[1] )
    assert( N == S.shape[0] )
    assert( N == S.shape[1] )
    if Y and (N == 4):
        positions = ["0,-2!","0,-4!","-1,0!","1,0!"]
        for i in range(N):
            G.add_node(i,label=obs[0,i],pos=positions[i])
    else:
        for i in range(N):
            G.add_node(i,label=obs[0,i])
    for i in range(N):
        for j in range(N):
            if B[i,j] != 0.0:
                col = abs(B[i,j] / maxB)
                if col > 1.0:
                    col = 1.0
                G.add_edge(i,j,color='0.0 ' + str(col * 0.8 + 0.2) + ' 1.0')
    for i in range(N):
        for j in range(i+1,N):
            if S[i,j] != 0.0:
                col = abs(S[i,j] / maxS)
                if col > 1.0:
                    col = 1.0
                G.add_edge(i,j,dir='both',color='0.7 ' + str(col * 0.8 + 0.2) + ' 1.0')
    #print G.string()
    if Y:
        G.layout(prog='neato')
    else:
        G.layout(prog='dot')
    G.draw('/tmp/pygraphviz.png')
    img = PL.imread('/tmp/pygraphviz.png')
    PL.imshow(img)
    return
