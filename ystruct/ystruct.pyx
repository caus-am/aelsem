import numpy as NP
cimport numpy as NP
from libcpp cimport bool
from libcpp.string cimport string
from numpy cimport int64_t

cdef extern from "numpy/arrayobject.h":
    ctypedef int intp
    ctypedef extern class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef int nd
        cdef intp *dimensions
        cdef intp *strides
        cdef int flags

cdef extern from "types.h":
    ctypedef double mfloat_t
    ctypedef int64_t mint_t
    
    cdef cppclass MatrixXd:
        MatrixXd()
        int rows()
        int cols()

    cdef cppclass MatrixXi:
        MatrixXi()
        int rows()
        int cols()
    
    cdef cppclass MMatrixXd:
        MMatrixXd(mfloat_t *,int,int)
        int rows()
        int cols()
    
    cdef cppclass MMatrixXi:
        MMatrixXi(mint_t *,int,int)
        int rows()
        int cols()
        
    cdef cppclass MMatrixXdRM:
        MMatrixXdRM(mfloat_t *,int,int)
        int rows()
        int cols()
        
##conversion for numpy arrays
def ndarrayF64toC(ndarray A):
    #transform numpy object to enforce Fortran contiguous byte order 
    #(meaining column-first order for Cpp interfacing)
    return NP.asarray(A, order="F")

cdef extern from "ystruct.h": 
    void c_copy "copy" (MMatrixXd* m_out, MatrixXd* m_in) except +
    void c_copy "copy" (MMatrixXi* m_out, MatrixXi* m_in) except +
    int c_searchYs "searchYs" (MatrixXi* extYs,MatrixXi* Ys,MMatrixXd* C,MMatrixXd* B,double Clo,double Chi,int oracle) except +
    int c_searchPattern "searchPattern" (MatrixXi* patterns,MMatrixXd* C,MMatrixXd* B,double Clo,double Chi,int oracle,int pattern) except +
    int c_analyse_Y "analyse_Y" (MMatrixXd* data_obs,MMatrixXd* data_int,MMatrixXi* data_intpos,MMatrixXd* B,MatrixXd* stats,double Clo,double Chi,int verbose, string struc_csv) except +
    bool c_pairwise_independent "pairwise_independent" (MMatrixXd* B,int x,int y) except +
    bool c_pairwise_conditionally_independent "pairwise_conditionally_independent" (MMatrixXd* B,int x,int y,int z) except +
    void c_pairwise_dependences "pairwise_dependences" (MMatrixXd* B,MatrixXi* tmp_deps) except +
    bool c_ancestor "ancestor" (MMatrixXd* B,int x,int y) except +
    void c_projectADMG "projectADMG" (MMatrixXd* B,MMatrixXd* S,MMatrixXi* obsVar,MatrixXd* projB,MatrixXd* projS,int verbose) except +

def searchYs(ndarray C,ndarray B,double Clo,double Chi,int oracle):
    # returns (extYs,Ys)
    # make sure numpy arrays are contiguous
    C = ndarrayF64toC(C)
    B = ndarrayF64toC(B)

    # map to Eigen
    cdef MMatrixXd* tmp_C = new MMatrixXd(<mfloat_t* > C.data,C.dimensions[0],C.dimensions[1])
    cdef MMatrixXd* tmp_B = new MMatrixXd(<mfloat_t* > B.data,B.dimensions[0],B.dimensions[1])
    cdef MatrixXi* tmp_extYs = new MatrixXi()
    cdef MatrixXi* tmp_Ys = new MatrixXi()

    # call C++ function
    c_searchYs(tmp_extYs,tmp_Ys,tmp_C,tmp_B,Clo,Chi,oracle)

    # copy results back to python structures
    cdef NP.ndarray extYs = NP.zeros([tmp_extYs.rows(),tmp_extYs.cols()],dtype=int,order="F")
    cdef NP.ndarray Ys = NP.zeros([tmp_Ys.rows(),tmp_Ys.cols()],dtype=int,order="F")
    cdef MMatrixXi* map_extYs = new MMatrixXi(<mint_t* > extYs.data,tmp_extYs.rows(),tmp_extYs.cols())
    cdef MMatrixXi* map_Ys = new MMatrixXi(<mint_t* > Ys.data,tmp_Ys.rows(),tmp_Ys.cols())
    c_copy(map_extYs,tmp_extYs)
    c_copy(map_Ys,tmp_Ys)

    #delete temporary matrices
    del tmp_Ys
    del tmp_extYs
    del tmp_B
    del tmp_C

    return (extYs,Ys)

def searchPattern(ndarray C,ndarray B,double Clo,double Chi,int oracle,int pattern):
    # returns patterns
    # make sure numpy arrays are contiguous
    C = ndarrayF64toC(C)
    B = ndarrayF64toC(B)

    # map to Eigen
    cdef MMatrixXd* tmp_C = new MMatrixXd(<mfloat_t* > C.data,C.dimensions[0],C.dimensions[1])
    cdef MMatrixXd* tmp_B = new MMatrixXd(<mfloat_t* > B.data,B.dimensions[0],B.dimensions[1])
    cdef MatrixXi* tmp_patterns = new MatrixXi()

    # call C++ function
    c_searchPattern(tmp_patterns,tmp_C,tmp_B,Clo,Chi,oracle,pattern)

    # copy results back to python structures
    cdef NP.ndarray patterns = NP.zeros([tmp_patterns.rows(),tmp_patterns.cols()],dtype=int,order="F")
    cdef MMatrixXi* map_patterns = new MMatrixXi(<mint_t* > patterns.data,tmp_patterns.rows(),tmp_patterns.cols())
    c_copy(map_patterns,tmp_patterns)

    #delete temporary matrices
    del tmp_patterns
    del tmp_B
    del tmp_C

    return patterns

def analyse_Y(ndarray data_obs, ndarray data_int, ndarray data_intpos, ndarray B, double Clo, double Chi, int verbose, string struc_csv):
    # print("Entered ystruct.pyx: analyse_Y(...)")
    # returns stats vector
    # analyses performance of (Ext) Y algorithm
    # make sure numpy arrays are contiguous
    data_obs = ndarrayF64toC(data_obs)
    data_int = ndarrayF64toC(data_int)
    data_intpos = ndarrayF64toC(data_intpos)
    B = ndarrayF64toC(B)

    # map to Eigen
    cdef MMatrixXd* tmp_data_obs = new MMatrixXd(<mfloat_t* > data_obs.data,data_obs.dimensions[0],data_obs.dimensions[1])
    cdef MMatrixXd* tmp_data_int = new MMatrixXd(<mfloat_t* > data_int.data,data_int.dimensions[0],data_int.dimensions[1])
    cdef MMatrixXi* tmp_data_intpos = new MMatrixXi(<mint_t* > data_intpos.data,data_intpos.dimensions[0],data_intpos.dimensions[1])
    cdef MMatrixXd* tmp_B = new MMatrixXd(<mfloat_t* > B.data,B.dimensions[0],B.dimensions[1])
    cdef MatrixXd* tmp_stats = new MatrixXd()

    # call C++ function
    c_analyse_Y(tmp_data_obs,tmp_data_int,tmp_data_intpos,tmp_B,tmp_stats,Clo,Chi,verbose,struc_csv)

    # copy results back to python structures
    cdef NP.ndarray stats = NP.zeros([tmp_stats.rows(),tmp_stats.cols()],dtype=float,order="F")
    cdef MMatrixXd* map_stats = new MMatrixXd(<mfloat_t* > stats.data,tmp_stats.rows(),tmp_stats.cols())
    c_copy(map_stats,tmp_stats)

    #delete temporary matrices
    del tmp_stats
    del tmp_B
    del tmp_data_intpos
    del tmp_data_int
    del tmp_data_obs

    return stats


def pairwise_dependences(ndarray B):
    # make sure numpy arrays are contiguous
    B = ndarrayF64toC(B)

    # map to Eigen
    cdef MMatrixXd* tmp_B = new MMatrixXd(<mfloat_t* > B.data,B.dimensions[0],B.dimensions[1])
    cdef MatrixXi* tmp_deps = new MatrixXi()

    # call C++ function
    c_pairwise_dependences(tmp_B,tmp_deps)
    
    # copy results back to python structures
    cdef NP.ndarray deps = NP.zeros([tmp_deps.rows(),tmp_deps.cols()],dtype=int,order="F")
    cdef MMatrixXi* map_deps = new MMatrixXi(<mint_t* > deps.data,tmp_deps.rows(),tmp_deps.cols())
    c_copy(map_deps,tmp_deps)

    #delete temporary matrices
    del tmp_deps

    return deps

def pairwise_independent(ndarray B,int x,int y):
    # make sure numpy arrays are contiguous
    B = ndarrayF64toC(B)

    # map to Eigen
    cdef MMatrixXd* tmp_B = new MMatrixXd(<mfloat_t* > B.data,B.dimensions[0],B.dimensions[1])

    # call C++ function
    b = c_pairwise_independent(tmp_B,x,y)

    #delete temporary matrices
    del tmp_B

    return b

def pairwise_conditionally_independent(ndarray B,int x,int y,int z):
    # make sure numpy arrays are contiguous
    B = ndarrayF64toC(B)

    # map to Eigen
    cdef MMatrixXd* tmp_B = new MMatrixXd(<mfloat_t* > B.data,B.dimensions[0],B.dimensions[1])

    # call C++ function
    b = c_pairwise_conditionally_independent(tmp_B,x,y,z)

    #delete temporary matrices
    del tmp_B

    return b

def ancestor(ndarray B,int x,int y):
    # make sure numpy arrays are contiguous
    B = ndarrayF64toC(B)

    # map to Eigen
    cdef MMatrixXd* tmp_B = new MMatrixXd(<mfloat_t* > B.data,B.dimensions[0],B.dimensions[1])

    # call C++ function
    b = c_ancestor(tmp_B,x,y)

    #delete temporary matrices
    del tmp_B

    return b

def projectADMG(ndarray B,ndarray S,ndarray obsVar,int verbose):
    # returns (projB,projS)
    # make sure numpy arrays are contiguous
    B = ndarrayF64toC(B)
    S = ndarrayF64toC(S)

    # map to Eigen
    cdef MMatrixXd* tmp_B = new MMatrixXd(<mfloat_t* > B.data,B.dimensions[0],B.dimensions[1])
    cdef MMatrixXd* tmp_S = new MMatrixXd(<mfloat_t* > S.data,S.dimensions[0],S.dimensions[1])
    cdef MMatrixXi* tmp_obsVar = new MMatrixXi(<mint_t* > obsVar.data,obsVar.dimensions[0],obsVar.dimensions[1])
    cdef MatrixXd* tmp_projB = new MatrixXd()
    cdef MatrixXd* tmp_projS = new MatrixXd()

    # call C++ function
    c_projectADMG(tmp_B,tmp_S,tmp_obsVar,tmp_projB,tmp_projS,verbose)

    # copy results back to python structures
    cdef NP.ndarray projB = NP.zeros([tmp_projB.rows(),tmp_projB.cols()],dtype=float,order="F")
    cdef NP.ndarray projS = NP.zeros([tmp_projS.rows(),tmp_projS.cols()],dtype=float,order="F")
    cdef MMatrixXd* map_projB = new MMatrixXd(<mfloat_t* > projB.data,tmp_projB.rows(),tmp_projB.cols())
    cdef MMatrixXd* map_projS = new MMatrixXd(<mfloat_t* > projS.data,tmp_projS.rows(),tmp_projS.cols())
    c_copy(map_projB,tmp_projB)
    c_copy(map_projS,tmp_projS)

    #delete temporary matrices
    del tmp_projS
    del tmp_projB

    return (projB,projS)
