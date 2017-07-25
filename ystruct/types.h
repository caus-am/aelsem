#ifndef TYPES_H
#define TYPES_H

#include <eigen3/Eigen/Dense>

// note: for swig it is important that everything is typedef'd and not merely "defined"
typedef double float64_t;
typedef float float32_t;
// typedef long int int64_t;
// typedef unsigned long int uint64_t;

typedef float64_t mfloat_t;
#if !defined(_WIN32)
typedef int64_t mint_t;
typedef uint64_t muint_t;
#else
// (my version of) MSVC doesn't know above, so using these alternatives
typedef __int64 mint_t;
typedef unsigned __int64 muint_t;
#endif

typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXd;
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXdRM;
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXi;
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXi;
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXd;

// maps for python
typedef Eigen::Map<MatrixXi> MMatrixXi;
typedef Eigen::Map<MatrixXd> MMatrixXd;
typedef Eigen::Map<MatrixXdRM> MMatrixXdRM;
typedef Eigen::Map<VectorXd> MVectorXd;

#endif /* TYPES_H */
