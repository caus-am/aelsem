#ifndef YSTRUCT_H
#define YSTRUCT_H


#include "types.h"
#include <iostream>
#include <cmath>
#include <cstdio>
#include <vector>
#include <set>
#include <queue>
#include <string>
#include <fstream>
#include <boost/foreach.hpp>
#include "dai/dag.h"
#include "dai/exceptions.h"


/// set of integers
typedef std::set<size_t> iSet;


/// Calculates partial correlation rho_{xy|z} from Pearson's correlations rho_{xy}, rho_{xz} and rho_{yz}
inline double partialcorr(double Cxy, double Cxz, double Cyz) {
    double Cxy_z = (Cxy - Cxz * Cyz) / sqrt((1.0  - Cxz * Cxz) * (1.0 - Cyz * Cyz));
    return Cxy_z;
}


/// ordered quadruple of integers
struct quadruple {
    size_t i, j, k, l;
    quadruple( size_t _i, size_t _j, size_t _k, size_t _l ) : i(_i), j(_j), k(_k), l(_l) {}
    bool operator< ( const quadruple& uvwx ) const { 
        if( i < uvwx.i )
            return true;
        else if( i == uvwx.i && j < uvwx.j )
            return true;
        else if( i == uvwx.i && j == uvwx.j && k < uvwx.k )
            return true;
        else if( i == uvwx.i && j == uvwx.j && k == uvwx.k && l < uvwx.l )
            return true;
        else
            return false;
    }
    bool operator==( const quadruple& uvwx ) const {
        if( i == uvwx.i && j == uvwx.j && k == uvwx.k && l == uvwx.l )
            return true;
        else
            return false;
    }
    friend std::ostream& operator << ( std::ostream& os, const quadruple& ijkl ) {
        os << "(" << ijkl.i << "," << ijkl.j << "," << ijkl.k << "," << ijkl.l << ")";
        return os;
    }
};


/// ordered triple of integers
struct triple {
    size_t i, j, k;
    triple( size_t _i, size_t _j, size_t _k ) : i(_i), j(_j), k(_k) {}
    bool operator< ( const triple& uvw ) const { 
        if( i < uvw.i )
            return true;
        else if( i == uvw.i && j < uvw.j )
            return true;
        else if( i == uvw.i && j == uvw.j && k < uvw.k )
            return true;
        else
            return false;
    }
    friend std::ostream& operator << ( std::ostream& os, const triple& ijk ) {
        os << "(" << ijk.i << "," << ijk.j << "," << ijk.k << ")";
        return os;
    }
};


/// ordered pair of integers
struct mypair {
    size_t i, j;
    mypair( size_t _i, size_t _j ) : i(_i), j(_j) {}
    bool operator< ( const mypair& uv ) const { 
        if( i < uv.i )
            return true;
        else if( i == uv.i && j < uv.j )
            return true;
        else
            return false;
    }
    friend std::ostream& operator << ( std::ostream& os, const mypair& ij ) {
        os << "(" << ij.i << "," << ij.j << ")";
        return os;
    }
};


/// construct set containing a single integer
inline iSet iSet1( size_t i ) {
    iSet result;
    result.insert( i );
    return result;
}


/// construct set containing two integers
inline iSet iSet2( size_t i, size_t j ) {
    iSet result;
    result.insert( i );
    result.insert( j );
    return result;
}


/// Writes a \c std::set<> to a \c std::ostream
template<class T>
std::ostream& operator << (std::ostream& os, const std::set<T> & x) {
    os << "{";
    for( typename std::set<T>::const_iterator it = x.begin(); it != x.end(); it++ )
        os << (it != x.begin() ? ", " : "") << *it;
    os << "}";
    return os;
}


/// useful for copying matrices between python and C++
template <typename Derived1,typename Derived2>
inline void copy(const Eigen::MatrixBase<Derived1>* out_, const Eigen::MatrixBase<Derived2>* in_)
{
	const_cast< Eigen::MatrixBase<Derived1>& >(*out_) = (*in_);
}


/// check if a set contains a certain element
template <typename T>
inline bool contains( const std::set<T>& S, const T& t ) {
    return S.find( t ) != S.end();
}


/// extended boolean
enum FUT {FALSE=-1, UNKNOWN=0, TRUE=1};


/// extended logical and operator
inline FUT operator&&( const FUT &a, const FUT &b ) {
    FUT ans = UNKNOWN;
    if( a == TRUE && b == TRUE )
        ans = TRUE;
    else if( a == FALSE || b == FALSE )
        ans = FALSE;
    return ans;
}


/// extended logical negation operator
inline FUT operator!( const FUT &a ) {
    FUT ans = UNKNOWN;
    if( a == TRUE )
        ans = FALSE;
    else if( a == FALSE )
        ans = TRUE;
    return ans;
}


/// returns the parent set of node j given adjacency matrix B
/// (slow implementation, using dense adjacency matrix B representation)
template <typename Derived>
inline iSet parents(const Eigen::MatrixBase<Derived>& B, size_t j) {
    size_t nvar = B.rows();
    DAI_ASSERT( B.cols() == nvar );
    DAI_ASSERT( j < nvar );

    iSet pa;
    for( size_t i = 0; i < nvar; i++ )
        if( B(i,j) )
            pa.insert( i );

    return pa;
}


/// returns the children set of node i given adjacency matrix B
/// (slow implementation, using dense adjacency matrix B representation)
template <typename Derived>
inline iSet children(const Eigen::MatrixBase<Derived>& B, size_t i) {
    size_t nvar = B.rows();
    DAI_ASSERT( B.cols() == nvar );
    DAI_ASSERT( i < nvar );

    iSet ch;
    for( size_t j = 0; j < nvar; j++ )
        if( B(i,j) )
            ch.insert( j );

    return ch;
}


/// returns complete set Y such that Y is d-connected with X given Z
/// (slow implementation, using dense adjacency matrix B representation)
template <typename Derived>
inline iSet bayesball(const Eigen::MatrixBase<Derived>& B,const iSet& X,const iSet& Z) {
    using namespace std;

    iSet dconnected;

    size_t nvar = B.rows();
    DAI_ASSERT( (size_t)B.cols() == nvar );
    vector<bool> observed( nvar, false );
    BOOST_FOREACH( size_t z, Z )
        observed[z] = true;

    queue<size_t> LV;
    BOOST_FOREACH( size_t x, X )
        LV.push( x );

    vector<bool> visitedfromchild( nvar, false );
    BOOST_FOREACH( size_t x, X )
        visitedfromchild[x] = 1;
    vector<bool> visitedfromparent( nvar, false );

    vector<bool> topmarked( nvar, false );
    vector<bool> bottommarked( nvar, false );

//    cout << "Bayesball visiting...";
    size_t iters = 0;
    while( !LV.empty() ) {
        iters++;
        size_t j = LV.front();
//        cout << j << " ";
        if( !observed[j] && visitedfromchild[j] ) {
            if( !topmarked[j] ) {
                topmarked[j] = true;
                for( size_t i = 0; i < nvar; i++ )
                    if( B(i,j) ) { // i is parent of j
                        LV.push( i );
                        visitedfromchild[i] = true;
                    }
            }
            if( !bottommarked[j] ) {
                bottommarked[j] = true;
                for( size_t k = 0; k < nvar; k++ )
                    if( B(j,k) ) { // k is child of j
                        LV.push( k );
                        visitedfromparent[k] = true;
                    }
            }
        }
        if( visitedfromparent[j] ) {
            if( observed[j] && !topmarked[j] ) {
                topmarked[j] = true;
                for( size_t i = 0; i < nvar; i++ )
                    if( B(i,j) ) { // i is parent of j
                        LV.push( i );
                        visitedfromchild[i] = true;
                    }
            }
            if( !observed[j] && !bottommarked[j] ) {
                bottommarked[j] = true;
                for( size_t k = 0; k < nvar; k++ )
                    if( B(j,k) ) { // k is child of j
                        LV.push( k );
                        visitedfromparent[k] = true;
                    }
            }
        }
        LV.pop();
    }
//    cout << endl;
//    cout << "Bayesball needed " << iters << " iterations." << endl;

    for( size_t i = 0; i < nvar; i++ )
        if( bottommarked[i] && !contains(X,i) )
            dconnected.insert( i );

    return dconnected;
}


/// returns true if x is independent of y, in a DAG B
/// (slow implementation, using dense adjacency matrix B representation)
template <typename Derived>
inline bool pairwise_independent(const Eigen::MatrixBase<Derived>* B_,size_t x,size_t y) { // x indep y?
    Eigen::MatrixBase<Derived>& B = const_cast< Eigen::MatrixBase<Derived>& >(*B_);
    return !contains(bayesball(B,iSet1(x),iSet()),y);
}


/// returns true if x is independent of y given z, in a DAG B
/// (slow implementation, using dense adjacency matrix B representation)
template <typename Derived>
inline bool pairwise_conditionally_independent(const Eigen::MatrixBase<Derived>* B_,size_t x,size_t y,size_t z) { // x indep y given z?
    Eigen::MatrixBase<Derived>& B = const_cast< Eigen::MatrixBase<Derived>& >(*B_);
    return !contains(bayesball(B,iSet1(x),iSet1(z)),y);
}


/// returns true if x is ancestor of y, in a DAG B
/// (semislow implementation, converts dense adjacency matrix B to sparse DAG representation)
template <typename Derived>
inline bool ancestor(const Eigen::MatrixBase<Derived>* B_,size_t x,size_t y) { // x ancestor of y?
    Eigen::MatrixBase<Derived>& B = const_cast< Eigen::MatrixBase<Derived>& >(*B_);

    size_t n1 = B.rows();
    size_t n2 = B.cols();
    DAI_ASSERT( n1 == n2 );
    size_t N = n1;

    // convert to DAG representation for speed
    dai::DAG G(N);
    for( size_t i = 0; i < N; i++ )
        for( size_t j = 0; j < N; j++ )
            if( B(i,j) != 0 )
                G.addEdge( i, j, true );

    return G.existsDirectedPath( x, y );
}


/// returns complete set Y such that Y is d-connected with X given Z
/// (fast implementation using dai::DAG)
inline iSet bayesball(const dai::DAG& B,const iSet& X,const iSet& Z) {
    using namespace std;

    // returns set Y such that Y is d-connected with X given Z
    iSet dconnected;

    size_t nvar = B.nrNodes();
    vector<bool> observed( nvar, false );
    BOOST_FOREACH( size_t z, Z )
        observed[z] = true;

    queue<size_t> LV;
    BOOST_FOREACH( size_t x, X )
        LV.push( x );

    vector<bool> visitedfromchild( nvar, false );
    BOOST_FOREACH( size_t x, X )
        visitedfromchild[x] = 1;
    vector<bool> visitedfromparent( nvar, false );

    vector<bool> topmarked( nvar, false );
    vector<bool> bottommarked( nvar, false );

//    cout << "Bayesball visiting...";
    size_t iters = 0;
    while( !LV.empty() ) {
        iters++;
        size_t j = LV.front();
//        cout << j << " ";
        if( !observed[j] && visitedfromchild[j] ) {
            if( !topmarked[j] ) {
                topmarked[j] = true;
                BOOST_FOREACH( const dai::Neighbor &i, B.pa( j ) ) { // i is parent of j
                    LV.push( i.node );
                    visitedfromchild[i.node] = true;
                }
            }
            if( !bottommarked[j] ) {
                bottommarked[j] = true;
                BOOST_FOREACH( const dai::Neighbor &k, B.ch( j ) ) { // k is child of j
                    LV.push( k.node );
                    visitedfromparent[k.node] = true;
                }
            }
        }
        if( visitedfromparent[j] ) {
            if( observed[j] && !topmarked[j] ) {
                topmarked[j] = true;
                BOOST_FOREACH( const dai::Neighbor &i, B.pa( j ) ) { // i is parent of j
                    LV.push( i.node );
                    visitedfromchild[i.node] = true;
                }
            }
            if( !observed[j] && !bottommarked[j] ) {
                bottommarked[j] = true;
                BOOST_FOREACH( const dai::Neighbor &k, B.ch( j ) ) { // k is child of j
                    LV.push( k.node );
                    visitedfromparent[k.node] = true;
                }
            }
        }
        LV.pop();
    }
//    cout << endl;

    for( size_t i = 0; i < nvar; i++ )
        if( bottommarked[i] && !contains(X,i) )
            dconnected.insert( i );

    return dconnected;
}


/// returns true if x is independent of y, in a DAG G
/// (fast implementation using dai::DAG)
inline bool pairwise_independent(const dai::DAG& G,size_t x,size_t y) { // x indep y?
    return !contains(bayesball(G,iSet1(x),iSet()),y);
}


/// returns true if x is independent of y given z, in a DAG G
/// (fast implementation using dai::DAG)
inline bool pairwise_conditionally_independent(const dai::DAG& G,size_t x,size_t y,size_t z) { // x indep y given z?
    return !contains(bayesball(G,iSet1(x),iSet1(z)),y);
}


/// returns true if x is independent of y given {z0,z1}, in a DAG G
/// (fast implementation using dai::DAG)
inline bool pairwise_conditionally_independent(const dai::DAG& G,size_t x,size_t y,size_t z0,size_t z1) { // x indep y given {z0,z1}?
    return !contains(bayesball(G,iSet1(x),iSet2(z0,z1)),y);
}


/// returns matrix of all pairwise dependencies
/// (semislow implementation, converts dense adjacency matrix B to sparse DAG representation)
template <typename Derived1, typename Derived2>
inline void pairwise_dependences(const Eigen::MatrixBase<Derived1>* B_,const Eigen::MatrixBase<Derived2>* deps_) { 
    // recast Eigen output to writeable element.
    Eigen::MatrixBase<Derived1>& B = const_cast< Eigen::MatrixBase<Derived1>& >(*B_);
    Eigen::MatrixBase<Derived2>& Mdeps = const_cast< Eigen::MatrixBase<Derived2>& >(*deps_);

    size_t n1 = B.rows();
    size_t n2 = B.cols();
    DAI_ASSERT( n1 == n2 );
    size_t N = n1;

    // convert to DAG representation for speed
    dai::DAG G(N);
    for( size_t i = 0; i < N; i++ )
        for( size_t j = 0; j < N; j++ )
            if( B(i,j) != 0 )
                G.addEdge( i, j, true );

    // prepare output
    Mdeps = MatrixXi::Zero(N,N);

    // call Bayes ball
    for( size_t i = 0; i < N; i++ ) {
        iSet d_connected_i = bayesball(G,iSet1(i),iSet());
        Mdeps(i,i) = 1;
        BOOST_FOREACH( size_t j, d_connected_i ) {
            Mdeps(i,j) = 1;
        }
        
    }
}


/// contingency table, used to count statistics for validation purposes
struct ConTab {
    size_t TP; // true  positives
    size_t FP; // false positives
    size_t TN; // true  negatives
    size_t FN; // false negatives
    size_t UP; // unknown positives
    size_t UN; // unknown negatives
    ConTab() : TP(0), FP(0), TN(0), FN(0), UP(0), UN(0) {}
    void update( FUT estval, FUT trueval ) {
        DAI_ASSERT( trueval != UNKNOWN );
        if( estval == TRUE ) {
            if( trueval == TRUE )
                TP++;
            else
                FP++;
        } else if( estval == FALSE ) {
            if( trueval == TRUE )
                FN++;
            else
                TN++;
        } else {
            if( trueval == TRUE )
                UP++;
            else
                UN++;
        }
    }
    double precision() const {
        return ((1.0 * TP) / (TP + FP));
    }
    double recall() const {
        return ((1.0 * TP) / (TP + FN + UP));
    }
    ConTab negate() const {
        ConTab result;
        result.TP = TN;
        result.FP = FN;
        result.TN = TP;
        result.FN = FP;
        result.UP = UN;
        result.UN = UP;
        return result;
    }
    friend std::ostream& operator << ( std::ostream& os, const ConTab& ct ) {
        return( os << "TP=" << ct.TP << ", FP=" << ct.FP << ", TN=" << ct.TN << ", FN=" << ct.FN << ", UP=" << ct.UP << ", UN=" << ct.UN );
    }
};


/// searches (extended) Y structures
/// NOT USED CURRENTLY
template <typename Derived1,typename Derived2,typename Derived3,typename Derived4>
inline int searchYs(const Eigen::MatrixBase<Derived1>* extYs_,const Eigen::MatrixBase<Derived2>* Ys_,const Eigen::MatrixBase<Derived3>* C_,const Eigen::MatrixBase<Derived4>* B_,double Clo,double Chi,int oracle)
{
    using namespace std;

    size_t verbose = 0;

    // recast Eigen output to writeable element.
    Eigen::MatrixBase<Derived1>& MextYs = const_cast< Eigen::MatrixBase<Derived1>& >(*extYs_);
    Eigen::MatrixBase<Derived2>& MYs = const_cast< Eigen::MatrixBase<Derived2>& >(*Ys_);
    Eigen::MatrixBase<Derived3>& C = const_cast< Eigen::MatrixBase<Derived3>& >(*C_);
    Eigen::MatrixBase<Derived4>& B = const_cast< Eigen::MatrixBase<Derived4>& >(*B_);

    size_t n1 = C.rows();
    size_t n2 = C.cols();
    DAI_ASSERT( n1 == n2 );
    size_t N = n1;
    DAI_ASSERT( (size_t)B.cols() == N );

    // convert to DAG representation for speed
    dai::DAG G(N);
    for( size_t i = 0; i < N; i++ )
        for( size_t j = 0; j < N; j++ )
            if( B(i,j) != 0 )
                G.addEdge( i, j, true );

    vector<set<size_t> > nonancestors( N );
    vector<set<size_t> > ancestors( N );
    vector<set<mypair> > possibledescendants( N );
    set<triple> mindeps;
    set<triple> minindeps;

//  find minimal conditional dependence i \notindep j \given [k]
    if( !oracle ) {
        for( size_t i = 0; i < N; i++ ) {
            for( size_t j = i + 1; j < N; j++ ) { // all unordered pairs {i,j}
                if( abs(C(i,j)) < Clo ) { // independence i \indep j
                    for( size_t k = 0; k < N; k++ ) {
                        if( k != i && k != j ) { // and a third node k
                            double Cij_k = partialcorr(C(i,j), C(i,k), C(j,k));
                            if( abs(Cij_k) >= Chi ) { // dependence i \notindep j \given k
                                if( verbose )
                                    cout << "minimal dependence i notindep j given [k]: (" << i << "," << j << "," << k << "); Cij = " << C(i,j) << ", Cij_k = " << Cij_k << endl;
    //                          this means k is not ancestor of i, nor of j
                                mindeps.insert( triple(i,j,k) );
                                mindeps.insert( triple(j,i,k) );
                                nonancestors[i].insert( k );
                                nonancestors[j].insert( k );
    //                          mindeps[(i,j,k)] = (C(i,j), Cij_k)
    //                          conf_k = abs(Cij_k) - abs(C(i,j)) // some measure of our confidence
    /*                          # add these two nonancestor relationships to our set and store the highest confidence
                                if not (k,i) in nonancestors:
                                    nonancestors[(k,i)] = conf_k
                                else:
                                    nonancestors[(k,i)] = max(conf_k, nonancestors[(k,i)])
                                if not (k,j) in nonancestors:
                                    nonancestors[(k,j)] = conf_k
                                else:
                                    nonancestors[(k,j)] = max(conf_k, nonancestors[(k,j)])*/
                            }
                        }
                    }
                }
            }
        }
    } else {
        for( size_t i = 0; i < N; i++ ) {
            for( size_t j = i + 1; j < N; j++ ) { // all unordered pairs {i,j}
                if( pairwise_independent( G, i, j ) ) { // independence i \indep j
                    for( size_t k = 0; k < N; k++ ) {
                        if( k != i && k != j ) { // and a third node k
                            if( !pairwise_conditionally_independent( G, i, j, k ) ) { // dependence i \notindep j \given k
                                if( verbose )
                                    cout << "minimal dependence i notindep j given [k]: (" << i << "," << j << "," << k << ")" << endl;
    //                          this means k is not ancestor of i, nor of j
                                mindeps.insert( triple(i,j,k) );
                                mindeps.insert( triple(j,i,k) );
                                nonancestors[i].insert( k );
                                nonancestors[j].insert( k );
                            }
                        }
                    }
                }
            }
        }
    }
    if( verbose )
        cout << "Found " << mindeps.size() << " smallest minimal dependences" << endl;
/*    Mmindeps = MatrixXi(mindeps.size(), 3);
    size_t row = 0;
    for( set<triple>::const_iterator it = mindeps.begin(); it != mindeps.end(); it++ ) {
        Mmindeps(row,0) = it->i;
        Mmindeps(row,1) = it->j;
        Mmindeps(row,2) = it->k;
        row++;
    }*/

//  find minimal conditional independence i \indep j \given [k]
    if( !oracle ) {
        for( size_t i = 0; i < N; i++ ) {
            for( size_t j = i + 1; j < N; j++ ) { // all unordered pairs {i,j}
                if( abs(C(i,j)) >= Chi ) { // dependence i \notindep j
                    for( size_t k = 0; k < N; k++ ) {
                        if( k != i && k != j ) { // and a third node k
                            double Cij_k = partialcorr(C(i,j), C(i,k), C(j,k));
                            if( abs(Cij_k) < Clo ) { // independence i \indep j \given k
                                if( verbose )
                                    cout << "minimal independence i indep j given [k]: (" << i << "," << j << "," << k << "); Cij = " << C(i,j) << ", Cij_k = " << Cij_k << endl;
    //                          this means k is ancestor of i or of j
                                minindeps.insert( triple(i,j,k) );
                                minindeps.insert( triple(j,i,k) );
                                possibledescendants[k].insert( mypair(i,j) );
    //                          conf_k = abs(C(i,j)) - abs(Cij_k) // some measure of our confidence
                            }
                        }
                    }
                }
            }
        }
    } else {
        for( size_t i = 0; i < N; i++ ) {
            for( size_t j = i + 1; j < N; j++ ) { // all unordered pairs {i,j}
                if( !pairwise_independent( G, i, j ) ) { // dependence i \notindep j
                    for( size_t k = 0; k < N; k++ ) {
                        if( k != i && k != j ) { // and a third node k
                            if( pairwise_conditionally_independent( G, i, j, k ) ) { // independence i \indep j \given k
                                if( verbose )
                                    cout << "minimal independence i indep j given [k]: (" << i << "," << j << "," << k << ")" << endl;
    //                          this means k is ancestor of i or of j
                                minindeps.insert( triple(i,j,k) );
                                minindeps.insert( triple(j,i,k) );
                                possibledescendants[k].insert( mypair(i,j) );
    //                          conf_k = abs(C(i,j)) - abs(Cij_k) // some measure of our confidence
                            }
                        }
                    }
                }
            }
        }
    }
    if( verbose )
        cout << "Found " << minindeps.size() << " smallest minimal independences" << endl;
/*    Mminindeps = MatrixXi(minindeps.size(), 3);
    row = 0;
    for( set<triple>::const_iterator it = minindeps.begin(); it != minindeps.end(); it++ ) {
        Mminindeps(row,0) = it->i;
        Mminindeps(row,1) = it->j;
        Mminindeps(row,2) = it->k;
        row++;
    }*/

    set<quadruple> extYs, Ys;
    // new algorithm!
    if( verbose )
        cout << "Now looking for admissible sets...be patient" << endl;
    for( set<triple>::const_iterator it_wyx = minindeps.begin(); it_wyx != minindeps.end(); it_wyx++ ) {
        for( set<triple>::const_iterator it_wux = mindeps.begin(); it_wux != mindeps.end(); it_wux++ ) {
            if( it_wyx->i == it_wux->i && it_wyx->k == it_wux-> k && it_wyx->j != it_wux->j ) {
                size_t w = it_wyx->i;
                size_t y = it_wyx->j;
                size_t x = it_wyx->k;
                size_t u = it_wux->j;
                if( !oracle && verbose ) {
                    double Cyw_x = partialcorr(C(y,w), C(y,x), C(w,x));
                    double Cuw_x = partialcorr(C(u,w), C(u,x), C(w,x));
                    cout << "(u,w,x,y) = (" << u << "," << w << "," << x << "," << y << ")" << endl;
                    cout << "C(w,y) = " << C(w,y) << " (should be large)" << endl;
                    cout << "C(w,y|x) = " << Cyw_x << " (should be small)" << endl;
                    cout << "C(w,u) = " << C(w,u) << " (should be small)" << endl;
                    cout << "C(w,u|x) = " << Cuw_x << " (should be large)" << endl;
                }
                extYs.insert( quadruple(x,y,u,w) );
                if( verbose )
                    cout << "X causes Y and we don't need to adjust for covariates (via: U,W): (" << x << "," << y << "," << u << ", " << w << ")" << endl;
                
                // additional check for "real" (MAG) Y-structure
                for( set<triple>::const_iterator it_uyx = minindeps.begin(); it_uyx != minindeps.end(); it_uyx++ ) {
                    if( it_uyx->i == u && it_uyx->j == y && it_uyx->k == x ) {
                        if( !oracle && verbose ) {
                            double Cyu_x = partialcorr(C(y,u), C(y,x), C(u,x));
                            cout << "Extra test:" << endl;
                            cout << "C(u,y) = " << C(u,y) << " (should be large)" << endl;
                            cout << "C(u,y|x) = " << Cyu_x << " (should be small)" << endl;
                        }
                        Ys.insert( quadruple(x,y,u,w) );
                    }
                }
            }
        }
    }

    MextYs = MatrixXi(extYs.size(), 4);
    size_t row = 0;
    for( set<quadruple>::const_iterator it = extYs.begin(); it != extYs.end(); it++ ) {
        MextYs(row,0) = it->i;
        MextYs(row,1) = it->j;
        MextYs(row,2) = it->k;
        MextYs(row,3) = it->l;
        row++;
    }

    MYs = MatrixXi(Ys.size(), 4);
    row = 0;
    for( set<quadruple>::const_iterator it = Ys.begin(); it != Ys.end(); it++ ) {
        MYs(row,0) = it->i;
        MYs(row,1) = it->j;
        MYs(row,2) = it->k;
        MYs(row,3) = it->l;
        row++;
    }
   
    return 0;
}


/// returns column-wise mean of a matrix
inline Eigen::VectorXd mean( const Eigen::MatrixXd& X ) {
    using namespace Eigen;
    return X.colwise().mean();
}


/// returns covariance matrix
inline Eigen::MatrixXd cov( const Eigen::MatrixXd& X ) {
    using namespace Eigen; // THIJS: MSVC 'ambiguous symbol MatrixXd'
    Eigen::MatrixXd centered = X.rowwise() - X.colwise().mean();
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(X.rows() - 1);
    return cov;
}


/// returns correlation matrix
inline Eigen::MatrixXd corr( const Eigen::MatrixXd& X ) {
    using namespace Eigen; // THIJS: MSVC 'ambiguous symbol MatrixXd'
    Eigen::MatrixXd C = cov(X);
    Eigen::VectorXd d = C.diagonal();
    Eigen::VectorXd invstd = d.array().sqrt().inverse();
    return invstd.asDiagonal() * C * invstd.asDiagonal();
}


/// evaluates average L1 and L2 error using observational and interventional data
/// for the prediction E(Y|X) = E(Y|do X)
template<typename Derived1, typename Derived2>
inline void evalL1L2Err( const Eigen::MatrixXd& data_obs_cov, const Eigen::VectorXd& data_obs_mean, const Eigen::MatrixBase<Derived1>* data_int_, const Eigen::MatrixBase<Derived2>* data_intpos_, size_t x, size_t y, size_t verbose, size_t& ErrCount, double& ErrL1, double& ErrL2 ) {
    using namespace std;
    Eigen::MatrixBase<Derived1>& data_int = const_cast< Eigen::MatrixBase<Derived1>& >(*data_int_);
    Eigen::MatrixBase<Derived2>& data_intpos = const_cast< Eigen::MatrixBase<Derived2>& >(*data_intpos_);
    ErrL1 = 0.0;
    ErrL2 = 0.0;
    ErrCount = 0;
    double slope = data_obs_cov(x,y) / data_obs_cov(x,x);
    double intercept = data_obs_mean(y) - slope * data_obs_mean(x);
    for( size_t i = 0; i < (size_t)data_int.rows(); i++ )
        if( (size_t)data_intpos(i) == x ) {
            double e = abs(data_int(i,y) - (data_int(i,x)*slope + intercept));
            if( verbose >= 2 )
                cout << "Pair (" << x << ", " << y << "): Error: " << e << endl;
            ErrL1 += e;
            ErrL2 += e*e;
            ErrCount++;
        }
}


/// evaluates baseline L1 and L2 errors using observational and interventional data
template<typename Derived1, typename Derived2>
inline void evalBaseline( const Eigen::MatrixXd& data_obs_cov, const Eigen::VectorXd& data_obs_mean, const Eigen::MatrixBase<Derived1>* data_int_, const Eigen::MatrixBase<Derived2>* data_intpos_, size_t verbose, size_t& ErrCount, double& ErrL1, double& ErrL2, size_t which ) {
    using namespace std;
    Eigen::MatrixBase<Derived1>& data_int = const_cast< Eigen::MatrixBase<Derived1>& >(*data_int_);
    Eigen::MatrixBase<Derived2>& data_intpos = const_cast< Eigen::MatrixBase<Derived2>& >(*data_intpos_);
    // baseline
    if (verbose >= 5)
        cout << "evalBaseline(...): const_casts complete" << endl;
    ErrCount = 0;
    ErrL1 = 0.0;
    ErrL2 = 0.0;
    for( size_t i = 0; i < (size_t)data_int.rows(); i++ ) {
        if (verbose >= 5)
            cout << i;
        size_t x = data_intpos(i);
        for( size_t y = 0; y < (size_t)data_int.cols(); y++ ) {
            if (verbose >= 5)
                cout << '[' << x << ',' << y << ']'
                     << data_obs_cov(x,y) << ',' << data_obs_cov(x,x)
                     << endl;
            // Crash caused by -9.80101e+303 / 1.11262e-306
            double slope = data_obs_cov(x,y) / data_obs_cov(x,x);
            double intercept = data_obs_mean(y) - slope * data_obs_mean(x);
            double e = 0.0;
            if (verbose >= 5)
                cout << ';';
            if( which == 1 )
                e = abs(data_int(i,y) - data_obs_mean(y));
            else if( which == 2 )
                e = abs(data_int(i,y) - (data_int(i,x)*slope + intercept));
            ErrL1 += e;
            ErrL2 += e*e;
            ErrCount++;
            if (verbose >= 5)
                cout << ')';
        }
    }
    if (verbose >= 5)
        cout << '.' << endl;
}


/// calculates partial correlation rho_{ij|k} from correlation matrix C
inline double pcorr( const Eigen::MatrixXd& C, size_t i, size_t j, size_t k ) {
    return partialcorr(C(i,j), C(i,k), C(j,k));
}


/// abstract base class for performing (conditional) independence tests
class indepTest {
    public:
        /// virtual destructor
        virtual ~indepTest() {}
        /// virtual copy-constructor
        virtual indepTest* clone() const = 0;
        /// test pairwise independence Xi _||_ Xj
        virtual FUT test_indep( size_t i, size_t j ) const = 0;
        /// test conditional independence Xi _||_ Xj | Xk
        virtual FUT test_condindep( size_t i, size_t j, size_t k ) const = 0;
        /// test conditional independence Xi _||_ Xj | {Xk, Xl}
        virtual FUT test_condindep( size_t i, size_t j, size_t k, size_t l ) const = 0;
        /// test conditional independence Xi _||_ Xj | XK
        virtual FUT test_condindep( size_t i, size_t j, const iSet& K ) const = 0;

        /// test minimal independence Xi || Xj | [Xk]
        FUT test_minindep( const triple& ijk ) const {
            // return( !test_indep( ijk.i, ijk.j ) && test_condindep( ijk.i, ijk.j, ijk.k ) );

            // optimized && operator:
            FUT a = !test_indep( ijk.i, ijk.j );
            if( a == FALSE )
                return a;
            else
                return( a && test_condindep( ijk.i, ijk.j, ijk.k ) );
        }
        /// test for minimal dependence Xi _/||/_ Xj | [Xk]
        FUT test_mindep( const triple& ijk ) const {
            // return( test_indep( ijk.i, ijk.j ) && !test_condindep( ijk.i, ijk.j, ijk.k ) );

            // optimized && operator:
            FUT a = test_indep( ijk.i, ijk.j );
            if( a == FALSE )
                return a;
            else
                return( a && !test_condindep( ijk.i, ijk.j, ijk.k ) );
        }
        /// test for extended Y structure
        FUT test_extY( const quadruple& xyuz ) const {
            size_t x = xyuz.i;
            size_t y = xyuz.j;
            size_t u = xyuz.k;
            size_t z = xyuz.l;
            // return test_minindep( triple( z, y, x ) ) && test_mindep( triple( z, u, x ) );

            // optimized && operator:
            FUT a = test_minindep( triple( z, y, x ) );
            if( a == FALSE )
                return a;
            else
                return( a && test_mindep( triple( z, u, x ) ) );
        }
        /// test for Y structure
        FUT test_Y( const quadruple& xyuz ) const {
            size_t x = xyuz.i;
            size_t y = xyuz.j;
            size_t u = xyuz.k;
            // size_t z = xyuz.l;
            // return test_minindep( triple( z, y, x ) ) && test_minindep( triple( u, y, x ) ) && test_mindep( triple( z, u, x ) );

            // return test_extY( xyuz ) && test_minindep( triple( u, y, x ) );

            // optimized && operator:
            FUT a = test_extY( xyuz );
            if( a == FALSE )
                return a;
            else
                return( a && test_minindep( triple( u, y, x ) ) );
        }
        /// test for Y structure and all corresponding "redundant" tests up to conditioning set size 1
        FUT test_Y1( const quadruple& xyuz ) const {
            size_t x = xyuz.i;
            size_t y = xyuz.j;
            size_t u = xyuz.k;
            size_t z = xyuz.l;
/*            return test_Y( xyuz )
                && !test_indep( x, y ) 
                && !test_indep( x, u ) 
                && !test_indep( x, z ) 
        //      && !test_indep( y, u ) 
        //      && !test_indep( y, z ) 
        //      && test_indep( u, z ) 
                && !test_condindep( x, y, u ) 
                && !test_condindep( x, y, z ) 
                && !test_condindep( x, u, y ) 
                && !test_condindep( x, u, z ) 
                && !test_condindep( x, z, y ) 
                && !test_condindep( x, z, u ) 
        //      && test_condindep( y, u, x ) 
                && !test_condindep( y, u, z ) 
        //      && test_condindep( y, z, x ) 
                && !test_condindep( y, z, u ) 
        //      && !test_condindep( u, z, x ) 
                && !test_condindep( u, z, y );*/
            FUT a = test_Y( xyuz );
            if( a == FALSE )
                return a;
            a = a && !test_indep( x, y );
            if( a == FALSE )
                return a;
            a = a && !test_indep( x, u );
            if( a == FALSE )
                return a;
            a = a && !test_indep( x, z );
            if( a == FALSE )
                return a;
            a = a && !test_condindep( x, y, u );
            if( a == FALSE )
                return a;
            a = a && !test_condindep( x, y, z );
            if( a == FALSE )
                return a;
            a = a && !test_condindep( x, u, y );
            if( a == FALSE )
                return a;
            a = a && !test_condindep( x, u, z );
            if( a == FALSE )
                return a;
            a = a && !test_condindep( x, z, y );
            if( a == FALSE )
                return a;
            a = a && !test_condindep( x, z, u );
            if( a == FALSE )
                return a;
            a = a && !test_condindep( y, u, z );
            if( a == FALSE )
                return a;
            a = a && !test_condindep( y, z, u );
            if( a == FALSE )
                return a;
            a = a && !test_condindep( u, z, y );
            return a;
        }
        FUT test_Y2( const quadruple& xyuz ) const {
            size_t x = xyuz.i;
            size_t y = xyuz.j;
            size_t u = xyuz.k;
            size_t z = xyuz.l;
/*            return test_Y1( xyuz )
                && test_condindep( u, y, x, z )
                && test_condindep( z, y, x, u )
                && !test_condindep( x, y, u, z ) 
                && !test_condindep( x, u, y, z ) 
                && !test_condindep( x, z, y, u ) 
                && !test_condindep( u, z, x, y ); */
            FUT a = test_Y1( xyuz );
            if( a == FALSE )
                return a;
            a = a && test_condindep( u, y, x, z );
            if( a == FALSE )
                return a;
            a = a && test_condindep( z, y, x, u );
            if( a == FALSE )
                return a;
            a = a && !test_condindep( x, y, u, z );
            if( a == FALSE )
                return a;
            a = a && !test_condindep( x, u, y, z );
            if( a == FALSE )
                return a;
            a = a && !test_condindep( x, z, y, u );
            if( a == FALSE )
                return a;
            a = a && !test_condindep( u, z, x, y );
            return a;
        }
};


/// oracle (conditional) independence testing
class indepTestOracle : public indepTest {
    private:
        dai::DAG G;

    public:
        /// construct from DAG
        indepTestOracle( const dai::DAG& _G ) : G(_G) {}
        /// virtual copy-constructor
        virtual indepTestOracle* clone() const { return new indepTestOracle(*this); }
        /// test pairwise independence Xi _||_ Xj
        FUT test_indep( size_t i, size_t j ) const {
            return !contains(bayesball(G,iSet1(i),iSet()),j) ? TRUE : FALSE;
        }
        /// test conditional independence Xi _||_ Xj | Xk
        FUT test_condindep( size_t i, size_t j, size_t k ) const {
            return !contains(bayesball(G,iSet1(i),iSet1(k)),j) ? TRUE : FALSE;
        }
        /// test conditional independence Xi _||_ Xj | {Xk, Xl}
        FUT test_condindep( size_t i, size_t j, size_t k, size_t l ) const {
            return !contains(bayesball(G,iSet1(i),iSet2(k,l)),j) ? TRUE : FALSE;
        }
        /// test conditional independence Xi _||_ Xj | XK
        FUT test_condindep( size_t i, size_t j, const iSet& K ) const {
            return !contains(bayesball(G,iSet1(i),K),j) ? TRUE : FALSE;
        }
};


/// (conditional) independence testing using partial correlations
class indepTestPCor : public indepTest {
    private:
        const Eigen::MatrixXd& C;
        double Clo;
        double Chi;

    public:
        /// construct from correlation matrix
        indepTestPCor( const Eigen::MatrixXd& _C, double _Clo, double _Chi ) : C(_C), Clo(_Clo), Chi(_Chi) {}
        /// virtual copy-constructor
        virtual indepTestPCor* clone() const { return new indepTestPCor(*this); }
        /// test pairwise independence Xi _||_ Xj
        FUT test_indep( size_t i, size_t j ) const {
            using namespace std;
            double Cij = C(i,j);
            FUT ans = UNKNOWN;
            if( abs(Cij) > Chi ) 
                ans = FALSE;
            else if( abs(Cij) < Clo )
                ans = TRUE;
            return ans;
        }
        /// test conditional independence Xi _||_ Xj | Xk
        FUT test_condindep( size_t i, size_t j, size_t k ) const {
            using namespace std;
            double Cij_k = partialcorr(C(i,j), C(i,k), C(j,k));
            FUT ans = UNKNOWN;
            if( abs(Cij_k) > Chi ) 
                ans = FALSE;
            else if( abs(Cij_k) < Clo )
                ans = TRUE;
            return ans;
        }
        /// test conditional independence Xi _||_ Xj | {Xk, Xl}
        FUT test_condindep( size_t i, size_t j, size_t k, size_t l ) const {
            // tests i _||_ j | {k,l}
            using namespace std;
            double Cij_l = pcorr(C, i, j, l);
            double Cik_l = pcorr(C, i, k, l);
            double Cjk_l = pcorr(C, j, k, l);
            double Cij_kl = (Cij_l - Cik_l * Cjk_l) / sqrt((1.0  - Cik_l*Cik_l) * (1.0 - Cjk_l*Cjk_l));

            FUT ans = UNKNOWN;
            if( abs(Cij_kl) > Chi ) 
                ans = FALSE;
            else if( abs(Cij_kl) < Clo )
                ans = TRUE;
            return ans;
        }
        /// test conditional independence Xi _||_ Xj | XK
        FUT test_condindep( size_t i, size_t j, const iSet& K ) const {
            // not implemented yet
            DAI_ASSERT( 0 == 1 );
        }
};


template <typename Derived1,typename Derived2,typename Derived3>
inline int searchPattern(const Eigen::MatrixBase<Derived1>* patterns_,const Eigen::MatrixBase<Derived2>* C_,const Eigen::MatrixBase<Derived3>* B_,double Clo,double Chi,int oracle,int pattern)
{
    using namespace std;

    size_t verbose = 0;

    // recast Eigen output to writeable element.
    Eigen::MatrixBase<Derived1>& Mpatterns = const_cast< Eigen::MatrixBase<Derived1>& >(*patterns_);
    Eigen::MatrixBase<Derived2>& C = const_cast< Eigen::MatrixBase<Derived2>& >(*C_);
    Eigen::MatrixBase<Derived3>& B = const_cast< Eigen::MatrixBase<Derived3>& >(*B_);

    size_t n1 = C.rows();
    size_t n2 = C.cols();
    DAI_ASSERT( n1 == n2 );
    size_t N = n1;
    DAI_ASSERT( (size_t)B.cols() == N );

    // convert to DAG representation for speed
    dai::DAG G(N);
    for( size_t i = 0; i < N; i++ )
        for( size_t j = 0; j < N; j++ )
            if( B(i,j) != 0 )
                G.addEdge( i, j, true );

    indepTest *T;
    if( oracle )
        T = new indepTestOracle( G );
    else
        T = new indepTestPCor( C, Clo, Chi );

    set<quadruple> patterns;
    if( verbose )
        cout << "Now looking for patterns...be patient" << endl;
    for( size_t x = 0; x < N; x++ ) {
        for( size_t y = 0; y < N; y++ ) {
            if( y != x ) {
                for( size_t u = 0; u < N; u++ ) {
                    if( u != x && u != y ) {
                        for( size_t z = 0; z < N; z++ ) {
                            if( z != x && z != y && z != u ) {
                                quadruple xyuz( x, y, u, z );
                                FUT test = FALSE;
                                if( pattern == 0 )
                                    test = T->test_extY( xyuz );
                                else if( pattern == 1 )
                                    test = T->test_Y( xyuz );
                                else if( pattern == 2 )
                                    test = T->test_Y1( xyuz );
                                else if( pattern == 3 )
                                    test = T->test_Y2( xyuz );
                                if( test == TRUE )
                                    patterns.insert( xyuz );
                            }
                        }
                    }
                }
            }
        }
    }
                                
    Mpatterns = MatrixXi(patterns.size(), 4);
    size_t row = 0;
    for( set<quadruple>::const_iterator it = patterns.begin(); it != patterns.end(); it++ ) {
        Mpatterns(row,0) = it->i;
        Mpatterns(row,1) = it->j;
        Mpatterns(row,2) = it->k;
        Mpatterns(row,3) = it->l;
        row++;
    }

    return 0;
}


template <typename Derived1,typename Derived2,typename Derived3,typename Derived4>
inline int analyse_Y(const Eigen::MatrixBase<Derived1>* data_obs_,const Eigen::MatrixBase<Derived1>* data_int_,const Eigen::MatrixBase<Derived2>* data_intpos_, const Eigen::MatrixBase<Derived3>* B_,const Eigen::MatrixBase<Derived4>* stats_,double Clo,double Chi,size_t verbose,std::string struc_csv)
{
    // calculate statistics for Y-structure algorithm
    using namespace std;

    // recast Eigen output to writeable element.
    Eigen::MatrixBase<Derived1>& data_obs = const_cast< Eigen::MatrixBase<Derived1>& >(*data_obs_);
    Eigen::MatrixBase<Derived1>& data_int = const_cast< Eigen::MatrixBase<Derived1>& >(*data_int_);
    Eigen::MatrixBase<Derived2>& data_intpos = const_cast< Eigen::MatrixBase<Derived2>& >(*data_intpos_);
    Eigen::MatrixBase<Derived3>& B = const_cast< Eigen::MatrixBase<Derived3>& >(*B_);
    Eigen::MatrixBase<Derived4>& stats = const_cast< Eigen::MatrixBase<Derived4>& >(*stats_);

    // calculate correlation matrix
    MatrixXd C = corr(data_obs);
    MatrixXd data_obs_cov = cov(data_obs);
    VectorXd data_obs_mean = mean(data_obs);
    size_t N = data_obs.cols();
    DAI_ASSERT( (size_t)B.cols() == N );

    // calculate set of intervention variables
    set<size_t> intvars;
    DAI_ASSERT( (size_t)data_intpos.rows() == 1 );
    for( size_t i = 0; i < (size_t)data_intpos.cols(); i++ )
        intvars.insert( data_intpos(0,i) );
    if (verbose >= 6) {
        cout << "intpos(0,i):";
        for( size_t i = 0; i < (size_t)data_intpos.cols(); i++ )
            cout << ' ' << i << ':' << hex << data_intpos(0,i) << dec;
        cout << endl;
        cout << "intpos(i):";
        for( size_t i = 0; i < (size_t)data_intpos.cols(); i++ )
            cout << ' ' << i << ':' << hex << data_intpos(i) << dec;
        cout << endl;
        // Both return 500 times 0xB0000000A, then 500 times garbage:
        // this function is passed 32-bit integers, but expects 64-bit ints.
        // Temporary fix: added dtype i8 in experiment_analyse.py
        // (True fix probably comes down to finding incompatible libraries
        // on my system)
    }

    // convert to DAG representation for speed
    dai::DAG G(N);
    for( size_t i = 0; i < N; i++ )
        for( size_t j = 0; j < N; j++ )
            if( B(i,j) != 0 )
                G.addEdge( i, j, true );

    // initialize independence testing objects
    indepTestPCor estT(C,Clo,Chi);
    indepTestOracle trueT(G);

    // calculate performance scores
    stats = MatrixXd::Zero(84+4*9+2*3,1);
    size_t p = 0;

    // independence, dependence
    ConTab CT_indep;
    for( size_t i = 0; i < N; i++ ) {
        for( size_t j = i + 1; j < N; j++ ) { // all unordered pairs {i,j}
            FUT est_indep = estT.test_indep( i, j );
            FUT true_indep = trueT.test_indep( i, j );
            CT_indep.update( est_indep, true_indep );
        }
    }
    stats(p+0) = CT_indep.TP; stats(p+1) = CT_indep.FP; stats(p+2) = CT_indep.TN; stats(p+3) = CT_indep.FN; stats(p+4) = CT_indep.UP; stats(p+5) = CT_indep.UN; p += 6;
    ConTab CT_dep = CT_indep.negate();
    stats(p+0) = CT_dep.TP; stats(p+1) = CT_dep.FP; stats(p+2) = CT_dep.TN; stats(p+3) = CT_dep.FN; stats(p+4) = CT_dep.UP; stats(p+5) = CT_dep.UN; p += 6;

    // conditional independence, conditional dependence, minimal independence, minimal dependence
    ConTab CT_condindep;
    ConTab CT_minindep;
    ConTab CT_mindep;
    for( size_t i = 0; i < N; i++ ) {
        for( size_t j = i + 1; j < N; j++ ) { // all unordered pairs {i,j}
            for( size_t k = 0; k < N; k++ ) {
                if( k != i && k != j ) {
                    triple ijk( i, j, k );

                    FUT est_condindep = estT.test_condindep( i, j, k );
                    FUT true_condindep = trueT.test_condindep( i, j, k );
                    CT_condindep.update( est_condindep, true_condindep );

                    FUT est_minindep = estT.test_minindep( ijk );
                    FUT true_minindep = trueT.test_minindep( ijk );
                    CT_minindep.update( est_minindep, true_minindep );

                    FUT est_mindep = estT.test_mindep( ijk );
                    FUT true_mindep = trueT.test_mindep( ijk );
                    CT_mindep.update( est_mindep, true_mindep );
                }
            }
        }
    }
    stats(p+0) = CT_condindep.TP; stats(p+1) = CT_condindep.FP; stats(p+2) = CT_condindep.TN; stats(p+3) = CT_condindep.FN; stats(p+4) = CT_condindep.UP; stats(p+5) = CT_condindep.UN; p += 6;
    ConTab CT_conddep = CT_condindep.negate();
    stats(p+0) = CT_conddep.TP; stats(p+1) = CT_conddep.FP; stats(p+2) = CT_conddep.TN; stats(p+3) = CT_conddep.FN; stats(p+4) = CT_conddep.UP; stats(p+5) = CT_conddep.UN; p += 6;
    stats(p+0) = CT_minindep.TP; stats(p+1) = CT_minindep.FP; stats(p+2) = CT_minindep.TN; stats(p+3) = CT_minindep.FN; stats(p+4) = CT_minindep.UP; stats(p+5) = CT_minindep.UN; p += 6;
    stats(p+0) = CT_mindep.TP; stats(p+1) = CT_mindep.FP; stats(p+2) = CT_mindep.TN; stats(p+3) = CT_mindep.FN; stats(p+4) = CT_mindep.UP; stats(p+5) = CT_mindep.UN; p += 6;

    set<quadruple> true_extYs; // (X,Y,U,W) quadruples in all true extY structures
    set<mypair> true_extY_xys; // (X,Y) tuples that are part of a true extY structure
    set<quadruple> true_Ys; // (X,Y,U,W) quadruples in all true Y structures
    set<mypair> true_Y_xys; // (X,Y) tuples that are part of a true extY structure
    set<quadruple> true_Y1s; // (X,Y,U,W) quadruples in all true Y1 structures
    set<quadruple> true_Y2s; // (X,Y,U,W) quadruples in all true Y2 structures
    // first use oracle
    for( size_t x = 0; x < N; x++ ) {
        for( size_t y = 0; y < N; y++ ) {
            if( y != x ) {
                if( trueT.test_indep( x, y ) == FALSE ) {
                    for( size_t z = 0; z < N; z++ ) {
                        if( z != x && z != y ) {
                            if( trueT.test_indep( y, z ) == FALSE ) {
                                if( trueT.test_condindep( y, z, x ) == TRUE ) {
                                    for( size_t u = 0; u < N; u++ ) {
                                        if( u != x && u != y && u != z ) {
                                            quadruple xyuz( x, y, u, z );

                                            FUT true_extY = trueT.test_extY( xyuz );
                                            if( true_extY == TRUE ) {
                                                true_extY_xys.insert( mypair( x, y ) );
                                                true_extYs.insert( xyuz );
                                            }

                                            FUT true_Y = (true_extY == TRUE) ? trueT.test_Y( xyuz ) : true_extY;
                                            //FUT true_Y = trueT.test_Y( xyuz );
                                            if( true_Y == TRUE ) {
                                                true_Y_xys.insert( mypair( x, y ) );
                                                true_Ys.insert( xyuz );
                                            }

                                            FUT true_Y1 = (true_Y == TRUE) ? trueT.test_Y1( xyuz ) : true_Y;
                                            if( true_Y1 == TRUE )
                                                true_Y1s.insert( xyuz );

                                            FUT true_Y2 = (true_Y1 == TRUE) ? trueT.test_Y2( xyuz ) : true_Y1;
                                            if( true_Y2 == TRUE )
                                                true_Y2s.insert( xyuz );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    DAI_ASSERT( true_Ys == true_Y1s );
    DAI_ASSERT( true_Ys == true_Y2s );
    // then estimate
    ConTab CT_extY;
    ConTab CT_Y;
    ConTab CT_Y1;
    ConTab CT_Y2;
    set<quadruple> est_extYs; // (X,Y,U,W) quadruples in all estimated extY structures
    set<mypair> est_extY_xys; // (X,Y) pairs in all estimated extY structures
    set<quadruple> est_Ys; // (X,Y,U,W) quadruples in all estimated Y structures
    set<mypair> est_Y_xys; // (X,Y) pairs in all estimated Y structures
    set<quadruple> est_Y1s; // (X,Y,U,W) quadruples in all estimated Y1 structures
    set<mypair> est_Y1_xys; // (X,Y) pairs in all estimated Y1 structures
    set<quadruple> est_Y2s; // (X,Y,U,W) quadruples in all estimated Y2 structures
    set<mypair> est_Y2_xys; // (X,Y) pairs in all estimated Y2 structures
    for( size_t x = 0; x < N; x++ ) {
        for( size_t y = 0; y < N; y++ ) {
            if( y != x ) {
                for( size_t z = 0; z < N; z++ ) {
                    if( z != x && z != y ) {
                        for( size_t u = 0; u < N; u++ ) {
                            if( u != x && u != y && u != z ) {
                                quadruple xyuz( x, y, u, z );

                                FUT est_extY = estT.test_extY( xyuz );
                                if( est_extY == TRUE ) {
                                    est_extY_xys.insert( mypair( x, y ) );
                                    est_extYs.insert( xyuz );
                                }
                                FUT true_extY = contains( true_extYs, xyuz ) ? TRUE : FALSE;
                                CT_extY.update( est_extY, true_extY );

                                FUT est_Y = (est_extY == TRUE) ? estT.test_Y( xyuz ) : est_extY;
                                if( est_Y == TRUE ) {
                                    est_Y_xys.insert( mypair( x, y ) );
                                    est_Ys.insert( xyuz );
                                }
                                FUT true_Y = contains( true_Ys, xyuz ) ? TRUE : FALSE;
                                CT_Y.update( est_Y, true_Y );

                                FUT est_Y1 = (est_Y == TRUE) ? estT.test_Y1( xyuz ) : est_Y;
                                if( est_Y1 == TRUE ) {
                                    est_Y1_xys.insert( mypair( x, y ) );
                                    est_Y1s.insert( xyuz );
                                }
                                FUT true_Y1 = contains( true_Y1s, xyuz ) ? TRUE : FALSE;
                                CT_Y1.update( est_Y1, true_Y1 );

                                FUT est_Y2 = (est_Y1 == TRUE) ? estT.test_Y2( xyuz ) : est_Y1;
                                if( est_Y2 == TRUE ) {
                                    est_Y2_xys.insert( mypair( x, y ) );
                                    est_Y2s.insert( xyuz );
                                }
                                FUT true_Y2 = contains( true_Y2s, xyuz ) ? TRUE : FALSE;
                                CT_Y2.update( est_Y2, true_Y2 );
                            }
                        }
                    }
                }
            }
        }
    }
    stats(p+0) = CT_extY.TP; stats(p+1) = CT_extY.FP; stats(p+2) = CT_extY.TN; stats(p+3) = CT_extY.FN; stats(p+4) = CT_extY.UP; stats(p+5) = CT_extY.UN; p += 6;
    stats(p+0) = CT_Y.TP; stats(p+1) = CT_Y.FP; stats(p+2) = CT_Y.TN; stats(p+3) = CT_Y.FN; stats(p+4) = CT_Y.UP; stats(p+5) = CT_Y.UN; p += 6;
    stats(p+0) = CT_Y1.TP; stats(p+1) = CT_Y1.FP; stats(p+2) = CT_Y1.TN; stats(p+3) = CT_Y1.FN; stats(p+4) = CT_Y1.UP; stats(p+5) = CT_Y1.UN; p += 6;
    stats(p+0) = CT_Y2.TP; stats(p+1) = CT_Y2.FP; stats(p+2) = CT_Y2.TN; stats(p+3) = CT_Y2.FN; stats(p+4) = CT_Y2.UP; stats(p+5) = CT_Y2.UN; p += 6;
//  cout << "# true_Ys: " << true_Ys.size() << ", # true_Y1s: " << true_Y1s.size() << ", # true_Y2s:" << true_Y2s.size() << endl;
//  cout << "# est_Ys: " << est_Ys.size() << ", # est_Y1s: " << est_Y1s.size() << ", # est_Y2s:" << est_Y2s.size() << endl;

    // THIJS: export true and estimated (ext)Y locations for RICF analysis
    /*int true_extYs_num = true_extYs.size();
    int true_Ys_num = true_Ys.size();
    std::cout << "Size difference: extY " << true_extYs_num << ",\tY "
    << true_Ys_num << std::endl;*/
    std::ofstream of;
    of.open(struc_csv.c_str()); // ios::trunc?
    for( size_t x = 0; x < N; x++ ) {
        for( size_t y = 0; y < N; y++ ) {
            if( y != x ) {
                for( size_t u = 0; u < N; u++ ) {
                    if( u != x && u != y ) {
                        for( size_t z = u + 1; z < N; z++ ) {
                            if( z != x && z != y ) {
                                quadruple xyuz( x, y, u, z );
                                quadruple xyuz_refl( x, y, z, u);
                                int true_res = 0;
                                if (contains(true_Ys, xyuz)) {
                                    true_res = 4;
                                }
                                else {
                                    // No Y, but maybe an extended Y.
                                    // Symmetry matters now: check both ways.
                                    if (contains(true_extYs, xyuz))
                                        true_res = 1;
                                    else if (contains(true_extYs, xyuz_refl))
                                        true_res = -1;
                                }
                                int est_res = 0;
                                if (contains(est_Ys, xyuz)) {
                                    if (contains(est_Y2s, xyuz))
                                        est_res = 4;
                                    else if (contains(est_Y1s, xyuz))
                                        est_res = 3;
                                    else
                                        est_res = 2;
                                }
                                else {
                                    // No Y, but maybe an extended Y.
                                    // Symmetry matters now: check both ways.
                                    if (contains(est_extYs, xyuz))
                                        est_res = 1;
                                    else if (contains(est_extYs, xyuz_refl))
                                        est_res = -1;
                                }
                                if (true_res || est_res) {
                                    of << x << ',' << y << ',' << u << ',' << z
                                       << ',' << true_res << ',' << est_res
                                       << std::endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    of.close();

    ConTab CT_extY_anY;
    for( size_t x = 0; x < N; x++ ) {
        for( size_t y = 0; y < N; y++ ) {
            if( y != x ) {
                FUT est_XanY = FALSE;
                if( contains( est_extY_xys, mypair(x,y) ) )
                    // (x,y) is part of an estimated extended Y structure
                    est_XanY = TRUE;
                FUT true_XanY = G.existsDirectedPath( x, y ) ? TRUE : FALSE;
                CT_extY_anY.update( est_XanY, true_XanY );
            }
        }
    }
    if( verbose >= 2 ) {
        cout << "extY an(X,Y): " << CT_extY_anY << "; recall=" << CT_extY_anY.recall() << ", precision=" << CT_extY_anY.precision() << endl;
    } else {
        ConTab bla = CT_extY_anY;
        stats(p+0) = bla.TP; stats(p+1) = bla.FP; stats(p+2) = bla.TN; stats(p+3) = bla.FN; stats(p+4) = bla.UP; stats(p+5) = bla.UN; p += 6;
        if( verbose >= 1 )
            cout << bla.TP << " " << bla.FP << " " << bla.TN << " " << bla.FN << " " << bla.recall() << " " << bla.precision() << " ";
    }

    ConTab CT_Y_anY;
    for( size_t x = 0; x < N; x++ ) {
        for( size_t y = 0; y < N; y++ ) {
            if( y != x ) {
                FUT est_XanY = FALSE;
                if( contains( est_Y_xys, mypair(x,y) ) )
                    // (x,y) is part of an estimated Y structure
                    est_XanY = TRUE;
                FUT true_XanY = G.existsDirectedPath( x, y ) ? TRUE : FALSE;
                CT_Y_anY.update( est_XanY, true_XanY );
            }
        }
    }
    if( verbose >= 2 ) {
        cout << "Y an(X,Y): " << CT_Y_anY << "; recall=" << CT_Y_anY.recall() << ", precision=" << CT_Y_anY.precision() << endl;
    } else {
        ConTab bla = CT_Y_anY;
        stats(p+0) = bla.TP; stats(p+1) = bla.FP; stats(p+2) = bla.TN; stats(p+3) = bla.FN; stats(p+4) = bla.UP; stats(p+5) = bla.UN; p += 6;
        if( verbose >= 1 )
            cout << bla.TP << " " << bla.FP << " " << bla.TN << " " << bla.FN << " " << bla.recall() << " " << bla.precision() << " ";
    }

    ConTab CT_Y1_anY;
    for( size_t x = 0; x < N; x++ ) {
        for( size_t y = 0; y < N; y++ ) {
            if( y != x ) {
                FUT est_XanY = FALSE;
                if( contains( est_Y1_xys, mypair(x,y) ) )
                    // (x,y) is part of an estimated Y structure
                    est_XanY = TRUE;
                FUT true_XanY = G.existsDirectedPath( x, y ) ? TRUE : FALSE;
                CT_Y1_anY.update( est_XanY, true_XanY );
            }
        }
    }
    if( verbose >= 2 ) {
        cout << "Y1 an(X,Y): " << CT_Y1_anY << "; recall=" << CT_Y1_anY.recall() << ", precision=" << CT_Y1_anY.precision() << endl;
    } else {
        ConTab bla = CT_Y1_anY;
        stats(p+0) = bla.TP; stats(p+1) = bla.FP; stats(p+2) = bla.TN; stats(p+3) = bla.FN; stats(p+4) = bla.UP; stats(p+5) = bla.UN; p += 6;
        if( verbose >= 1 )
            cout << bla.TP << " " << bla.FP << " " << bla.TN << " " << bla.FN << " " << bla.recall() << " " << bla.precision() << " ";
    }

    ConTab CT_Y2_anY;
    for( size_t x = 0; x < N; x++ ) {
        for( size_t y = 0; y < N; y++ ) {
            if( y != x ) {
                FUT est_XanY = FALSE;
                if( contains( est_Y2_xys, mypair(x,y) ) )
                    // (x,y) is part of an estimated Y structure
                    est_XanY = TRUE;
                FUT true_XanY = G.existsDirectedPath( x, y ) ? TRUE : FALSE;
                CT_Y2_anY.update( est_XanY, true_XanY );
            }
        }
    }
    if( verbose >= 2 ) {
        cout << "Y2 an(X,Y): " << CT_Y2_anY << "; recall=" << CT_Y2_anY.recall() << ", precision=" << CT_Y2_anY.precision() << endl;
        cout << "next up: predictions" << endl;
    } else {
        ConTab bla = CT_Y2_anY;
        stats(p+0) = bla.TP; stats(p+1) = bla.FP; stats(p+2) = bla.TN; stats(p+3) = bla.FN; stats(p+4) = bla.UP; stats(p+5) = bla.UN; p += 6;
        if( verbose >= 1 )
            cout << bla.TP << " " << bla.FP << " " << bla.TN << " " << bla.FN << " " << bla.recall() << " " << bla.precision() << " ";
    }

    // extended Y structures prediction errors
    for( size_t x = 0; x < N; x++ ) {
        for( size_t y = 0; y < N; y++ ) {
            if( y != x )
                if (verbose >= 5)
                    cout << 'a';
                if( contains( est_extY_xys, mypair(x,y) ) ) {
                    if (verbose >= 5)
                        cout << 'b';
                    bool z = contains( true_extY_xys, mypair(x,y) );
                    // (x,y) is part of an estimated extended Y structure
                    if( contains( intvars, x ) ) {
                        double ErrL1, ErrL2;
                        size_t ErrCount;
                        evalL1L2Err( data_obs_cov, data_obs_mean, data_int_, data_intpos_, x, y, verbose, ErrCount, ErrL1, ErrL2);
                        stats(p+0) += ErrCount; stats(p+1) += ErrL1; stats(p+2) += ErrL2;
                        if( z ) {
                            stats(p+3) += ErrCount; stats(p+4) += ErrL1; stats(p+5) += ErrL2;
                        } else {
                            stats(p+6) += ErrCount; stats(p+7) += ErrL1; stats(p+8) += ErrL2;
                        }
                    }
                }
                if (verbose >= 5)
                    cout << '.';
        }
    }
    p += 9;
    if( verbose >= 5 ) {
        cout << "Finished evaluating predictions for extY" << endl;
    }
    /*(extY_avgL1Err,extY_avgL2Err) = avgL1L2Err(stats2[0,0], stats2[1,0], stats2[2,0])
    (extY_avgL1ErrTP,extY_avgL2ErrTP) = avgL1L2Err(stats2[3,0], stats2[4,0], stats2[5,0])
    (extY_avgL1ErrFP,extY_avgL2ErrFP) = avgL1L2Err(stats2[6,0], stats2[7,0], stats2[8,0])
    if verbose:
        print 'extY: True positives: ', TP, ', False positives: ', FP
        print 'Average L1 error: ', extY_avgL1Err, ' (count: ', extY_ErrCount, ')'
        print 'Average L2 error: ', extY_avgL2Err, ' (count: ', extY_ErrCount, ')'
        print 'Average L1 error TP: ', extY_avgL1ErrTP, ' (count: ', extY_ErrCount_TP, ')'
        print 'Average L2 error TP: ', extY_avgL2ErrTP, ' (count: ', extY_ErrCount_TP, ')'
        print 'Average L1 error FP: ', extY_avgL1ErrFP, ' (count: ', extY_ErrCount_FP, ')'
        print 'Average L2 error FP: ', extY_avgL2ErrFP, ' (count: ', extY_ErrCount_FP, ')'
    #else:
    #    print extY_avgL1Err, extY_avgL2Err, extY_avgL1ErrTP, extY_avgL2ErrTP, extY_avgL1ErrFP, extY_avgL2ErrFP,*/

    // Y structures prediction errors
    for( size_t x = 0; x < N; x++ ) {
        for( size_t y = 0; y < N; y++ ) {
            if( y != x ) 
                if( contains( est_Y_xys, mypair(x,y) ) ) {
                    bool z = contains( true_Y_xys, mypair(x,y) );
                    // (x,y) is part of an estimated extended Y structure
                    if( contains( intvars, x ) ) {
                        double ErrL1, ErrL2;
                        size_t ErrCount;
                        evalL1L2Err( data_obs_cov, data_obs_mean, data_int_, data_intpos_, x, y, verbose, ErrCount, ErrL1, ErrL2);
                        stats(p+0) += ErrCount; stats(p+1) += ErrL1; stats(p+2) += ErrL2;
                        if( z ) {
                            stats(p+3) += ErrCount; stats(p+4) += ErrL1; stats(p+5) += ErrL2;
                        } else {
                            stats(p+6) += ErrCount; stats(p+7) += ErrL1; stats(p+8) += ErrL2;
                        }
                    }
                }
        }
    }
    p += 9;
    /*(Y_avgL1Err,Y_avgL2Err) = avgL1L2Err(stats2[0,0], stats2[1,0], stats2[2,0])
    (Y_avgL1ErrTP,Y_avgL2ErrTP) = avgL1L2Err(stats2[3,0], stats2[4,0], stats2[5,0])
    (Y_avgL1ErrFP,Y_avgL2ErrFP) = avgL1L2Err(stats2[6,0], stats2[7,0], stats2[8,0])
    if verbose:
        print 'Y: True positives: ', TP, ', False positives: ', FP
        print 'Average L1 error: ', Y_avgL1Err, ' (count: ', Y_ErrCount, ')'
        print 'Average L2 error: ', Y_avgL2Err, ' (count: ', Y_ErrCount, ')'
        print 'Average L1 error TP: ', Y_avgL1ErrTP, ' (count: ', Y_ErrCount_TP, ')'
        print 'Average L2 error TP: ', Y_avgL2ErrTP, ' (count: ', Y_ErrCount_TP, ')'
        print 'Average L1 error FP: ', Y_avgL1ErrFP, ' (count: ', Y_ErrCount_FP, ')'
        print 'Average L2 error FP: ', Y_avgL2ErrFP, ' (count: ', Y_ErrCount_FP, ')'
    #else:
    #    print Y_avgL1Err, Y_avgL2Err, Y_avgL1ErrTP, Y_avgL2ErrTP, Y_avgL1ErrFP, Y_avgL2ErrFP,*/

    // Y1 structures prediction errors
    for( size_t x = 0; x < N; x++ ) {
        for( size_t y = 0; y < N; y++ ) {
            if( y != x ) 
                if( contains( est_Y1_xys, mypair(x,y) ) ) {
                    bool z = contains( true_Y_xys, mypair(x,y) );
                    // (x,y) is part of an estimated extended Y structure
                    if( contains( intvars, x ) ) {
                        double ErrL1, ErrL2;
                        size_t ErrCount;
                        evalL1L2Err( data_obs_cov, data_obs_mean, data_int_, data_intpos_, x, y, verbose, ErrCount, ErrL1, ErrL2);
                        stats(p+0) += ErrCount; stats(p+1) += ErrL1; stats(p+2) += ErrL2;
                        if( z ) {
                            stats(p+3) += ErrCount; stats(p+4) += ErrL1; stats(p+5) += ErrL2;
                        } else {
                            stats(p+6) += ErrCount; stats(p+7) += ErrL1; stats(p+8) += ErrL2;
                        }
                    }
                }
        }
    }
    p += 9;

    // Y2 structures prediction errors
    for( size_t x = 0; x < N; x++ ) {
        for( size_t y = 0; y < N; y++ ) {
            if( y != x ) 
                if( contains( est_Y2_xys, mypair(x,y) ) ) {
                    bool z = contains( true_Y_xys, mypair(x,y) );
                    // (x,y) is part of an estimated extended Y structure
                    if( contains( intvars, x ) ) {
                        double ErrL1, ErrL2;
                        size_t ErrCount;
                        evalL1L2Err( data_obs_cov, data_obs_mean, data_int_, data_intpos_, x, y, verbose, ErrCount, ErrL1, ErrL2);
                        stats(p+0) += ErrCount; stats(p+1) += ErrL1; stats(p+2) += ErrL2;
                        if( z ) {
                            stats(p+3) += ErrCount; stats(p+4) += ErrL1; stats(p+5) += ErrL2;
                        } else {
                            stats(p+6) += ErrCount; stats(p+7) += ErrL1; stats(p+8) += ErrL2;
                        }
                    }
                }
        }
    }
    p += 9;
    if( verbose >= 5 ) {
        cout << "Finished evaluating predictions for Y2" << endl;
    }

    // baseline 1
    double bl1_ErrL1, bl1_ErrL2;
    size_t bl1_ErrCount;
    evalBaseline( data_obs_cov, data_obs_mean, data_int_, data_intpos_, verbose, bl1_ErrCount, bl1_ErrL1, bl1_ErrL2, 1 );
    if( verbose >= 5 ) {
        cout << "Returned from first call to evalBaseline(...)" << endl;
    }
    stats(p+0) += bl1_ErrCount; stats(p+1) += bl1_ErrL1; stats(p+2) += bl1_ErrL2;
    p += 3;
/*  if verbose:
        print 'Average bl1 L1 error: ', bl1_ErrL1 / bl1_ErrCount
        print 'Average bl1 L2 error: ', math.sqrt(bl1_ErrL2 / bl1_ErrCount)
    #else:
    #    print bl1_ErrL1 / bl1_ErrCount, math.sqrt(bl1_ErrL2 / bl1_ErrCount),*/

    // baseline 2
    double bl2_ErrL1, bl2_ErrL2;
    size_t bl2_ErrCount;
    evalBaseline( data_obs_cov, data_obs_mean, data_int_, data_intpos_, verbose, bl2_ErrCount, bl2_ErrL1, bl2_ErrL2, 2 );
    stats(p+0) += bl2_ErrCount; stats(p+1) += bl2_ErrL1; stats(p+2) += bl2_ErrL2;
    p += 3;
/*  if verbose:
        print 'Average bl2 L1 error: ', bl2_ErrL1 / bl2_ErrCount
        print 'Average bl2 L2 error: ', math.sqrt(bl2_ErrL2 / bl2_ErrCount)
    #else:
    #    print bl2_ErrL1 / bl2_ErrCount, math.sqrt(bl2_ErrL2 / bl2_ErrCount),*/

    if( verbose >= 5 ) {
        cout << "Finished evaluating baseline predictions; function done" << endl;
    }

    return 0;
}


template <typename Derived1,typename Derived2,typename Derived3,typename Derived4,typename Derived5>
inline int projectADMG(const Eigen::MatrixBase<Derived1>* B_,const Eigen::MatrixBase<Derived2>* S_,const Eigen::MatrixBase<Derived3>* obsVar_,const Eigen::MatrixBase<Derived4>* projB_,const Eigen::MatrixBase<Derived5>* projS_,size_t verbose)
{
    // assumption: 0,1,..,N is a topological ordering of B

    using namespace std;

    // recast Eigen output to writeable element.
    Eigen::MatrixBase<Derived1>& B = const_cast< Eigen::MatrixBase<Derived1>& >(*B_);
    Eigen::MatrixBase<Derived2>& S = const_cast< Eigen::MatrixBase<Derived2>& >(*S_);
    Eigen::MatrixBase<Derived3>& obsVar = const_cast< Eigen::MatrixBase<Derived3>& >(*obsVar_);
    Eigen::MatrixBase<Derived4>& projB = const_cast< Eigen::MatrixBase<Derived4>& >(*projB_);
    Eigen::MatrixBase<Derived5>& projS = const_cast< Eigen::MatrixBase<Derived5>& >(*projS_);

    size_t N = B.rows();
    DAI_ASSERT( (size_t)B.cols() == N );
    DAI_ASSERT( (size_t)S.rows() == N );
    DAI_ASSERT( (size_t)S.cols() == N );
    DAI_ASSERT( obsVar.rows() == 1 );

    set<size_t> obsVarSet;
    for( size_t i = 0; i < (size_t)obsVar.cols(); i++ )
        obsVarSet.insert( obsVar(0,i) );
    size_t projN = obsVarSet.size();
    DAI_ASSERT( projN == (size_t)obsVar.cols() );

    /*
    // convert to DAG representation for speed
    dai::DAG G(N);
    for( size_t i = 0; i < N; i++ )
        for( size_t j = 0; j < N; j++ )
            if( B(i,j) != 0 )
                G.addEdge( i, j, true );
*/

    // initialize
    MatrixXd tmpB( N, N );
    MatrixXd tmpS( N, N );
    for( size_t i = 0; i < N; i++ ) {
        for( size_t j = 0; j < N; j++ ) {
            tmpB(i,j) = B(i,j);
            tmpS(i,j) = (i == j) * 1.0;
        }
    }

    if( verbose ) {
        cout << tmpB << endl;
        cout << tmpS << endl;
    }
    // substitute equations of variables that become latent
    for( size_t i = 0; i < N; i++ ) {
        for( size_t j = 0; j < N; j++ ) {
            if( tmpB(i,j) != 0.0 ) {
                if( i >= j )
                    cerr << "Error: B doesn't have the natural topological ordering!" << endl;
                else {
                    if( !contains( obsVarSet, i ) ) { // parent i of j becomes latent
                        if( verbose )
                            cout << "Add equation of " << i << " to equation of " << j << endl;
                        double c = tmpB(i,j);
                        for( size_t k = 0; k < N; k++ ) {
                            tmpB(k,j) += c * tmpB(k,i); // add eq. of i to eq. of j
                            tmpS(k,j) += c * tmpS(k,i);
                        }
                        tmpB(i,j) -= c; // compensate (to keep algebraically equivalent equations)
                        if( verbose ) {
                            cout << tmpB << endl;
                            cout << tmpS << endl;
                        }
                    }
                }
            }
        }
    }

    // select submatrix
    projB = MatrixXd( projN, projN );
    for( size_t ind_i = 0; ind_i < projN; ind_i++ ) {
        for( size_t ind_j = 0; ind_j < projN; ind_j++ ) {
            size_t i = obsVar(0,ind_i);
            size_t j = obsVar(0,ind_j);
            projB(ind_i,ind_j) = tmpB(i,j);
        }
    }
    
    projS = MatrixXd( projN, N );
    for( size_t ind_i = 0; ind_i < projN; ind_i++ ) {
        for( size_t j = 0; j < N; j++ ) {
            size_t i = obsVar(0,ind_i);
            projS(ind_i,j) = tmpS(j,i);
        }
    }
    if( verbose )
        cout << endl << projS << "x" << endl << S << "x" << endl << projS.transpose() << endl;
    projS = projS * S * projS.transpose();

    if( verbose ) {
        cout << "Selecting observed variables" << endl;
        cout << projB << endl;
        cout << projS << endl;
    }

    return 0;
}


#endif // YSTRUCT_H
