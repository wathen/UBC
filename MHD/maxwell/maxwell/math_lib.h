#ifndef MATH_LIB_H
#define MATH_LIB_H

#include <vector>
using namespace std;

//void mycross(double *a, double *b, double *c);
void mycross(int m, double *a, double *b, double *c);
double mydot(int m, double *a, double *b);
void mat_vec_mult(int m, int n, double *A, double *x, double *f);
void mat_mat_mult(int m, int n, int l, double *A, double *B, double *C);
void mat_sum(int m, int n, double f1, double f2, double *A, double *B, double *C);
void mat_scale(int m, int n, double f, double *A, double *B); 
void mat_trans(int m, int n, double *A, double *B); // B = A'
void mat_inv(int m, double *A, double *B); //B = A^{-1}
void mat_iden(int m, double *A); //A = iden(m);
double mat_det(int m, double *A); //det(A)
void vec_scale(int m, double f, double *v, double *t); //t = f*v
void vec_sum(int m, double f1, double f2, double *a, double *b, double *c); //c = f1*a + f2*b;
double vec_normal(int m, double *a);

// functions for std::vector
void mat_scale(vector<vector<double> > &mat, double f); //mat = f*mat
void vec_scale(vector<double>  &vec, double f); //vec = f*vec
#endif

