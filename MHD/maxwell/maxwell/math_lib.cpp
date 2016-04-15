#include "math_lib.h"
#include <math.h>

//void mycross(double *a, double *b, double *c)
//{
//	c[0] = a[1]*b[2] - a[2]*b[1];
//	c[1] = a[2]*b[0] - a[0]*b[2];
//	c[2] = a[0]*b[1] - a[1]*b[0];
//}

void mycross(int m, double *a, double *b, double *c)
{
	if (m==3) {
		c[0] = a[1]*b[2] - a[2]*b[1];
		c[1] = a[2]*b[0] - a[0]*b[2];
		c[2] = a[0]*b[1] - a[1]*b[0];
	}
	if (m==2) {
		c[0] = 0;
		c[1] = 0;
		c[2] = a[0]*b[1] - a[1]*b[0];
	}
}


double mydot(int m, double *a, double *b)
{
	if (m==3)
		return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
	if (m==2)
		return a[0]*b[0]+a[1]*b[1];
}

// f = A*x
void mat_vec_mult(int m, int n, double *A, double *x, double *f)
{
	for (int i=0; i<m; i++) {
		f[i] = 0;
		for (int j=0; j<n; j++)  
			f[i] += A[i*n+j]*x[j];		
	}
}

// C = A*B
void mat_mat_mult(int m, int n, int l, double *A, double *B, double *C)
{
	for (int i=0; i<m; i++) 
		for (int j=0; j<l; j++) {
			C[i*l+j] = 0;
			for (int k=0; k<n; k++) 
				C[i*l+j] +=A[i*n+k]*B[k*l+j];
		}
}

//C = f1*A + f2*B
void mat_sum(int m, int n, double f1, double f2, double *A, double *B, double *C)
{
	for (int i=0; i<m; i++) 
		for (int j=0; j<n; j++) 
			C[i*n+j] = f1*A[i*n+j] + f2*B[i*n+j];
}

//B = f*A 
void mat_scale(int m, int n, double f, double *A, double *B)
{
	for (int i=0; i<m; i++) 
		for (int j=0; j<n; j++) 
			B[i*n+j] = f*A[i*n+j];
}

//B = A'
void mat_trans(int m, int n, double *A, double *B)
{
	for (int i=0; i<m; i++) 
		for (int j=0; j<n; j++) {
			//B[j][i] =A[i][j]
			B[j*m+i]=A[i*n+j];
		}
}

//B = inv(A);
void mat_inv(int m, double *A, double *B)
{
	if (m==2) {
		double a = A[0];
		double b = A[1];
		double c = A[2];
		double d = A[3];

		double D = a*d-b*c;
		double Di = 1.0/D;
		B[0] =  Di*d;
		B[1] = -Di*b;
		B[2] = -Di*c;
		B[3] =  Di*a;
	}

	if (m==3) {
		double a = A[0];
		double b = A[1];
		double c = A[2];
		double d = A[3];
		double e = A[4];
		double f = A[5];
		double g = A[6];
		double h = A[7];
		double i = A[8];

		double D = a*(e*i-f*h) - b*(d*i-g*f) + c*(d*h-e*g);
		double Di = 1.0f/D;
		B[0] = Di*(e*i-f*h);
		B[1] = Di*(c*h-i*b);
		B[2] = Di*(b*f-e*c);
		B[3] = Di*(f*g-i*d);
		B[4] = Di*(a*i-g*c);
		B[5] = Di*(c*d-a*f);
		B[6] = Di*(d*h-e*g);
		B[7] = Di*(b*g-h*a);
		B[8] = Di*(a*e-b*d);
	}
}

//A = iden(m)
void mat_iden(int m, double *A)
{
	for (int i=0; i<m; i++)
		for (int j=0; j<m; j++) 
			A[i*m+j] = (i==j);
}

//det(A)
double mat_det(int m, double *A)
{
	if (m==2) {
		double a = A[0];
		double b = A[1];
		double c = A[2];
		double d = A[3];

		return a*d-b*c;
	}
	if (m==3) {
		double a = A[0];
		double b = A[1];
		double c = A[2];
		double d = A[3];
		double e = A[4];
		double f = A[5];
		double g = A[6];
		double h = A[7];
		double i = A[8];

		return a*(e*i-f*h) - b*(d*i-g*f) + c*(d*h-e*g);
	}
}


//t = f*v
void vec_scale(int m, double f, double *v, double *t)
{
	for (int i=0; i<m; i++) 
		t[i] = f*v[i];
}

//c = f1*a + f2*b;
void vec_sum(int m, double f1, double f2, double *a, double *b, double *c)
{
	for (int i=0; i<m; i++) 
		c[i] = f1*a[i] + f2*b[i];
}

// normal of a
double vec_normal(int m, double *a)
{
	double r = 0;
	for (int i=0; i<m; i++)
		r += a[i]*a[i];
	return sqrt(r);
}

void mat_scale(vector<vector<double> > &mat, double f)
{
	int m, n;
	m = mat.size();
	n = mat[0].size();

	for (int i=0; i<m; i++) 
		for (int j=0; j<n; j++)
			mat[i][j] *= f;
}

void vec_scale(vector<double>  &vec, double f)
{
	int m;
	m = vec.size();

	for (int i=0; i<m; i++) 
		vec[i] *= f;
}
