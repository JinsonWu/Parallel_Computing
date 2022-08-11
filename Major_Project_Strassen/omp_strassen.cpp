#include <omp.h>
#include <bits/stdc++.h>
#include <stdio.h>      
#include <stdlib.h>  
#include <math.h>
#include <time.h>   

using namespace std;

#define MAX_THREADS 2048
#define MAX_MATRIX_SIZE 65536
#define DEBUG false
#define THREAD 32
// For better debugging and lower computing loading
// I set up MAX_ELEMENT_VALUE here to limit the maximum value in matrices
#define MAX_ELEMENT_VALUE 1048
// naive limit to initiate parallel execution
#define NAIVE_BARRIER 32

typedef unsigned long long ull;

int matrix_size, num_threads;   // global variables
//int* ptr;
//ull** a;
//ull** b;
//ull** prod;
//ull** prod_naive;

// print matrix while debugging
void print(int n, ull** mat){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++) cout << mat[i][j] << " ";
        cout << endl;
    }
    cout << endl;
}

ull** allocateMatrix(int n){
    /* allocate memory for matrix */
    ull* data = (ull*)malloc(n * n * sizeof(ull));
    ull** array = (ull**)malloc(n * sizeof(ull*));
    for (int i = 0; i < n; i++) array[i] = &(data[n * i]);
    return array;
}

void fillMatrix(int n, ull**& mat){
    /* Utilize random number fill in the matrix */
    srand48(0);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            mat[i][j] = lrand48() % MAX_ELEMENT_VALUE;
        }
    }
}

// naive method
void naive(int n, ull**& a, ull**& b, ull**& prod_naive){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            prod_naive[i][j] = 0;
            for (int k = 0; k < n; k++) prod_naive[i][j] += a[i][k] * b[k][j];
        }
    }
}

// naive method with omp
ull** naive_omp(int n, ull** mat1, ull** mat2){
    ull** result = allocateMatrix(n);

    #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                result[i][j] = 0;
                for (int k = 0; k < n; k++) result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    return result;
}

// fetch the desired matrix slices (a11-a22, b11-b22)
ull** getSlice(int n, ull** mat, int off_i, int off_j){
    int m = n / 2;
    ull** slice = allocateMatrix(m);
    //
    for (int i = 0; i < m; i++){
        for (int j = 0; j < m; j++){
            slice[i][j] = mat[off_i + i][off_j + j];
        }
    }
    return slice;
}

// matrix operations
ull** addMatrices(int n, ull** mat1, ull** mat2, bool op){
    ull** result = allocateMatrix(n);

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if (op) result[i][j] = mat1[i][j] + mat2[i][j];
            else result[i][j] = mat1[i][j] - mat2[i][j];
        }
    }
    return result;
}

// merge the final matrix
ull** combineMatrices(int m, ull** c11, ull** c12, ull** c21, ull** c22){
    int n = 2 * m;
    ull** result = allocateMatrix(n);

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if (i < m && j < m) result[i][j] = c11[i][j];
            else if (i < m) result[i][j] = c12[i][j - m];
            else if (j < m) result[i][j] = c21[i - m][j];
            else result[i][j] = c22[i - m][j - m];
        }
    }
    return result;
}

ull** strassen(int n, ull** mat1, ull** mat2){
    //int np = matrix_size/num_threads;
    int m = n/2;
    bool add = true, sub = false;

    //printf("Thread No.%d\n", omp_get_thread_num());

    if (n <= NAIVE_BARRIER) return naive_omp(n, mat1, mat2);

    // obtain slices from a11~b22
    ull** a11 = getSlice(n, mat1, 0, 0);
    ull** a12 = getSlice(n, mat1, 0, m);
    ull** a21 = getSlice(n, mat1, m, 0);
    ull** a22 = getSlice(n, mat1, m, m);
    ull** b11 = getSlice(n, mat2, 0, 0);
    ull** b12 = getSlice(n, mat2, 0, m);
    ull** b21 = getSlice(n, mat2, m, 0);
    ull** b22 = getSlice(n, mat2, m, m);

    // utilize task shared to perform parallel execution and make sure data coherence
    ull** m1;
    #pragma omp task shared(m1)
    {
        ull** m11 = addMatrices(m, a11, a22, add);
        ull** m12 = addMatrices(m, b11, b22, add);
        m1 = strassen(m, m11, m12);
        free(m11);
        free(m12);
    }

    ull** m2;
    #pragma omp task shared(m2)
    {
        ull** m21 = addMatrices(m, a21, a22, add);
        m2 = strassen(m, m21, b11);
        free(m21);
    }

    ull** m3;
    #pragma omp task shared(m3)
    {
        ull** m31 = addMatrices(m, b12, b22, sub);
        m3 = strassen(m, a11, m31);
        free(m31);
    }

    ull** m4;
    #pragma omp task shared(m4)
    {
        ull** m41 = addMatrices(m, b21, b11, sub);
        m4 = strassen(m, a22, m41);
        free(m41);
    }

    ull** m5;
    #pragma omp task shared(m5)
    {
        ull** m51 = addMatrices(m, a11, a12, add);
        m5 = strassen(m, m51, b22);
        free(m51);
    }

    ull** m6;
    #pragma omp task shared(m6)
    {
        ull** m61 = addMatrices(m, a21, a11, sub);
        ull** m62 = addMatrices(m, b11, b12, add);
        m6 = strassen(m, m61, m62);
        free(m61);
        free(m62);
    }

    ull** m7;
    #pragma omp task shared(m7)
    {
        ull** m71 = addMatrices(m, a12, a22, sub);
        ull** m72 = addMatrices(m, b21, b22, add);
        m7 = strassen(m, m71, m72);
        free(m71);
        free(m72);
    }

    // wait to sync
    #pragma omp taskwait

    free(a11); free(a12); free(a21); free(a22);
    free(b11); free(b12); free(b21); free(b22);

    // combine previously-calculated matrices to have c11~c22
    ull** c11;
    #pragma omp task shared(c11)
    {
        ull** c11_ = addMatrices(m, m1, m4, add);
        ull** c12_ = addMatrices(m, m5, m7, sub);
        c11 = addMatrices(m, c11_, c12_, sub);
        free(c11_);
        free(c12_);
    }

    ull** c12;
    #pragma omp task shared(c12)
    {
        c12 = addMatrices(m, m3, m5, add);
    }

    ull** c21;
    #pragma omp task shared(c21)
    {
        c21 = addMatrices(m, m2, m4, add);
    }

    ull** c22;
    #pragma omp task shared(c22)
    {
        ull** c21_ = addMatrices(m, m1, m2, sub);
        ull** c22_ = addMatrices(m, m3, m6, add);
        c22 = addMatrices(m, c21_, c22_, add);
        free(c21_);
        free(c22_);
    }

    // wait to sync
    #pragma omp taskwait

    free(m1); free(m2); free(m3); free(m4);
    free(m5); free(m6); free(m7);

    // combine matrices to get the eventual matrix c
    ull** prod = combineMatrices(m, c11, c12, c21, c22);

    free(c11); free(c12); free(c21); free(c22);

    return prod;
}

int main(int argc, char *argv[]){
    // variables declaration
    int k, q, n, np; //
    struct timespec start, stop, stop_naive;
    double total_time, total_time_naive;  
    int error = 0;
    
    if (argc != 3){
        printf("Please Enter the Size of Matrix and Number of Threads!\n");
        exit(0);
    }
    else{
        k = atoi(argv[argc-2]);
        if ((n = 1 << k) > MAX_MATRIX_SIZE){
            printf("Exceed Maximum Matrix Size: %d!\n", MAX_MATRIX_SIZE);
            exit(0);
        }
        q = atoi(argv[argc-1]);
        if ((num_threads = 1 << q) > MAX_THREADS){
            printf("Exceed Maximum Threads: %d!\n", MAX_THREADS);
            exit(0);
        }
    } 

    matrix_size = n;

    // malloc matrices
    ull** a = allocateMatrix(n);
    ull** b = allocateMatrix(n);
    ull** prod = allocateMatrix(n);
    ull** prod_naive = allocateMatrix(n);
    //ptr = (int *) malloc((num_threads+1) * sizeof(int));

    // put random numbers as element in the intial matrices a & b
    fillMatrix(n, a);
    fillMatrix(n, b);

    if(DEBUG) print(n, a);
    if(DEBUG) print(n, b);

    //
    //printf("Start Parallel Matrix Multiplication Execution!\n");

    // set up starting time
    clock_gettime(CLOCK_REALTIME, &start);

    /*
    np = matrix_size / num_threads; // Sub list size

    // Initialize starting position for each sublist
    for (int i = 0; i < num_threads; i++) ptr[i] = i * np;
    ptr[num_threads] = matrix_size;
    */

    // set up num_threads
    omp_set_num_threads(num_threads);

    // parallel computing
    #pragma omp parallel
    {
    #pragma omp single
            prod = strassen(n, a, b);
    }

    if(DEBUG) print(n, prod);

    // calculate the execution time for the strassen algorithm
    clock_gettime(CLOCK_REALTIME, &stop);
    total_time = (stop.tv_sec-start.tv_sec)
	+0.000000001*(stop.tv_nsec-start.tv_nsec);
    
    // calculate the execution time for the naive algorithm
    //printf("Start Naive Matrix Multiplication!\n");
    naive(n, a, b, prod_naive);
    if(DEBUG) print(n, prod_naive);
    clock_gettime(CLOCK_REALTIME, &stop_naive);
    total_time_naive = (stop_naive.tv_sec-stop.tv_sec)
	+0.000000001*(stop_naive.tv_nsec-stop.tv_nsec);

    // check answer here
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if (prod_naive[i][j] != prod[i][j]) error = 1;
        }
    }

    if (error != 0) printf("Incorrect Answer!!\n");

    printf("Matrix Size (nxn) = %d, Threads = %d, error = %d, time_strassen (sec) = %8.5f, time_naive = %8.5f\n", 
	    n, num_threads, error, total_time, total_time_naive);

    free(a); free(b); free(prod); free(prod_naive);

    return 0;
}