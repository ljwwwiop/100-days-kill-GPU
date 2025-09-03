#include <iostream>
#include <cuda_runtime.h>

// CPU and GPU
// __host__ __device__ int add(int a, int b) {
//     return a + b;
// }

// init sqrt function


__device__ int sqrtArr(int x){
    return x*x;
}

__global__ void voidKernel(int *d_A,int *d_B, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x; //
    if (tid < N) {
        d_B[tid] = sqrtArr(d_A[tid]);
    }
}

int main(){
    const int N = 20;
    int size = N * sizeof(int);

    int *h_A = new int[N];
    int *h_B = new int[N];

    for(int i = 0; i < N; i++){
        h_A[i] = 1 + i;
    }

    int *d_A, *d_B;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    voidKernel<<<1, N>>>(d_A, d_B, N);

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost); // dst, src
    for(int i = 0; i<N; i++){
        printf("%d\n", h_B[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);

    delete[] h_A;
    delete[] h_B;

    return 0;
}

// nvcc VectAdd.cu -o vec