#include <iostream>
#include <cuda_runtime.h>

// Matrix Add and Mul
__global__ void matAdd(float *A, float *B, float *C, int H, int W){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < H && col < W){
        int idx = row * H + col;
        C[idx] = A[idx] + B[idx];
    }

}

// [2,3] * [3,2] = [2,2]
__global__ void MatMul(float *A, float *B, float *C, int M, int K, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < M && col < N){
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }

}



int main(){
    
    int M = 2;
    int K = 3;
    int N = 2;

    // init array
    int s = 6;
    float h_A[s] = {2,3,3,1,2,2};
    float h_B[s] = {1,1,1,2,2,2};
    // float h_C[s] = {};
    float h_C_add[6], h_C_mul[4];

    float *d_A, *d_B, *d_C;

    // matrix A(2x3) + B(2x3)
    // cuda malloc
    cudaMalloc((void**)&d_A, s * sizeof(float));
    cudaMalloc((void**)&d_B, s * sizeof(float));
    cudaMalloc((void**)&d_C, s * sizeof(float));
    cudaMemcpy(d_A, h_A, s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, s * sizeof(float), cudaMemcpyHostToDevice);

    // cuda thread
    dim3 block(16, 16); // 16 * 16 = 256 线程, (x,y)
    dim3 grid((3+15)/16, (2+15)/16); // grid.x = (18/16) = 1 , grid.y = (17/16) = 1, (3x2)
    
    // mat add
    matAdd<<<grid, block>>>(d_A, d_B, d_C, M, K);
    
    // printf("Matrix Add (2x3):\n");
    // for (int i = 0; i < 6; i++) {
    //     printf("%.1f ", h_A[i]);
    //     if ((i+1)%3==0) printf("\n");
    // }
    cudaMemcpy(h_C_add, d_C, s * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Matrix Add (2x3):\n");
    for (int i = 0; i < 6; i++) {
        printf("%.1f ", h_C_add[i]);
        if ((i+1)%3==0) printf("\n");
    }

    // mat mul
    cudaFree(d_C);

    cudaMalloc((void**)&d_C, 4 * sizeof(float));
    dim3 gridMul((N+15)/16, (M+15)/16);
    MatMul<<<gridMul, block>>>(d_A, d_B, d_C, M, K, N);
    cudaMemcpy(h_C_mul, d_C, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nMatrix Mul (2x2):\n");
    for (int i = 0; i < 4; i++) {
        printf("%.1f ", h_C_mul[i]);
        if ((i+1)%2==0) printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // delete[] h_A;
    // delete[] h_B;
    // delete[] h_C_add;
    // delete[] h_C_mul;

    return 0;
}

