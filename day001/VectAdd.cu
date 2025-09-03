#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

// GPU kernel function
// kernel<<< gridDim, blockDim, sharedMemSize, stream >>>(args...);
// 

__global__ void vectorAdd(const float* A , const float *B, float *C, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x; //
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

int main(){

    const int N = 1024; // elements in vec
    const int size = N * sizeof(int); // total size of vectors in bytes

    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    for(int i=0; i < N; i++){
        h_A[i] = 1;
        h_B[i] = i;
    }

    float *d_A, *d_B, *d_C;

    // 在GPU上分配内存
    cudaMalloc((void**)&d_A, size); 
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // cudaMemcpy 
    // cudaMemcpyHostToDevice（CPU → GPU）
    // cudaMemcpyDeviceToHost（GPU → CPU）
    // cudaMemcpyDeviceToDevice（GPU → GPU）
    // cudaMemcpyHostToHost（CPU → CPU）
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = ( N + threadsPerBlock -1) / threadsPerBlock;

    // addOne<<<1, N>>>(d_A, N); // 总共线程数 = gridDim * blockDim = 1 * N = N
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); // dst, src

    for (int i = N-20; i < N; i++) {
        // printf("%lf", h_C[i]);
        printf("%d\n", (int)h_C[i]);
        // std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }
    printf("\n");

    // free pts
    cudaFree(d_A); // free malloc memory
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A; 
    delete[] h_B; 
    delete[] h_C;


    printf("Vector Add Done!\n");
    
}
