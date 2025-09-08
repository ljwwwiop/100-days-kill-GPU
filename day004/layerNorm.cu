#include <iostream>
#include <cuda_runtime.h>
#include <cmath>


__global__ void LayerNorm(float *A, float *B, const int rows, const int cols){
    int row = blockIdx.x;   // 每个 block 处理一行
    
    if (row < rows){
        extern __shared__ float shared[]; 
        float *row_data = shared;       // 存一行数据
        float *reduce_buf = shared + cols; // 存归约的中间结果

        // === 1. load row into shared memory ===
        float local_sum = 0.0f;
        for(int col = threadIdx.x; col < cols; col += blockDim.x){
            float v = A[row * cols + col];
            row_data[col] = v;
            local_sum += v;
        }

        // === 2. reduction for mean ===
        reduce_buf[threadIdx.x] = local_sum;
        __syncthreads();

        // block 内归约 (假设 blockDim.x >= cols, 简化版)
        for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
            if(threadIdx.x < stride){
                reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
            }
            __syncthreads();
        }
        float mean = reduce_buf[0] / cols;
        __syncthreads();

        // === 3. compute variance ===
        float local_var = 0.0f;
        for(int col = threadIdx.x; col < cols; col += blockDim.x){
            float diff = row_data[col] - mean;
            local_var += diff * diff;
        }
        reduce_buf[threadIdx.x] = local_var;
        __syncthreads();

        for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
            if(threadIdx.x < stride){
                reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
            }
            __syncthreads();
        }
        float var = reduce_buf[0] / cols;
        float std = sqrtf(var + 1e-5);
        __syncthreads();

        // === 4. normalize + write back ===
        for(int col = threadIdx.x; col < cols; col += blockDim.x){
            B[row * cols + col] = (row_data[col] - mean) / std;
        }
    }
}

int main(){
    
    const int rows = 4, cols = 4;

    float *A, *B;

    // malloc
    A = (float*)malloc(rows * cols * sizeof(float));
    B = (float*)malloc(rows * cols * sizeof(float));


    // init array
    for(int i=0; i< rows; i++){
        for(int j=0;j < cols; j++){
            A[i * cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    

    // cudaMemcpy
    float *d_a, *d_b;
    cudaMalloc(&d_a, rows * cols * sizeof(float));
    cudaMalloc(&d_b, rows * cols * sizeof(float));

    cudaMemcpy(d_a, A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    // int blockSize = 256;
    // int gridsize = (rows + blockSize - 1) / blockSize;
    // size_t shared_memory_size = cols * sizeof(float);

    int blockSize = 32;  // 一行用 32 个线程足够
    int gridSize = rows; // 每个 block 处理一行
    size_t shared_memory_size = cols * sizeof(float);
    LayerNorm<<<gridSize, blockSize, shared_memory_size>>>(d_a, d_b, rows, cols);


    // Synchronize device
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_b, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printf("A:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", A[i * cols + j]);
        }
        printf("\n");
    }

    printf("\nB:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", B[i * cols + j]);
        }
        printf("\n");
    }

    // cudaFree
    cudaFree(d_a);
    cudaFree(d_b);

    // delete free
    free(A);
    free(B);


    return 0;
}

// A:
// 0.84 0.39 0.78 0.80 
// 0.91 0.20 0.34 0.77 
// 0.28 0.55 0.48 0.63 
// 0.36 0.51 0.95 0.92 

// B:
// 0.76 -1.72 0.44 0.52 
// 1.21 -1.20 -0.74 0.73 
// -1.58 0.53 -0.05 1.10 
// -1.27 -0.68 1.05 0.91 