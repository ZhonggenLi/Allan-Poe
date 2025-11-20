#pragma once
#include <stdio.h>
#include<chrono>

#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <set>
#include <utility>

#include <curand_kernel.h>
#include <random>
#include <cuda_fp16.h>
#include <cstring>

#include <cstddef>
#include <mutex>
#include <bitset>
#include <cmath>
#include <boost/dynamic_bitset.hpp>
#include <stack>

#define HASHLEN 1024
#define HASHSIZE 10
#define DIM 1024

using namespace std;
using namespace nvcuda;

__device__ __forceinline__ void swap(float &a, float &b){
    const float t = a;
    a = b;
    b = t;
}

__device__ void __forceinline__ swap_ids(unsigned &a, unsigned &b){
    const unsigned t = a;
    a = b;
    b = t;
}

__device__ void __forceinline__ swap_dis_and_id(float &a, float &b, unsigned &c, unsigned &d){
    const float t = a;
    a = b;
    b = t;
    const unsigned s = c;
    c = d;
    d = s;
}

__device__ void __forceinline__ swap_bool(bool &a, bool &b){
    bool t = a;
    a = b;
    b = t;
}

struct __align__(8) half4 {
  half2 x, y;
};

__device__ __forceinline__ half4 BitCast(const float2& src) noexcept {
  half4 dst;
  std::memcpy(&dst, &src, sizeof(half4));
  return dst;
}

__device__ __forceinline__ half4 Load(const half* address) {
    float2 x = __ldg(reinterpret_cast<const float2*>(address));
    return BitCast(x);
}

__global__ __forceinline__ void f2h(float* data, half* data_half, unsigned points_num){
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(unsigned i = tid; i < points_num * DIM; i += blockDim.x * gridDim.x){
        data_half[i] = __float2half(data[i]);
    }
}

__device__ __forceinline__ void bitonic_sort_id_by_dis(float* shared_arr, unsigned* ids, bool* visit, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(unsigned stride = 1; stride < len; stride <<= 1){
        for(unsigned step = stride; step > 0; step >>= 1){
            for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                unsigned a = 2 * step * (k / step);
                unsigned b = k % step;
                unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                unsigned d = a + b + step;
                if(d < len && shared_arr[u] > shared_arr[d]){
                    swap(shared_arr[u],shared_arr[d]);
                    swap_ids(ids[u],ids[d]);
                    swap_bool(visit[u], visit[d]);
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void bitonic_sort_id_and_dis(float* shared_arr, unsigned* ids, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(unsigned stride = 1; stride < len; stride <<= 1){
        for(unsigned step = stride; step > 0; step >>= 1){
            for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                unsigned a = 2 * step * (k / step);
                unsigned b = k % step;
                unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                unsigned d = a + b + step;
                if(d < len && shared_arr[u] > shared_arr[d]){
                    swap(shared_arr[u],shared_arr[d]);
                    swap_ids(ids[u],ids[d]);
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void bitonic_sort_new2(float* shared_arr, unsigned* ids, unsigned* ids2, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    // if(tid < len / 2){
        for(unsigned stride = 1; stride < len; stride <<= 1){
            for(unsigned step = stride; step > 0; step >>= 1){
                for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                    unsigned a = 2 * step * (k / step);
                    unsigned b = k % step;
                    unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                    unsigned d = a + b + step;
                    if(d < len && shared_arr[u] > shared_arr[d]){
                        swap(shared_arr[u],shared_arr[d]);
                        swap_ids(ids[u],ids[d]);
                        swap_ids(ids2[u],ids2[d]);
                    }
                }
                __syncthreads();
            }
        }
    // }
    // __syncthreads();
}

__device__ __forceinline__ void bitonic_sort_id_by_dis_no_explore(float* shared_arr, unsigned* ids, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(unsigned stride = 1; stride < len; stride <<= 1){
        for(unsigned step = stride; step > 0; step >>= 1){
            for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                const unsigned a = 2 * step * (k / step);
                const unsigned b = k % step;
                const unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                const unsigned d = a + b + step;
                if(d < len && shared_arr[u] > shared_arr[d]){
                    // swap(shared_arr[u],shared_arr[d]);
                    // swap_ids(ids[u],ids[d]);
                    swap_dis_and_id(shared_arr[u],shared_arr[d], ids[u],ids[d]);
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void bitonic_sort_id_by_dis_no_explore_descend(float* shared_arr, unsigned* ids, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(unsigned stride = 1; stride < len; stride <<= 1){
        for(unsigned step = stride; step > 0; step >>= 1){
            for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                const unsigned a = 2 * step * (k / step);
                const unsigned b = k % step;
                const unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                const unsigned d = a + b + step;
                if(d < len && shared_arr[u] < shared_arr[d]){
                    // swap(shared_arr[u],shared_arr[d]);
                    // swap_ids(ids[u],ids[d]);
                    swap_dis_and_id(shared_arr[u],shared_arr[d], ids[u],ids[d]);
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void bitonic_sort_id_by_dis_no_explore_second(float* shared_arr, unsigned* ids, unsigned len, unsigned stride){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(unsigned step = stride; step > 0; step >>= 1){
        for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
            const unsigned a = 2 * step * (k / step);
            const unsigned b = k % step;
            const unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
            const unsigned d = a + b + step;
            if(d < len && shared_arr[u] > shared_arr[d]){
                // swap(shared_arr[u],shared_arr[d]);
                // swap_ids(ids[u],ids[d]);
                swap_dis_and_id(shared_arr[u],shared_arr[d], ids[u],ids[d]);
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void bitonic_sort_id_new2(unsigned* shared_arr, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
        for(unsigned stride = 1; stride < len; stride <<= 1){
            for(unsigned step = stride; step > 0; step >>= 1){
                for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                    unsigned a = 2 * step * (k / step);
                    unsigned b = k % step;
                    unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                    unsigned d = a + b + step;
                    if(d < len && shared_arr[u] > shared_arr[d]){
                        swap_ids(shared_arr[u],shared_arr[d]);
                    }
                }
                __syncthreads();
            }
        }
}

__device__ __forceinline__ void bitonic_sort_by_id(float* shared_arr, unsigned* ids, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(unsigned stride = 1; stride < len; stride <<= 1){
        for(unsigned step = stride; step > 0; step >>= 1){
            for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                unsigned a = 2 * step * (k / step);
                unsigned b = k % step;
                unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                unsigned d = a + b + step;
                if(d < len && ids[u] > ids[d]){
                    swap(shared_arr[u],shared_arr[d]);
                    swap_ids(ids[u],ids[d]);
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void bitonic_sort_id_by_detour(unsigned* shared_arr, unsigned* ids, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(unsigned stride = 1; stride < len; stride <<= 1){
        for(unsigned step = stride; step > 0; step >>= 1){
            for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                unsigned a = 2 * step * (k / step);
                unsigned b = k % step;
                unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                unsigned d = a + b + step;
                if(d < len && shared_arr[u] > shared_arr[d]){
                    swap_ids(shared_arr[u],shared_arr[d]);
                    swap_ids(ids[u],ids[d]);
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void bitonic_sort_id_by_dis_kg(float* shared_arr, unsigned* ids, unsigned* ids2, unsigned* ids3, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(unsigned stride = 1; stride < len; stride <<= 1){
        for(unsigned step = stride; step > 0; step >>= 1){
            for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                const unsigned a = 2 * step * (k / step);
                const unsigned b = k % step;
                const unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                const unsigned d = a + b + step;
                if(d < len && shared_arr[u] > shared_arr[d]){
                    // swap(shared_arr[u],shared_arr[d]);
                    swap_ids(ids2[u],ids2[d]);
                    swap_ids(ids3[u],ids3[d]);
                    swap_dis_and_id(shared_arr[u],shared_arr[d], ids[u],ids[d]);
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ unsigned hash_insert(unsigned *hash_table, unsigned key){
    const unsigned bit_mask = HASHLEN - 1;
    unsigned index = ((key ^ (key >> HASHSIZE)) & bit_mask);
    const unsigned stride = 1;
    for(unsigned i = 0; i < HASHLEN; i++){
        const unsigned old = atomicCAS(&hash_table[index], 0xFFFFFFFF, key);
        if(old == 0xFFFFFFFF){
            return 1;
        }
        else if(old == key){
            return 0;
        }
        index = (index + stride) & bit_mask;
    }
    return 0;
}

__device__ __forceinline__ float cal_sparse_dist(unsigned* node_idx1, float* node_val1, unsigned num1, unsigned* node_idx2, float* node_val2, unsigned num2, unsigned laneid){
    float res_dist = 0.0;
    for(unsigned j = laneid; j < num1; j += blockDim.x){
        unsigned val = node_idx1[j];
        unsigned tmp = num2, res_id = 0;
        while (tmp > 1) {
            unsigned halfsize = tmp / 2;
            unsigned cand = node_idx2[res_id + halfsize];
            res_id += ((cand < val) ? halfsize : 0);
            tmp -= halfsize;
        }
        res_id += (node_idx2[res_id] < val);
        if(res_id <= (num2 - 1) && node_idx2[res_id] == val){
            res_dist += (node_val2[res_id] * node_val1[j]);
        }
    }
    return res_dist;
}

__device__ __forceinline__ float cal_bm25_dist(unsigned* node_idx1, float* node_val1, unsigned num1, unsigned* node_idx2, float* node_val2, unsigned num2, unsigned laneid, unsigned* key_hit){
    float res_dist = 0.0;
    for(unsigned j = laneid; j < num1; j += blockDim.x){
        unsigned val = node_idx1[j];
        unsigned tmp = num2, res_id = 0;
        while (tmp > 1) {
            unsigned halfsize = tmp / 2;
            unsigned cand = (node_idx2[res_id + halfsize] & 0x7FFFFFFF);
            res_id += ((cand < val) ? halfsize : 0);
            tmp -= halfsize;
        }
        res_id += ((node_idx2[res_id] & 0x7FFFFFFF) < val);
        if(res_id <= (num2 - 1) && (node_idx2[res_id] & 0x7FFFFFFF) == val){
            res_dist += (node_val2[res_id] * node_val1[j]);
            if((node_idx2[res_id] & 0x80000000) != 0){
                // res_dist *= 1.2;
                // atomicAdd(key_hit, 1);
                *key_hit = 1;
            }
        }
    }
    return res_dist;
}

__forceinline__ void load_data_bin(char* filename, float*& data, unsigned& num,unsigned& dim){// load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
    // in.read((char*)&dim,4);
    dim = 100;
    in.seekg(0,std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim) / 4);
    data = new float[num * dim * sizeof(float)];

    in.seekg(0,std::ios::beg);
    for(size_t i = 0; i < num; i++){
        // in.seekg(4,std::ios::cur);
        in.read((char*)(data+i*dim),dim*4);
    }
    in.close();
}

__forceinline__ void load_data(char* filename, float*& data, unsigned& num,unsigned& dim){// load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
    in.read((char*)&dim,4);
    // dim = 100;
    in.seekg(0,std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim+1) / 4);
    data = new float[num * dim * sizeof(float)];

    in.seekg(0,std::ios::beg);
    for(size_t i = 0; i < num; i++){
        in.seekg(4,std::ios::cur);
        in.read((char*)(data+i*dim),dim*4);
    }
    in.close();
}

__forceinline__ void load_sparse_data(char* filename, unsigned* &sparse_off, unsigned* &sparse_idx, float* &sparse_val, unsigned num, unsigned& non_zero_num){
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
    // unsigned non_zero_num;
    in.read((char*)(&non_zero_num),4);

    cout << "Non zero num: " << non_zero_num << endl;

    sparse_off = new unsigned[num];
    in.read((char*)(sparse_off),num*4);

    sparse_idx = new unsigned[non_zero_num];
    sparse_val = new float[non_zero_num];

    in.read((char*)(sparse_idx),non_zero_num*4);
    in.read((char*)(sparse_val),non_zero_num*4);

    in.close();
}