#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err__ = (call);                                             \
    if (err__ != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err__));                                  \
      exit(EXIT_FAILURE);                                                   \
    }                                                                      \
  } while (0)

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 512
#endif

#ifndef BLOCKS_PER_SM
#define BLOCKS_PER_SM 16
#endif

static inline int div_up(int a, int b) { return (a + b - 1) / b; }
static inline int min_int(int a, int b) { return a < b ? a : b; }

bool checkResults(float *gold, float *d_data, int dimx, int dimy,
                  float rel_tol) {
  for (int iy = 0; iy < dimy; ++iy) {
    for (int ix = 0; ix < dimx; ++ix) {
      int idx = iy * dimx + ix;

      float gdata = gold[idx];
      float ddata = d_data[idx];

      if (isnan(gdata) || isnan(ddata)) {
        printf("Nan detected: gold %f, device %f\n", gdata, ddata);
        return false;
      }

      float rdiff;
      if (fabsf(gdata) == 0.f)
        rdiff = fabsf(ddata);
      else
        rdiff = fabsf(gdata - ddata) / fabsf(gdata);

      if (rdiff > rel_tol) {
        printf("Error solutions don't match at iy=%d, ix=%d.\n", iy, ix);
        printf("gold: %f, device: %f\n", gdata, ddata);
        printf("rdiff: %f\n", rdiff);
        return false;
      }
    }
  }
  return true;
}

void computeCpuResults(float *g_data, int dimx, int dimy, int niterations,
                       int nreps) {
  for (int r = 0; r < nreps; r++) {
#pragma omp parallel for
    for (int iy = 0; iy < dimy; ++iy) {
      for (int ix = 0; ix < dimx; ++ix) {
        int idx = iy * dimx + ix;

        float value = g_data[idx];

        for (int i = 0; i < niterations; i++) {
          if (ix % 4 == 0) {
            value += sqrtf(logf(value) + 1.f);
          } else if (ix % 4 == 1) {
            value += sqrtf(cosf(value) + 1.f);
          } else if (ix % 4 == 2) {
            value += sqrtf(sinf(value) + 1.f);
          } else {
            value += sqrtf(tanf(value) + 1.f);
          }
        }
        g_data[idx] = value;
      }
    }
  }
}

__device__ __forceinline__ float baseline_step(float value, int ix) {
  if ((ix & 3) == 0) {
    value += sqrtf(logf(value) + 1.f);
  } else if ((ix & 3) == 1) {
    value += sqrtf(cosf(value) + 1.f);
  } else if ((ix & 3) == 2) {
    value += sqrtf(sinf(value) + 1.f);
  } else {
    value += sqrtf(tanf(value) + 1.f);
  }
  return value;
}

__device__ __forceinline__ float fast_log_step(float value) {
  return value + sqrtf(__logf(value) + 1.f);
}

__device__ __forceinline__ float fast_cos_step(float value) {
  return value + sqrtf(__cosf(value) + 1.f);
}

__device__ __forceinline__ float fast_sin_step(float value) {
  return value + sqrtf(__sinf(value) + 1.f);
}

__device__ __forceinline__ float fast_tan_step(float value) {
  return value + sqrtf(__tanf(value) + 1.f);
}

template <int NITER>
__device__ __forceinline__ float apply_fast_log(float value) {
#pragma unroll
  for (int i = 0; i < NITER; ++i) value = fast_log_step(value);
  return value;
}

template <int NITER>
__device__ __forceinline__ float apply_fast_cos(float value) {
#pragma unroll
  for (int i = 0; i < NITER; ++i) value = fast_cos_step(value);
  return value;
}

template <int NITER>
__device__ __forceinline__ float apply_fast_sin(float value) {
#pragma unroll
  for (int i = 0; i < NITER; ++i) value = fast_sin_step(value);
  return value;
}

template <int NITER>
__device__ __forceinline__ float apply_fast_tan(float value) {
#pragma unroll
  for (int i = 0; i < NITER; ++i) value = fast_tan_step(value);
  return value;
}

__device__ __forceinline__ float apply_fast_dynamic(float value, int lane,
                                                    int niterations) {
  for (int i = 0; i < niterations; ++i) {
    if (lane == 0) {
      value = fast_log_step(value);
    } else if (lane == 1) {
      value = fast_cos_step(value);
    } else if (lane == 2) {
      value = fast_sin_step(value);
    } else {
      value = fast_tan_step(value);
    }
  }
  return value;
}

__global__ void kernel_original(float *g_data, int dimx, int dimy,
                                int niterations) {
  for (int iy = blockIdx.y * blockDim.y + threadIdx.y; iy < dimy;
       iy += blockDim.y * gridDim.y) {
    for (int ix = blockIdx.x * blockDim.x + threadIdx.x; ix < dimx;
         ix += blockDim.x * gridDim.x) {
      int idx = iy * dimx + ix;
      float value = g_data[idx];

      for (int i = 0; i < niterations; i++) {
        value = baseline_step(value, ix);
      }
      g_data[idx] = value;
    }
  }
}

template <int NITER>
__global__ void kernel_scalar_coalesced(float *__restrict__ g_data, int total,
                                        int dimx) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; idx < total; idx += stride) {
    int ix = idx - (idx / dimx) * dimx;
    int lane = ix & 3;

    float value = g_data[idx];
    if (lane == 0) {
      value = apply_fast_log<NITER>(value);
    } else if (lane == 1) {
      value = apply_fast_cos<NITER>(value);
    } else if (lane == 2) {
      value = apply_fast_sin<NITER>(value);
    } else {
      value = apply_fast_tan<NITER>(value);
    }
    g_data[idx] = value;
  }
}

__global__ void kernel_scalar_coalesced_dynamic(float *__restrict__ g_data,
                                                int total, int dimx,
                                                int niterations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; idx < total; idx += stride) {
    int ix = idx - (idx / dimx) * dimx;
    int lane = ix & 3;
    g_data[idx] = apply_fast_dynamic(g_data[idx], lane, niterations);
  }
}

template <int NITER>
__global__ void kernel_vector4_fast(float4 *__restrict__ g_data4, int groups) {
  int group = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; group < groups; group += stride) {
    float4 value = g_data4[group];
    value.x = apply_fast_log<NITER>(value.x);
    value.y = apply_fast_cos<NITER>(value.y);
    value.z = apply_fast_sin<NITER>(value.z);
    value.w = apply_fast_tan<NITER>(value.w);
    g_data4[group] = value;
  }
}

__global__ void kernel_vector4_fast_dynamic(float4 *__restrict__ g_data4,
                                            int groups, int niterations) {
  int group = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; group < groups; group += stride) {
    float4 value = g_data4[group];
    for (int i = 0; i < niterations; ++i) {
      value.x = fast_log_step(value.x);
      value.y = fast_cos_step(value.y);
      value.z = fast_sin_step(value.z);
      value.w = fast_tan_step(value.w);
    }
    g_data4[group] = value;
  }
}

enum KernelVariant {
  VARIANT_ORIGINAL = 0,
  VARIANT_SCALAR_COALESCED = 1,
  VARIANT_VECTOR4_FAST = 2,
};

const char *variant_name(KernelVariant variant) {
  switch (variant) {
    case VARIANT_ORIGINAL:
      return "original_row_stride";
    case VARIANT_SCALAR_COALESCED:
      return "scalar_coalesced_fast";
    case VARIANT_VECTOR4_FAST:
      return "vector4_coalesced_fast";
    default:
      return "unknown";
  }
}

int get_sm_count() {
  static int cached_sms = 0;
  if (cached_sms == 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cached_sms = prop.multiProcessorCount;
  }
  return cached_sms;
}

int tuned_grid_size(int work_items, int block_size) {
  int max_grid = get_sm_count() * BLOCKS_PER_SM;
  return min_int(div_up(work_items, block_size), max_grid);
}

void launch_variant(KernelVariant variant, float *d_data, int dimx, int dimy,
                    int niterations) {
  int total = dimx * dimy;
  int block_size = THREADS_PER_BLOCK;

  if (variant == VARIANT_ORIGINAL) {
    dim3 block(1, 32);
    dim3 grid(1, get_sm_count());
    kernel_original<<<grid, block>>>(d_data, dimx, dimy, niterations);
    return;
  }

  if (variant == VARIANT_SCALAR_COALESCED) {
    dim3 block(block_size);
    dim3 grid(tuned_grid_size(total, block_size));
    if (niterations == 5) {
      kernel_scalar_coalesced<5><<<grid, block>>>(d_data, total, dimx);
    } else {
      kernel_scalar_coalesced_dynamic<<<grid, block>>>(d_data, total, dimx,
                                                       niterations);
    }
    return;
  }

  bool vector_safe = ((dimx & 3) == 0) && ((total & 3) == 0) &&
                     ((((uintptr_t)d_data) & (sizeof(float4) - 1)) == 0);
  if (variant == VARIANT_VECTOR4_FAST && vector_safe) {
    int groups = total / 4;
    dim3 block(block_size);
    dim3 grid(tuned_grid_size(groups, block_size));
    float4 *d_data4 = reinterpret_cast<float4 *>(d_data);
    if (niterations == 5) {
      kernel_vector4_fast<5><<<grid, block>>>(d_data4, groups);
    } else {
      kernel_vector4_fast_dynamic<<<grid, block>>>(d_data4, groups,
                                                   niterations);
    }
    return;
  }

  dim3 block(block_size);
  dim3 grid(tuned_grid_size(total, block_size));
  kernel_scalar_coalesced_dynamic<<<grid, block>>>(d_data, total, dimx,
                                                   niterations);
}

void launchKernel(float *d_data, int dimx, int dimy, int niterations) {
  launch_variant(VARIANT_VECTOR4_FAST, d_data, dimx, dimy, niterations);
}

float timing_experiment(KernelVariant variant, float *d_data, int dimx,
                        int dimy, int niterations, int nreps) {
  float elapsed_time_ms = 0.0f;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, 0));
  for (int i = 0; i < nreps; i++) {
    launch_variant(variant, d_data, dimx, dimy, niterations);
  }
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_ms, start, stop));
  elapsed_time_ms /= nreps;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return elapsed_time_ms;
}

bool verify_variant(KernelVariant variant, float *d_data, float *h_data,
                    float *h_gold, const float *h_initial, int dimx, int dimy,
                    int niterations, int nbytes, float rel_tol) {
  memcpy(h_gold, h_initial, nbytes);
  CUDA_CHECK(cudaMemcpy(d_data, h_initial, nbytes, cudaMemcpyHostToDevice));
  launch_variant(variant, d_data, dimx, dimy, niterations);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_data, d_data, nbytes, cudaMemcpyDeviceToHost));

  computeCpuResults(h_gold, dimx, dimy, niterations, 1);
  return checkResults(h_gold, h_data, dimx, dimy, rel_tol);
}

float benchmark_variant(KernelVariant variant, float *d_data,
                        const float *h_initial, int dimx, int dimy,
                        int niterations, int nreps, int nbytes) {
  CUDA_CHECK(cudaMemcpy(d_data, h_initial, nbytes, cudaMemcpyHostToDevice));
  launch_variant(variant, d_data, dimx, dimy, niterations);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(d_data, h_initial, nbytes, cudaMemcpyHostToDevice));
  return timing_experiment(variant, d_data, dimx, dimy, niterations, nreps);
}

int main() {
  int dimx = 8 * 1024;
  int dimy = 8 * 1024;

  int nreps = 10;
  int niterations = 5;
  int total = dimx * dimy;
  int nbytes = total * (int)sizeof(float);

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s, compute capability %d.%d, SMs %d\n", prop.name, prop.major,
         prop.minor, prop.multiProcessorCount);

  float *d_data = 0, *h_data = 0, *h_gold = 0, *h_initial = 0;
  CUDA_CHECK(cudaMalloc((void **)&d_data, nbytes));
  printf("allocated %.2f MB on GPU\n", nbytes / (1024.f * 1024.f));

  h_data = (float *)malloc(nbytes);
  h_gold = (float *)malloc(nbytes);
  h_initial = (float *)malloc(nbytes);
  if (0 == h_data || 0 == h_gold || 0 == h_initial) {
    printf("couldn't allocate CPU memory\n");
    return -2;
  }
  printf("allocated %.2f MB on CPU\n", 3.0f * nbytes / (1024.f * 1024.f));

  srand(1234);
  for (int i = 0; i < total; i++) {
    h_initial[i] = 1.0f + 0.01f * (float)rand() / (float)RAND_MAX;
  }

  const KernelVariant variants[] = {
      VARIANT_ORIGINAL,
      VARIANT_SCALAR_COALESCED,
      VARIANT_VECTOR4_FAST,
  };
  const int variant_count = sizeof(variants) / sizeof(variants[0]);
  float rel_tol = .001f;
  bool all_pass = true;

  printf("variant,correct,time_ms\n");
  for (int i = 0; i < variant_count; ++i) {
    KernelVariant variant = variants[i];
    bool pass = verify_variant(variant, d_data, h_data, h_gold, h_initial, dimx,
                               dimy, niterations, nbytes, rel_tol);
    float elapsed_time_ms =
        benchmark_variant(variant, d_data, h_initial, dimx, dimy, niterations,
                          nreps, nbytes);
    printf("%s,%s,%8.2f\n", variant_name(variant), pass ? "yes" : "no",
           elapsed_time_ms);
    all_pass = all_pass && pass;
  }

  printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));

  if (d_data) CUDA_CHECK(cudaFree(d_data));
  if (h_data) free(h_data);
  if (h_gold) free(h_gold);
  if (h_initial) free(h_initial);

  CUDA_CHECK(cudaDeviceReset());

  return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
