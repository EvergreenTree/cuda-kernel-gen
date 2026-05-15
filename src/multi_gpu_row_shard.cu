#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
#define BLOCKS_PER_SM 32
#endif

#ifndef NREPS
#define NREPS 100
#endif

#ifndef DIMX
#define DIMX (8 * 1024)
#endif

#ifndef DIMY
#define DIMY (8 * 1024)
#endif

#ifndef MAX_MULTI_GPU_DEVICES
#define MAX_MULTI_GPU_DEVICES 16
#endif

static inline int div_up(int a, int b) { return (a + b - 1) / b; }

static double wall_time_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static float initial_value(int global_index) {
  unsigned int x = (unsigned int)global_index + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return 1.0f + 0.01f * ((float)(x & 0x00ffffffu) / 16777215.0f);
}

static float cpu_step(float value, int ix) {
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

static float cpu_expected(int global_index, int dimx) {
  int ix = global_index - (global_index / dimx) * dimx;
  float value = initial_value(global_index);
  for (int i = 0; i < 5; ++i) value = cpu_step(value, ix);
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

template <int NITER>
__global__ void kernel_vector4_fast(float4 *__restrict__ data4, int groups) {
  int group = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; group < groups; group += stride) {
    float4 value = data4[group];
    value.x = apply_fast_log<NITER>(value.x);
    value.y = apply_fast_cos<NITER>(value.y);
    value.z = apply_fast_sin<NITER>(value.z);
    value.w = apply_fast_tan<NITER>(value.w);
    data4[group] = value;
  }
}

struct Shard {
  int device;
  int row_start;
  int rows;
  int sms;
  size_t elements;
  size_t groups;
  size_t bytes;
  float *h_initial;
  float *h_output;
  float *d_data;
  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t stop;
};

struct TimedResult {
  float event_ms;
  double wall_ms;
};

static void init_shard(Shard *shard, int dimx) {
  for (int row = 0; row < shard->rows; ++row) {
    int global_row = shard->row_start + row;
    for (int ix = 0; ix < dimx; ++ix) {
      int local_index = row * dimx + ix;
      int global_index = global_row * dimx + ix;
      shard->h_initial[local_index] = initial_value(global_index);
    }
  }
}

static bool verify_shards(Shard *shards, int shard_count, int dimx,
                          float rel_tol) {
  for (int s = 0; s < shard_count; ++s) {
    Shard *shard = &shards[s];
    for (int row = 0; row < shard->rows; ++row) {
      int global_row = shard->row_start + row;
      for (int ix = 0; ix < dimx; ++ix) {
        int local_index = row * dimx + ix;
        int global_index = global_row * dimx + ix;
        float gold = cpu_expected(global_index, dimx);
        float got = shard->h_output[local_index];
        if (isnan(gold) || isnan(got)) return false;
        float rdiff =
            fabsf(gold) == 0.f ? fabsf(got) : fabsf(gold - got) / fabsf(gold);
        if (rdiff > rel_tol) {
          fprintf(stderr,
                  "multi-gpu mismatch gpu=%d row=%d ix=%d gold=%f got=%f "
                  "rdiff=%f\n",
                  shard->device, global_row, ix, gold, got, rdiff);
          return false;
        }
      }
    }
  }
  return true;
}

static void allocate_shards(Shard *shards, int shard_count, int dimx, int dimy,
                            const int *devices) {
  int base_rows = dimy / shard_count;
  int extra_rows = dimy % shard_count;
  int row_start = 0;

  for (int i = 0; i < shard_count; ++i) {
    Shard *shard = &shards[i];
    cudaDeviceProp prop;
    int rows = base_rows + (i < extra_rows ? 1 : 0);
    memset(shard, 0, sizeof(*shard));
    shard->device = devices[i];
    shard->row_start = row_start;
    shard->rows = rows;
    shard->elements = (size_t)rows * (size_t)dimx;
    shard->groups = shard->elements / 4;
    shard->bytes = shard->elements * sizeof(float);
    row_start += rows;

    CUDA_CHECK(cudaSetDevice(shard->device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, shard->device));
    shard->sms = prop.multiProcessorCount;
    CUDA_CHECK(cudaMalloc((void **)&shard->d_data, shard->bytes));
    CUDA_CHECK(cudaStreamCreate(&shard->stream));
    CUDA_CHECK(cudaEventCreate(&shard->start));
    CUDA_CHECK(cudaEventCreate(&shard->stop));

    shard->h_initial = (float *)malloc(shard->bytes);
    shard->h_output = (float *)malloc(shard->bytes);
    if (!shard->h_initial || !shard->h_output) {
      fprintf(stderr, "could not allocate host shard memory\n");
      exit(EXIT_FAILURE);
    }
    init_shard(shard, dimx);
  }
}

static void free_shards(Shard *shards, int shard_count) {
  for (int i = 0; i < shard_count; ++i) {
    Shard *shard = &shards[i];
    CUDA_CHECK(cudaSetDevice(shard->device));
    if (shard->d_data) CUDA_CHECK(cudaFree(shard->d_data));
    if (shard->start) CUDA_CHECK(cudaEventDestroy(shard->start));
    if (shard->stop) CUDA_CHECK(cudaEventDestroy(shard->stop));
    if (shard->stream) CUDA_CHECK(cudaStreamDestroy(shard->stream));
    if (shard->h_initial) free(shard->h_initial);
    if (shard->h_output) free(shard->h_output);
  }
}

static void launch_shard(Shard *shard) {
  int block = THREADS_PER_BLOCK;
  int grid = shard->sms * BLOCKS_PER_SM;
  int min_grid = div_up((int)shard->groups, block);
  if (grid > min_grid) grid = min_grid;
  kernel_vector4_fast<5><<<grid, block, 0, shard->stream>>>(
      reinterpret_cast<float4 *>(shard->d_data), (int)shard->groups);
}

static TimedResult run_sharded(Shard *shards, int shard_count, int nreps,
                               bool timed_gather) {
  float total_event_ms = 0.0f;
  double total_wall_ms = 0.0;

  for (int rep = 0; rep < nreps; ++rep) {
    for (int i = 0; i < shard_count; ++i) {
      Shard *shard = &shards[i];
      CUDA_CHECK(cudaSetDevice(shard->device));
      CUDA_CHECK(cudaMemcpyAsync(shard->d_data, shard->h_initial, shard->bytes,
                                 cudaMemcpyHostToDevice, shard->stream));
    }
    for (int i = 0; i < shard_count; ++i) {
      CUDA_CHECK(cudaSetDevice(shards[i].device));
      CUDA_CHECK(cudaStreamSynchronize(shards[i].stream));
    }

    double start_ms = wall_time_ms();
    for (int i = 0; i < shard_count; ++i) {
      Shard *shard = &shards[i];
      CUDA_CHECK(cudaSetDevice(shard->device));
      CUDA_CHECK(cudaEventRecord(shard->start, shard->stream));
      launch_shard(shard);
      if (timed_gather) {
        CUDA_CHECK(cudaMemcpyAsync(shard->h_output, shard->d_data, shard->bytes,
                                   cudaMemcpyDeviceToHost, shard->stream));
      }
      CUDA_CHECK(cudaEventRecord(shard->stop, shard->stream));
    }

    for (int i = 0; i < shard_count; ++i) {
      CUDA_CHECK(cudaSetDevice(shards[i].device));
      CUDA_CHECK(cudaEventSynchronize(shards[i].stop));
    }
    double end_ms = wall_time_ms();

    float max_event_ms = 0.0f;
    for (int i = 0; i < shard_count; ++i) {
      float elapsed = 0.0f;
      CUDA_CHECK(cudaSetDevice(shards[i].device));
      CUDA_CHECK(cudaEventElapsedTime(&elapsed, shards[i].start, shards[i].stop));
      if (elapsed > max_event_ms) max_event_ms = elapsed;
    }

    total_event_ms += max_event_ms;
    total_wall_ms += end_ms - start_ms;
  }

  TimedResult result;
  result.event_ms = total_event_ms / (float)nreps;
  result.wall_ms = total_wall_ms / (double)nreps;
  return result;
}

static void copy_outputs(Shard *shards, int shard_count) {
  for (int i = 0; i < shard_count; ++i) {
    Shard *shard = &shards[i];
    CUDA_CHECK(cudaSetDevice(shard->device));
    CUDA_CHECK(cudaMemcpy(shard->h_output, shard->d_data, shard->bytes,
                          cudaMemcpyDeviceToHost));
  }
}

int main() {
  int dimx = DIMX;
  int dimy = DIMY;
  int nreps = NREPS;
  int total_devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&total_devices));
  if (total_devices < 1) {
    fprintf(stderr, "No CUDA devices are visible.\n");
    return EXIT_FAILURE;
  }

  int multi_count =
      total_devices < MAX_MULTI_GPU_DEVICES ? total_devices : MAX_MULTI_GPU_DEVICES;
  int single_device[1] = {0};
  int multi_devices[MAX_MULTI_GPU_DEVICES];
  Shard single[1];
  Shard multi[MAX_MULTI_GPU_DEVICES];
  long long logical_kernel_bytes = (long long)dimx * (long long)dimy * 2LL *
                                   (long long)sizeof(float);
  long long logical_gather_bytes = logical_kernel_bytes +
                                   (long long)dimx * (long long)dimy *
                                       (long long)sizeof(float);

  for (int i = 0; i < multi_count; ++i) multi_devices[i] = i;

  allocate_shards(single, 1, dimx, dimy, single_device);
  allocate_shards(multi, multi_count, dimx, dimy, multi_devices);

  printf("variant,gpu_count,correct,kernel_or_total_ms,host_wall_ms,logical_bytes\n");

  TimedResult single_timing = run_sharded(single, 1, nreps, false);
  copy_outputs(single, 1);
  bool single_ok = verify_shards(single, 1, dimx, 0.001f);
  printf("single_gpu_kernel,1,%s,%.6f,%.6f,%lld\n",
         single_ok ? "yes" : "no", single_timing.event_ms,
         single_timing.wall_ms, logical_kernel_bytes);

  TimedResult shard_timing = run_sharded(multi, multi_count, nreps, false);
  copy_outputs(multi, multi_count);
  bool shard_ok = verify_shards(multi, multi_count, dimx, 0.001f);
  printf("row_sharded_kernel_no_gather,%d,%s,%.6f,%.6f,%lld\n", multi_count,
         shard_ok ? "yes" : "no", shard_timing.event_ms,
         shard_timing.wall_ms, logical_kernel_bytes);

  TimedResult gather_timing = run_sharded(multi, multi_count, nreps, true);
  bool gather_ok = verify_shards(multi, multi_count, dimx, 0.001f);
  printf("row_sharded_kernel_host_gather,%d,%s,%.6f,%.6f,%lld\n", multi_count,
         gather_ok ? "yes" : "no", gather_timing.event_ms,
         gather_timing.wall_ms, logical_gather_bytes);

  printf("partition,gpu,row_start,row_end,rows\n");
  for (int i = 0; i < multi_count; ++i) {
    printf("partition,%d,%d,%d,%d\n", multi[i].device, multi[i].row_start,
           multi[i].row_start + multi[i].rows, multi[i].rows);
  }
  printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));

  free_shards(single, 1);
  free_shards(multi, multi_count);
  CUDA_CHECK(cudaDeviceReset());

  return single_ok && shard_ok && gather_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
