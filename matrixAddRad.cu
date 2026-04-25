// System includes
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <string>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

// ============================================================================
//  KERNEL A — GLOBAL MEMORY, COALESCED ACCESS
//  (parametry: input A pointer, output C pointer, N, R, k)
// ============================================================================

template <int BLOCK_SIZE>
__global__ void MatrixAddRadCUDA_A(const float *A, float *C, int N, int R, int k)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int outWidth  = N - 2 * R;
    int outHeight = N - 2 * R;

    // global output coordinates
    int outX0 = (bx * BLOCK_SIZE + tx) * k;
    int outY  =  by * BLOCK_SIZE + ty;

    if (outY >= outHeight)
        return;

    // compute k outputs
    for (int t = 0; t < k; ++t)
    {
        int outX = outX0 + t;
        if (outX >= outWidth)
            break;

        int inXcenter = outX + R;
        int inYcenter = outY + R;

        float sum = 0.0f;

        // fully coalesced: threads in warp read adjacent A[] elements
        for (int dy = -R; dy <= R; ++dy)
        {
            int rowBase = (inYcenter + dy) * N;
            for (int dx = -R; dx <= R; ++dx)
            {
                sum += A[rowBase + (inXcenter + dx)];
            }
        }

        C[outY * outWidth + outX] = sum;
    }
}

// ============================================================================
//  KERNEL B — GLOBAL MEMORY, NON-COALESCED ACCESS (semantic preserved)
//  (parametry: input A pointer, output C pointer, N, R, k)
// ============================================================================

template <int BLOCK_SIZE>
__global__ void MatrixAddRadCUDA_B(const float *A, float *C, int N, int R, int k)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int outWidth  = N - 2 * R;
    int outHeight = N - 2 * R;

    // Permute tx inside block to create a non-coalesced mapping.
    const int half = BLOCK_SIZE / 2;
    int perm_tx = (tx % 2 == 0) ? (tx / 2) : (half + tx / 2);

    // global output coordinates (permuted in X)
    int outX0 = (bx * BLOCK_SIZE + perm_tx) * k;
    int outY  =  by * BLOCK_SIZE + ty;

    if (outY >= outHeight)
        return;

    // compute k outputs
    for (int t = 0; t < k; ++t)
    {
        int outX = outX0 + t;
        if (outX >= outWidth)
            break;

        int inXcenter = outX + R;
        int inYcenter = outY + R;

        float sum = 0.0f;

        // Correct semantics preserved: read A in row-major order,
        // but because outX mapping is permuted across threads,
        // accesses across a warp are non-coalesced.
        for (int dy = -R; dy <= R; ++dy)
        {
            int rowBase = (inYcenter + dy) * N;
            for (int dx = -R; dx <= R; ++dx)
            {
                sum += A[rowBase + (inXcenter + dx)];
            }
        }

        C[outY * outWidth + outX] = sum;
    }
}

// ============================================================================
//  KERNEL C — SHARED MEMORY, COALESCED ACCESS (fixed: dynamic shared memory,
//  correct tile dimensions, safe indexing)
//  (parametry: input A pointer, output C pointer, N, R, k)
// ============================================================================

template <int BLOCK_SIZE>
__global__ void MatrixAddRadCUDA_C(const float *A, float *C, int N, int R, int k)
{
    extern __shared__ float tile[]; // 1D shared buffer

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int outWidth  = N - 2 * R;
    int outHeight = N - 2 * R;

    // Coordinates of first output (in output-space) computed by this thread
    int blockOutX = bx * (BLOCK_SIZE * k);
    int blockOutY = by * BLOCK_SIZE;

    int outX0 = blockOutX + tx * k;
    int outY  = blockOutY + ty;

    // tile dimensions (include halo)
    int tileWidth  = BLOCK_SIZE * k + 2 * R; // covers BLOCK_SIZE*k outputs in X plus halos
    int tileHeight = BLOCK_SIZE + 2 * R;     // covers BLOCK_SIZE outputs in Y plus halos

    // // global coordinates (in A) of tile top-left (including halo) WRONG
    // int tileX0 = blockOutX - R;
    // int tileY0 = blockOutY - R;

    // global coordinates (in A) of tile top-left (including halo)
    // tile must start at the first output column/row for this block (blockOutX/blockOutY),
    // and extend to the right/down by BLOCK_SIZE*k + 2*R and BLOCK_SIZE + 2*R respectively.
    int tileX0 = blockOutX;
    int tileY0 = blockOutY;

    // Load tile into shared memory (coalesced loads by rows)
    for (int dy = ty; dy < tileHeight; dy += BLOCK_SIZE)
    {
        int globalY = tileY0 + dy;
        // clamp
        if (globalY < 0)
            globalY = 0;
        else if (globalY > N - 1)
            globalY = N - 1;

        int base = dy * tileWidth;
        for (int dx = tx; dx < tileWidth; dx += BLOCK_SIZE)
        {
            int globalX = tileX0 + dx;
            if (globalX < 0)
                globalX = 0;
            else if (globalX > N - 1)
                globalX = N - 1;

            tile[base + dx] = A[globalY * N + globalX];
        }
    }

    __syncthreads();

    if (outY >= outHeight)
        return;

    // Compute up to k outputs
    for (int t = 0; t < k; ++t)
    {
        int outX = outX0 + t;
        if (outX >= outWidth)
            break;

        // local coords inside shared memory (account for halo offset R)
        int lx = tx * k + t + R;
        int ly = ty + R;

        float sum = 0.0f;

        // Sum over window in shared memory
        for (int dy = -R; dy <= R; ++dy)
        {
            int rowBase = (ly + dy) * tileWidth;
            for (int dx = -R; dx <= R; ++dx)
            {
                sum += tile[rowBase + (lx + dx)];
            }
        }

        C[outY * outWidth + outX] = sum;
    }
}

// ============================================================================
//  KERNEL D — SHARED MEMORY WITH BANK CONFLICTS (fixed semantics)
//  (parametry: input A pointer, output C pointer, N, R, k)
// ============================================================================

template <int BLOCK_SIZE>
__global__ void MatrixAddRadCUDA_D(const float *A, float *C, int N, int R, int k)
{
    extern __shared__ float tile[]; // 1D shared buffer

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int outWidth  = N - 2 * R;
    int outHeight = N - 2 * R;

    int blockOutX = bx * (BLOCK_SIZE * k);
    int blockOutY = by * BLOCK_SIZE;

    int outX0 = blockOutX + tx * k;
    int outY  = blockOutY + ty;

    // tile dimensions (include halo)
    int tileWidth  = BLOCK_SIZE * k + 2 * R; // logical width
    int tileHeight = BLOCK_SIZE + 2 * R;     // logical height

    // global coordinates (in A) of tile top-left (including halo)
    // int tileX0 = blockOutX - R; WRONG
    // int tileY0 = blockOutY - R; WRONG
    int tileX0 = blockOutX;
    int tileY0 = blockOutY;

    // Load tile into shared memory, but store transposed:
    // logical element (dy,dx) will be stored at index (dx * tileHeight + dy).
    // Later we'll read using that transposed layout to retrieve logical (row,col).
    for (int dy = ty; dy < tileHeight; dy += BLOCK_SIZE)
    {
        int globalY = tileY0 + dy;
        if (globalY < 0)
            globalY = 0;
        else if (globalY > N - 1)
            globalY = N - 1;

        for (int dx = tx; dx < tileWidth; dx += BLOCK_SIZE)
        {
            int globalX = tileX0 + dx;
            if (globalX < 0)
                globalX = 0;
            else if (globalX > N - 1)
                globalX = N - 1;

            // store transposed: index = dx * tileHeight + dy
            tile[dx * tileHeight + dy] = A[globalY * N + globalX];
        }
    }

    __syncthreads();

    if (outY >= outHeight)
        return;

    // Compute up to k outputs
    for (int t = 0; t < k; ++t)
    {
        int outX = outX0 + t;
        if (outX >= outWidth)
            break;

        // local coords inside logical tile (with halo)
        int lx = tx * k + t + R;
        int ly = ty + R;

        float sum = 0.0f;

        // Read from transposed storage: logical (ly+dy, lx+dx) is at
        // index ( (lx+dx) * tileHeight + (ly+dy) ).
        for (int dy = -R; dy <= R; ++dy)
        {
            for (int dx = -R; dx <= R; ++dx)
            {
                sum += tile[(lx + dx) * tileHeight + (ly + dy)];
            }
        }

        C[outY * outWidth + outX] = sum;
    }
}

// ============================================================================
//  KERNEL DISPATCHER — SELECT A/B/C/D VARIANT
//  Now accepts (mode, d_A, d_C, N, R, k, stream) and ensures shared memory limit.
//  Returns the actual mode used (e.g., when fallback to 'a' occurs) and reports
//  required sharedBytes via out parameter.
// ============================================================================

template <int BLOCK_SIZE>
char LaunchMatrixAddKernel(
    char mode,
    const float *d_A,
    float *d_C,
    int N,
    int R,
    int k,
    cudaStream_t stream,
    size_t *sharedBytesOut)
{
    int outWidth  = N - 2 * R;
    int outHeight = N - 2 * R;

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    // Each thread computes k outputs along X
    int gridDimX = (outWidth  + BLOCK_SIZE * k - 1) / (BLOCK_SIZE * k);
    int gridDimY = (outHeight + BLOCK_SIZE     - 1) /  BLOCK_SIZE;

    dim3 grid(gridDimX, gridDimY);

    // determine device shared memory limit
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    size_t sharedLimit = prop.sharedMemPerBlock; // bytes

    // compute required shared memory for C/D variants
    size_t sharedBytes = 0;
    char actualMode = mode;
    if (mode == 'c' || mode == 'C' || mode == 'd' || mode == 'D')
    {
        int tileWidth  = BLOCK_SIZE * k + 2 * R;
        int tileHeight = BLOCK_SIZE + 2 * R;
        // ensure no overflow
        sharedBytes = (size_t)tileWidth * (size_t)tileHeight * sizeof(float);
        if (sharedBytes > sharedLimit)
        {
            // fallback to global coalesced kernel A to avoid exceeding shared memory limit
            printf(
                "WARNING: Requested shared memory (%zu B) exceeds device limit (%zu B). Falling back to global memory kernel A.\n",
                sharedBytes, (size_t)sharedLimit);
            actualMode = 'a';
            sharedBytes = 0;
        }
    }

    if (sharedBytesOut)
        *sharedBytesOut = sharedBytes;

    // Print a concise launch summary for diagnostics
    printf("Launching kernel mode='%c' (actual='%c') BLOCK=%d k=%d grid=(%d,%d) sharedBytes=%zu\n",
           mode, actualMode, BLOCK_SIZE, k, gridDimX, gridDimY, sharedBytes);

    switch (actualMode)
    {
        case 'a':
        case 'A':
            MatrixAddRadCUDA_A<BLOCK_SIZE><<<grid, threads, 0, stream>>>(
                d_A, d_C, N, R, k);
            break;

        case 'b':
        case 'B':
            MatrixAddRadCUDA_B<BLOCK_SIZE><<<grid, threads, 0, stream>>>(
                d_A, d_C, N, R, k);
            break;

        case 'c':
        case 'C':
            MatrixAddRadCUDA_C<BLOCK_SIZE><<<grid, threads, sharedBytes, stream>>>(
                d_A, d_C, N, R, k);
            break;

        case 'd':
        case 'D':
            MatrixAddRadCUDA_D<BLOCK_SIZE><<<grid, threads, sharedBytes, stream>>>(
                d_A, d_C, N, R, k);
            break;

        default:
            printf("ERROR: Unknown kernel mode '%c'. Use a/b/c/d.\n", mode);
            return actualMode;
    }

    // check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    return actualMode;
}

// ============================================================================
//  CPU REFERENCE IMPLEMENTATION
// ============================================================================

void MatrixAddRadCPU(float *C_ref, const float *A, int N, int R)
{
    int outWidth  = N - 2 * R;
    int outHeight = N - 2 * R;

    for (int y = 0; y < outHeight; ++y)
    {
        for (int x = 0; x < outWidth; ++x)
        {
            int inXcenter = x + R;
            int inYcenter = y + R;

            float sum = 0.0f;

            for (int dy = -R; dy <= R; ++dy)
            {
                int rowBase = (inYcenter + dy) * N;

                for (int dx = -R; dx <= R; ++dx)
                {
                    sum += A[rowBase + (inXcenter + dx)];
                }
            }

            C_ref[y * outWidth + x] = sum;
        }
    }
}

// ============================================================================
//  HOST FUNCTION — RUN TEST FOR A/B/C/D KERNEL
//  Note: adjusted calls to LaunchMatrixAddKernel to pass d_A then d_C.
//  Accepts nIter (number of repetitions) and optional csv filename to append results.
// ============================================================================

int MatrixAddRadHost(
    int block_size,
    int N,
    int R,
    int k,
    char mode,
    int nIter,
    const char *csv_filename)
{
    int outWidth  = N - 2 * R;
    int outHeight = N - 2 * R;

    if (outWidth <= 0 || outHeight <= 0)
    {
        printf("ERROR: N must be > 2R. N=%d R=%d\n", N, R);
        return EXIT_FAILURE;
    }

    // Host memory
    size_t memA = sizeof(float) * (size_t)N * (size_t)N;
    size_t memC = sizeof(float) * (size_t)outWidth * (size_t)outHeight;

    float *h_A, *h_C, *h_C_ref;

    checkCudaErrors(cudaMallocHost(&h_A,     memA));
    checkCudaErrors(cudaMallocHost(&h_C,     memC));
    checkCudaErrors(cudaMallocHost(&h_C_ref, memC));

    // Initialize input
    for (int i = 0; i < N * N; ++i)
        h_A[i] = 1.0f;   // simple predictable pattern

    // Device memory
    float *d_A, *d_C;
    checkCudaErrors(cudaMalloc(&d_A, memA));
    checkCudaErrors(cudaMalloc(&d_C, memC));

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Copy input to device
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, memA, cudaMemcpyHostToDevice, stream));

    // CUDA events for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    size_t sharedBytesUsed = 0;
    char actualMode = mode;

    // Warmup
    if (block_size == 8)
        actualMode = LaunchMatrixAddKernel<8>(mode, d_A, d_C, N, R, k, stream, &sharedBytesUsed);
    else if (block_size == 16)
        actualMode = LaunchMatrixAddKernel<16>(mode, d_A, d_C, N, R, k, stream, &sharedBytesUsed);
    else
        actualMode = LaunchMatrixAddKernel<32>(mode, d_A, d_C, N, R, k, stream, &sharedBytesUsed);

    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());

    // Start timing (kernel-only) using events
    checkCudaErrors(cudaEventRecord(start, stream));

    for (int i = 0; i < nIter; ++i)
    {
        if (block_size == 8)
            LaunchMatrixAddKernel<8>(mode, d_A, d_C, N, R, k, stream, &sharedBytesUsed);
        else if (block_size == 16)
            LaunchMatrixAddKernel<16>(mode, d_A, d_C, N, R, k, stream, &sharedBytesUsed);
        else
            LaunchMatrixAddKernel<32>(mode, d_A, d_C, N, R, k, stream, &sharedBytesUsed);
    }

    // Stop timing
    checkCudaErrors(cudaEventRecord(stop, stream));
    checkCudaErrors(cudaEventSynchronize(stop));

    float msec = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msec, start, stop)); // ms for all nIter

    float avgKernelMs = msec / (float)nIter;

    // Compute performance
    double window = (2 * R + 1) * (2 * R + 1);
    double ops = (double)outWidth * (double)outHeight * window;
    double gflops = (ops * 1e-9) / (avgKernelMs / 1000.0);

    printf(
        "Kernel %c (actual %c) | BLOCK=%d | k=%d | N=%d | R=%d | avg kernel time=%.3f ms | Perf=%.2f GFLOPS | sharedBytes=%zu\n",
        mode, actualMode, block_size, k, N, R, avgKernelMs, gflops, sharedBytesUsed);

    // Copy result back
    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, memC, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());

    // CPU reference with timing (host)
    double cpu_msec = 0.0;
    {
        auto cpu_start = std::chrono::high_resolution_clock::now();

        MatrixAddRadCPU(h_C_ref, h_A, N, R);

        auto cpu_end = std::chrono::high_resolution_clock::now();
        cpu_msec = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    }

    double cpu_gflops = (ops * 1e-9) / (cpu_msec / 1000.0);

    printf("CPU  | Time=%.3f ms | Perf=%.2f GFLOPS\n", cpu_msec, cpu_gflops);

    if (avgKernelMs > 0.0f)
    {
        double speedup = cpu_msec / (double)avgKernelMs; // CPU_time / GPU_time
        printf("Speedup (CPU_time / GPU_kernel_time) = %.2fx\n", speedup);
    }
    else
    {
        printf("GPU kernel time is zero or not measured; cannot compute speedup.\n");
    }

    // Validate — relative tolerance
    bool ok = true;
    double eps = 1e-4;

    for (int i = 0; i < outWidth * outHeight; ++i)
    {
        double diff = fabs(h_C[i] - h_C_ref[i]);
        double ref = fabs(h_C_ref[i]);
        if (diff > eps && diff > eps * ref)
        {
            printf("Mismatch at %d: GPU=%.6f CPU=%.6f (diff=%.6f)\n",
                   i, h_C[i], h_C_ref[i], diff);
            ok = false;
            break;
        }
    }

    printf("Result: %s\n", ok ? "PASS" : "FAIL");

    // Optionally append CSV line
    if (csv_filename != nullptr && csv_filename[0] != '\0')
    {
        FILE *f = fopen(csv_filename, "a");
        if (f)
        {
            // Header not written here; user may create header manually before first run.
            // CSV fields:
            // mode,actualMode,BS,N,R,k,nIter,sharedBytes,avgKernelMs,gflops,cpu_msec,cpu_gflops,valid
            fprintf(f, "%c,%c,%d,%d,%d,%d,%d,%zu,%.6f,%.6f,%.6f,%.6f,%s\n",
                    mode, actualMode, block_size, N, R, k, nIter,
                    sharedBytesUsed, avgKernelMs, gflops, cpu_msec, cpu_gflops,
                    ok ? "PASS" : "FAIL");
            fclose(f);
        }
        else
        {
            printf("Warning: cannot open csv file '%s' for append\n", csv_filename);
        }
    }

    // Cleanup
    cudaFreeHost(h_A);
    cudaFreeHost(h_C);
    cudaFreeHost(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

// =============================================================================
// AUTOMATED TESTS
// =============================================================================

// Helper: format double to string with comma as decimal separator.
static void format_double_comma(char *buf, size_t bufsize, double v, int precision = 6)
{
    // Write with dot
    char tmp[128];
    snprintf(tmp, sizeof(tmp), "%.*f", precision, v);
    // Replace '.' with ','
    size_t j = 0;
    for (size_t i = 0; i < strlen(tmp) && j + 1 < bufsize; ++i)
    {
        buf[j++] = (tmp[i] == '.') ? ',' : tmp[i];
    }
    buf[j] = '\0';
}

// Helper: append CSV header (if new file)
static void write_csv_header_if_needed(const char *fname)
{
    FILE *f = fopen(fname, "r");
    if (f)
    {
        // file exists, close and return
        fclose(f);
        return;
    }
    f = fopen(fname, "w");
    if (!f)
    {
        printf("Warning: cannot create CSV file '%s'\n", fname);
        return;
    }
    // Write header (semicolon-separated)
    fprintf(f,
        "package;N;R;BS;k;mode;actualMode;nIter;sharedBytes;avgKernelMs;gflops;cpuMs;cpuGflops;valid\n");
    fclose(f);
}

// Helper: safe random float generator in [0,1)
static inline float randf()
{
    return (float)rand() / (float)RAND_MAX;
}

// The batch test runner. Single argument: number of packages (distinct A).
// It writes results to "batch_results.csv" in the current directory.
void RunBatchTests(int packages)
{
    if (packages <= 0)
    {
        printf("RunBatchTests: packages must be > 0\n");
        return;
    }

    const char *csv_name = "batch_results.csv";
    write_csv_header_if_needed(csv_name);

    // Config lists (can be adjusted)
    const std::vector<int> N_list = {256, 512, 1024, 2048, 4096}; // keep moderate by default
    const std::vector<int> BS_list = {8, 16, 32};
    const std::vector<int> R_list = {1, 20}; // one smaller than BS, one larger
    const std::vector<int> k_list = {1, 2, 8};
    const std::vector<char> modes = {'a', 'b', 'c', 'd'};

    // Number of kernel repeats for averaging (per mode/config)
    const int kernelIters = 20;

    // Seed RNG for reproducibility
    unsigned int base_seed = 12345u;
    srand(base_seed);

    printf("RunBatchTests: packages=%d, kernelIters=%d\n", packages, kernelIters);
    printf("CSV output: %s\n", csv_name);

    // For each package generate new input A and run all configs.
    for (int pkg = 0; pkg < packages; ++pkg)
    {
        unsigned int seed = base_seed + pkg;
        srand(seed);

        printf("\n=== Package %d / %d (seed=%u) ===\n", pkg + 1, packages, seed);

        // For each N/R/BS/k combination
        for (int N : N_list)
        {
            for (int R : R_list)
            {
                if (N <= 2 * R) // skip invalid
                    continue;

                int outWidth = N - 2 * R;
                int outHeight = N - 2 * R;
                size_t memA = sizeof(float) * (size_t)N * (size_t)N;
                size_t memC = sizeof(float) * (size_t)outWidth * (size_t)outHeight;

                // Generate host input A for this package/config
                // We'll allocate pinned host memory for faster H2D.
                float *h_A = nullptr;
                float *h_C = nullptr;
                float *h_C_ref = nullptr;

                if (cudaHostAlloc((void **)&h_A, memA, cudaHostAllocDefault) != cudaSuccess ||
                    cudaHostAlloc((void **)&h_C, memC, cudaHostAllocDefault) != cudaSuccess ||
                    cudaHostAlloc((void **)&h_C_ref, memC, cudaHostAllocDefault) != cudaSuccess)
                {
                    printf("Warning: host allocation failed for N=%d; skipping this config.\n", N);
                    if (h_A) cudaFreeHost(h_A);
                    if (h_C) cudaFreeHost(h_C);
                    if (h_C_ref) cudaFreeHost(h_C_ref);
                    continue;
                }

                // Fill h_A with random values (re-seeded per package)
                for (size_t i = 0; i < (size_t)N * (size_t)N; ++i)
                    h_A[i] = randf();

                // Device allocations (d_A once per config so all modes use same device input)
                float *d_A = nullptr;
                float *d_C = nullptr;
                if (cudaMalloc((void **)&d_A, memA) != cudaSuccess ||
                    cudaMalloc((void **)&d_C, memC) != cudaSuccess)
                {
                    printf("Warning: device allocation failed for N=%d; skipping\n", N);
                    if (d_A) cudaFree(d_A);
                    if (d_C) cudaFree(d_C);
                    cudaFreeHost(h_A);
                    cudaFreeHost(h_C);
                    cudaFreeHost(h_C_ref);
                    continue;
                }

                // Copy input to device once
                cudaError_t err = cudaMemcpy(d_A, h_A, memA, cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    printf("Warning: cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
                    cudaFree(d_A);
                    cudaFree(d_C);
                    cudaFreeHost(h_A);
                    cudaFreeHost(h_C);
                    cudaFreeHost(h_C_ref);
                    continue;
                }
                
                        
                // CPU reference timing (use same h_A)
                auto cpu_t0 = std::chrono::high_resolution_clock::now();
                MatrixAddRadCPU(h_C_ref, h_A, N, R);
                auto cpu_t1 = std::chrono::high_resolution_clock::now();
                double cpuMs = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();


                // For each block size and k, do tests
                for (int BS : BS_list)
                {
                    for (int k : k_list)
                    {
                        // Skip combinations where BS not supported by launcher templates
                        if (!(BS == 8 || BS == 16 || BS == 32))
                            continue;


                        // For each mode (a,b,c,d) we will measure kernel andcompare to  CPU.
                        for (char mode : modes)
                        {
                            // Some configs may not be meaningful: e.g., shared kernel may fallback.
                            // We'll measure actualMode returned by launcher.
                            // Prepare CUDA stream and events
                            cudaStream_t stream;
                            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

                            cudaEvent_t start, stop;
                            cudaEventCreate(&start);
                            cudaEventCreate(&stop);

                            size_t sharedBytesUsed = 0;
                            char actualMode = mode;

                            // Warmup + measurement loop
                            // Use the templated launcher based on BS
                            // Record start
                            checkCudaErrors(cudaEventRecord(start, stream));

                            for (int iter = 0; iter < kernelIters; ++iter)
                            {
                                if (BS == 8)
                                    actualMode = LaunchMatrixAddKernel<8>(mode, d_A, d_C, N, R, k, stream, &sharedBytesUsed);
                                else if (BS == 16)
                                    actualMode = LaunchMatrixAddKernel<16>(mode, d_A, d_C, N, R, k, stream, &sharedBytesUsed);
                                else
                                    actualMode = LaunchMatrixAddKernel<32>(mode, d_A, d_C, N, R, k, stream, &sharedBytesUsed);

                                // optional: check kernel launch error immediately
                                cudaError_t lerr = cudaGetLastError();
                                if (lerr != cudaSuccess)
                                {
                                    printf("Kernel launch error (mode=%c): %s\n", mode, cudaGetErrorString(lerr));
                                }
                            }

                            // Record stop and synchronize
                            checkCudaErrors(cudaEventRecord(stop, stream));
                            checkCudaErrors(cudaEventSynchronize(stop));

                            float msecTotal = 0.0f;
                            checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop)); 
                            
                            double avgKernelMs = (double)msecTotal / (double)kernelIters;

                            
                            double window = (2 * R + 1) * (2 * R + 1);
                            double ops = (double)outWidth * (double)outHeight * window;
                            double gflops = (ops * 1e-9) / (avgKernelMs / 1000.0);
                            double cpu_gflops = (ops * 1e-9) / (cpuMs / 1000.0);

                            // Copy result back
                            checkCudaErrors(cudaMemcpyAsync(h_C, d_C, memC, cudaMemcpyDeviceToHost, stream));
                            checkCudaErrors(cudaStreamSynchronize(stream));


                            // Validation (relative tolerance)
                            bool ok = true;
                            double eps = 1e-4;
                            int bad_i = -1; //debbuging c/d
                            for (int i = 0; i < outWidth * outHeight; ++i)
                            {
                                double diff = fabs(h_C[i] - h_C_ref[i]);
                                double ref = fabs(h_C_ref[i]);
                                if (diff > eps && diff > eps * ref)
                                {
                                    ok = false;
                                    bad_i = i;
                                    break;
                                }
                            }
                            //debbuing c/d
                            if (!ok && bad_i >= 0)
{
    int y = bad_i / outWidth;
    int x = bad_i % outWidth;
    printf("First mismatch at out(%d,%d) idx=%d: GPU=%.6f CPU=%.6f\n",
           x, y, bad_i, h_C[bad_i], h_C_ref[bad_i]);

    // Print small neighborhood from A safely
    int cx = x + R;
    int cy = y + R;
    printf("Neighborhood A around center (%d,%d) (clamped to bounds):\n", cx, cy);
    for (int yy = cy - R; yy <= cy + R; ++yy)
    {
        for (int xx = cx - R; xx <= cx + R; ++xx)
        {
            if (xx >= 0 && xx < N && yy >= 0 && yy < N)
                printf("%.6f ", h_A[yy * N + xx]);
            else
                printf("OUT ");
        }
        printf("\n");
    }

    // Print small block of outputs around mismatch for both GPU and CPU (safe)
    int dumpRadius = 2;
    printf("GPU outputs around mismatch (out coords):\n");
    for (int yy = max(0, y - dumpRadius); yy <= min(outHeight - 1, y + dumpRadius); ++yy)
    {
        for (int xx = max(0, x - dumpRadius); xx <= min(outWidth - 1, x + dumpRadius); ++xx)
            printf("G:%.6f ", h_C[yy * outWidth + xx]);
        printf("\n");
    }
    printf("CPU outputs around mismatch (out coords):\n");
    for (int yy = max(0, y - dumpRadius); yy <= min(outHeight - 1, y + dumpRadius); ++yy)
    {
        for (int xx = max(0, x - dumpRadius); xx <= min(outWidth - 1, x + dumpRadius); ++xx)
            printf("C:%.6f ", h_C_ref[yy * outWidth + xx]);
        printf("\n");
    }
}
else if (!ok)
{
    // defensive: mismatch flagged but bad_i invalid (shouldn't happen)
    printf("Validation detected mismatch but index is invalid (bad_i=%d). Skipping neighborhood dump.\n", bad_i);
}

                            // Print to console summary
                            printf("pkg=%d N=%d R=%d BS=%d k=%d mode=%c act=%c avgKernel=%.3f ms GFLOPS=%.2f cpu=%.3f ms cpuGFLOPS=%.2f valid=%s sharedBytes=%zu\n",
                                pkg, N, R, BS, k, mode, actualMode, avgKernelMs, gflops, cpuMs, cpu_gflops,
                                ok ? "PASS" : "FAIL", sharedBytesUsed);

                            // Append CSV line (semicolon-separated, decimals with comma)
                            FILE *f = fopen(csv_name, "a");
                            if (f)
                            {
                                char buf_avgKernel[64], buf_gflops[64], buf_cpuMs[64], buf_cpuGflops[64];
                                format_double_comma(buf_avgKernel, sizeof(buf_avgKernel), avgKernelMs, 6);
                                format_double_comma(buf_gflops, sizeof(buf_gflops), gflops, 6);
                                format_double_comma(buf_cpuMs, sizeof(buf_cpuMs), cpuMs, 6);
                                format_double_comma(buf_cpuGflops, sizeof(buf_cpuGflops), cpu_gflops, 6);

                                fprintf(f, "%d;%d;%d;%d;%d;%c;%c;%d;%zu;%s;%s;%s;%s;%s\n",
                                    pkg, N, R, BS, k, mode, actualMode, kernelIters, sharedBytesUsed,
                                    buf_avgKernel, buf_gflops, buf_cpuMs, buf_cpuGflops, ok ? "PASS" : "FAIL");
                                fclose(f);
                            }
                            else
                            {
                                printf("Warning: cannot open CSV '%s' for append\n", csv_name);
                            }

                            // Cleanup per-mode
                            cudaEventDestroy(start);
                            cudaEventDestroy(stop);
                            cudaStreamDestroy(stream);
                        } // modes loop
                    } // k
                } // BS

                // Free device and host buffers for this config
                cudaFree(d_A);
                cudaFree(d_C);
                cudaFreeHost(h_A);
                cudaFreeHost(h_C);
                cudaFreeHost(h_C_ref);
            } // R
        } // N
    } // packages

    printf("RunBatchTests finished. Results appended to %s\n", csv_name);
}

// ============================================================================
//  MAIN PROGRAM (updated to accept -batch argument)
// ============================================================================

int main(int argc, char **argv)
{
    printf("[Matrix Add Radius Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage:\n");
        printf("  -device=n           Select CUDA device\n");
        printf("  -N=MatrixSize       Matrix is N x N\n");
        printf("  -R=Radius           Window radius (R > 0, N > 2R)\n");
        printf("  -bs=BlockSize       Block size (8, 16, 32)\n");
        printf("  -k=OutputsPerThread Number of outputs per thread (1,2,8)\n");
        printf("  -mode=a|b|c|d       Kernel variant\n");
        printf("  -iters=n            Number of kernel iterations to average (default 10)\n");
        printf("  -csv=filename       Append results to CSV file (optional)\n");
        printf("  -batch=n            Run batch experiment with n distinct input packages (writes batch_results.csv)\n");
        printf("\n");
        printf("Example:\n");
        printf("  ./matrixAddRad -N=2048 -R=3 -bs=16 -k=2 -mode=c -iters=10 -csv=results.csv\n");
        printf("  ./matrixAddRad -batch=5   # runs batch experiment with 5 randomized input packages\n");
        return EXIT_SUCCESS;
    }

    // If user specified batch run, execute it and exit
    if (checkCmdLineFlag(argc, (const char **)argv, "batch"))
    {
        int packages = getCmdLineArgumentInt(argc, (const char **)argv, "batch");
        if (packages <= 0)
        {
            printf("Invalid -batch value. Must be > 0.\n");
            return EXIT_FAILURE;
        }
        // Initialize CUDA device (so device props etc. work)
        int dev = findCudaDevice(argc, (const char **)argv);
        (void)dev;

        RunBatchTests(packages);
        return EXIT_SUCCESS;
    }

    // Otherwise fall back to normal single-run flow (existing code path)
    // Select CUDA device
    int dev = findCudaDevice(argc, (const char **)argv);
    (void)dev;

    // Default parameters
    int N = 1024;
    int R = 1;
    int block_size = 32;
    int k = 1;
    char mode = 'a';
    int nIter = 10;
    const char *csv_filename = nullptr;
    static char csv_buf[1024] = {0};

    // Parse command-line arguments
    if (checkCmdLineFlag(argc, (const char **)argv, "N"))
        N = getCmdLineArgumentInt(argc, (const char **)argv, "N");

    if (checkCmdLineFlag(argc, (const char **)argv, "R"))
        R = getCmdLineArgumentInt(argc, (const char **)argv, "R");

    if (checkCmdLineFlag(argc, (const char **)argv, "bs"))
        block_size = getCmdLineArgumentInt(argc, (const char **)argv, "bs");

    if (checkCmdLineFlag(argc, (const char **)argv, "k"))
        k = getCmdLineArgumentInt(argc, (const char **)argv, "k");

    if (checkCmdLineFlag(argc, (const char **)argv, "iters"))
        nIter = getCmdLineArgumentInt(argc, (const char **)argv, "iters");

    if (checkCmdLineFlag(argc, (const char **)argv, "mode"))
    {
        char *m = NULL;
        getCmdLineArgumentString(argc, (const char **)argv, "mode", &m);
        if (m != NULL)
            mode = m[0];
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "csv"))
    {
        char *c = NULL;
        getCmdLineArgumentString(argc, (const char **)argv, "csv", &c);
        if (c != NULL)
        {
            // store in static buffer to ensure lifetime
            strncpy(csv_buf, c, sizeof(csv_buf) - 1);
            csv_filename = csv_buf;
        }
    }

    // Validate block size
    if (block_size != 8 && block_size != 16 && block_size != 32)
    {
        printf("WARNING: Unsupported block size %d. Using 32.\n", block_size);
        block_size = 32;
    }

    // Validate N > 2R
    if (N <= 2 * R)
    {
        printf("ERROR: N must be > 2R. N=%d R=%d\n", N, R);
        return EXIT_FAILURE;
    }

    printf("Running kernel %c with parameters:\n", mode);
    printf("  N=%d, R=%d, BLOCK_SIZE=%d, k=%d, iters=%d\n", N, R, block_size, k, nIter);
    if (csv_filename)
        printf("  CSV output: %s\n", csv_filename);

    checkCudaErrors(cudaProfilerStart());

    // Assume MatrixAddRadHost signature present (as in previous version)
    int result = MatrixAddRadHost(block_size, N, R, k, mode, nIter, csv_filename);

    checkCudaErrors(cudaProfilerStop());

    return result;
}

// ============================================================================
//  MAIN PROGRAM - OLD FOR TESTING
// ============================================================================

// int main(int argc, char **argv)
// {
//     printf("[Matrix Add Radius Using CUDA] - Starting...\n");

//     if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
//         checkCmdLineFlag(argc, (const char **)argv, "?"))
//     {
//         printf("Usage:\n");
//         printf("  -device=n           Select CUDA device\n");
//         printf("  -N=MatrixSize       Matrix is N x N\n");
//         printf("  -R=Radius           Window radius (R > 0, N > 2R)\n");
//         printf("  -bs=BlockSize       Block size (8, 16, 32)\n");
//         printf("  -k=OutputsPerThread Number of outputs per thread (1,2,8)\n");
//         printf("  -mode=a|b|c|d       Kernel variant\n");
//         printf("  -iters=n            Number of kernel iterations to average (default 10)\n");
//         printf("  -csv=filename       Append results to CSV file (optional)\n");
//         printf("\n");
//         printf("Example:\n");
//         printf("  ./matrixAddRad -N=2048 -R=3 -bs=16 -k=2 -mode=c -iters=10 -csv=results.csv\n");
//         return EXIT_SUCCESS;
//     }

//     // Select CUDA device
//     int dev = findCudaDevice(argc, (const char **)argv);
//     (void)dev;

//     // Default parameters
//     int N = 1024;
//     int R = 1;
//     int block_size = 32;
//     int k = 1;
//     char mode = 'a';
//     int nIter = 10;
//     const char *csv_filename = nullptr;
//     static char csv_buf[1024] = {0};

//     // Parse command-line arguments
//     if (checkCmdLineFlag(argc, (const char **)argv, "N"))
//         N = getCmdLineArgumentInt(argc, (const char **)argv, "N");

//     if (checkCmdLineFlag(argc, (const char **)argv, "R"))
//         R = getCmdLineArgumentInt(argc, (const char **)argv, "R");

//     if (checkCmdLineFlag(argc, (const char **)argv, "bs"))
//         block_size = getCmdLineArgumentInt(argc, (const char **)argv, "bs");

//     if (checkCmdLineFlag(argc, (const char **)argv, "k"))
//         k = getCmdLineArgumentInt(argc, (const char **)argv, "k");

//     if (checkCmdLineFlag(argc, (const char **)argv, "iters"))
//         nIter = getCmdLineArgumentInt(argc, (const char **)argv, "iters");

//     if (checkCmdLineFlag(argc, (const char **)argv, "mode"))
//     {
//         char *m = NULL;
//         getCmdLineArgumentString(argc, (const char **)argv, "mode", &m);
//         if (m != NULL)
//             mode = m[0];
//     }

//     if (checkCmdLineFlag(argc, (const char **)argv, "csv"))
//     {
//         char *c = NULL;
//         getCmdLineArgumentString(argc, (const char **)argv, "csv", &c);
//         if (c != NULL)
//         {
//             // store in static buffer to ensure lifetime
//             strncpy(csv_buf, c, sizeof(csv_buf) - 1);
//             csv_filename = csv_buf;
//         }
//     }

//     // Validate block size
//     if (block_size != 8 && block_size != 16 && block_size != 32)
//     {
//         printf("WARNING: Unsupported block size %d. Using 32.\n", block_size);
//         block_size = 32;
//     }

//     // Validate N > 2R
//     if (N <= 2 * R)
//     {
//         printf("ERROR: N must be > 2R. N=%d R=%d\n", N, R);
//         return EXIT_FAILURE;
//     }

//     printf("Running kernel %c with parameters:\n", mode);
//     printf("  N=%d, R=%d, BLOCK_SIZE=%d, k=%d, iters=%d\n", N, R, block_size, k, nIter);
//     if (csv_filename)
//         printf("  CSV output: %s\n", csv_filename);

//     checkCudaErrors(cudaProfilerStart());

//     int result = MatrixAddRadHost(block_size, N, R, k, mode, nIter, csv_filename);

//     checkCudaErrors(cudaProfilerStop());

//     return result;
// }

