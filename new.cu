#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <sys/time.h>
#include <pthread.h>
#include <locale.h>
#include "sha256.cuh"

#define TEXT "Tanmay Bakshi"
#define TEXT_LEN 13
#define BLOCK_SIZE 400000
#define GPUS 4
#define DIFFICULTY 3
#define RANDOM_LEN 20

__constant__ BYTE characterSet[63] = {"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"};

__global__ void initSolutionMemory(int *blockContainsSolution) {
    *blockContainsSolution = -1;
}

__device__ void deviceRandomGen(unsigned long *x) {
    *x ^= (*x << 21);
    *x ^= (*x >> 35);
    *x ^= (*x << 4);
}

__global__ void sha256_cuda(BYTE *prefix, BYTE *solution, int *blockContainsSolution, unsigned long baseSeed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < BLOCK_SIZE) {
        SHA256_CTX ctx;
        BYTE digest[32];
        BYTE random[20];
        unsigned long seed = baseSeed;
        seed += (unsigned long) i;
        for (int j = 0; j < RANDOM_LEN; j++) {
            deviceRandomGen(&seed);
            int randomIdx = (int) (seed % 62);
            random[j] = characterSet[randomIdx];
        }
        sha256_init(&ctx);
        sha256_update(&ctx, prefix, TEXT_LEN);
        sha256_update(&ctx, random, RANDOM_LEN);
        sha256_final(&ctx, digest);
        for (int j = 0; j < DIFFICULTY; j++)
            if (digest[j] > 0)
                return;
//        if ((digest[DIFFICULTY] & 0xF0) > 0)
//            return;
        if (*blockContainsSolution == 1)
            return;
        *blockContainsSolution = 1;
        for (int j = 0; j < RANDOM_LEN; j++)
            solution[j] = random[j];
    }
}

void hostRandomGen(unsigned long *x) {
    *x ^= (*x << 21);
    *x ^= (*x >> 35);
    *x ^= (*x << 4);
}

void pre_sha256() {
    cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice);
}

long long timems() {
    struct timeval end;
    gettimeofday(&end, NULL);
    return end.tv_sec * 1000LL + end.tv_usec / 1000;
}

struct HandlerInput {
    int device;
    unsigned long hashesProcessed;
};
typedef struct HandlerInput HandlerInput;

pthread_mutex_t provideSolutionLock;
pthread_mutex_t solutionLock;
BYTE *solution;

void *launchGPUHandlerThread(void *vargp) {
    HandlerInput *hi = (HandlerInput *) vargp;
    cudaSetDevice(hi->device);

    pre_sha256();

    BYTE cpuPrefix[] = {TEXT};
    BYTE *d_prefix;
    cudaMalloc(&d_prefix, TEXT_LEN);
    cudaMemcpy(d_prefix, cpuPrefix, TEXT_LEN, cudaMemcpyHostToDevice);

    BYTE *blockSolution = (BYTE *) malloc(sizeof(BYTE) * RANDOM_LEN);
    BYTE *d_solution;
    cudaMalloc(&d_solution, sizeof(BYTE) * RANDOM_LEN);

    int *blockContainsSolution = (int *) malloc(sizeof(int));
    int *d_blockContainsSolution;
    cudaMalloc(&d_blockContainsSolution, sizeof(int));

    unsigned long rngSeed = timems();
    hostRandomGen(&rngSeed);

    initSolutionMemory<<<1, 1>>>(d_blockContainsSolution);

    while (1) {
        hostRandomGen(&rngSeed);

        hi->hashesProcessed += BLOCK_SIZE;
        pthread_mutex_lock(&provideSolutionLock);
        sha256_cuda<<<BLOCK_SIZE / 256, 256>>>(d_prefix, d_solution, d_blockContainsSolution, rngSeed);
        pthread_mutex_unlock(&provideSolutionLock);
        cudaDeviceSynchronize();

        cudaMemcpy(blockContainsSolution, d_blockContainsSolution, sizeof(int), cudaMemcpyDeviceToHost);
        if (*blockContainsSolution == 1) {
            cudaMemcpy(blockSolution, d_solution, sizeof(BYTE) * RANDOM_LEN, cudaMemcpyDeviceToHost);
            pthread_mutex_lock(&provideSolutionLock);
            solution = blockSolution;
            pthread_mutex_unlock(&solutionLock);
            goto finish;
        }
    }

finish:
    return NULL;
}

int main() {
    pthread_mutex_init(&provideSolutionLock, NULL);
    pthread_mutex_init(&solutionLock, NULL);

    pthread_mutex_lock(&solutionLock);

    unsigned long **processedPtrs = (unsigned long **) malloc(sizeof(unsigned long *) * GPUS);
    long long start = timems();
    for (int i = 0; i < GPUS; i++) {
        HandlerInput *hi = (HandlerInput *) malloc(sizeof(HandlerInput));
        hi->device = i;
        hi->hashesProcessed = 0;
        processedPtrs[i] = &hi->hashesProcessed;
        pthread_t tid;
        pthread_create(&tid, NULL, launchGPUHandlerThread, hi);
        usleep(10);
    }

    pthread_mutex_lock(&solutionLock);
    long long end = timems();
    long long elapsed = end - start;

    unsigned long totalProcessed = 0;
    for (int i = 0; i < GPUS; i++) {
        totalProcessed += *(processedPtrs[i]);
    }
    setlocale(LC_NUMERIC, "");
    printf("Solution: %.20s\n", solution);
    printf("Hashes processed: %'lu\n", totalProcessed);
    printf("Time: %llu\n", elapsed);
    printf("Hashes/sec: %'lu\n", (unsigned long) ((double) totalProcessed / (double) elapsed) * 1000);

    return 0;
}
