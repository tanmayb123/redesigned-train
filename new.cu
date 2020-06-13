#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <sys/time.h>
#include <pthread.h>
#include "sha256.cuh"

#define TEXT "Tanmay Bakshi"
#define TEXT_LEN 13
#define BLOCK_SIZE 400000
#define GPUS 4
#define DIFFICULTY 4
#define RANDOM_LEN 20

__constant__ BYTE characterSet[63] = {"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"};

__global__ void sha256_cuda(BYTE *prefix, BYTE *randoms, BYTE *solves) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < BLOCK_SIZE) {
        SHA256_CTX ctx;
        BYTE digest[32];
        sha256_init(&ctx);
        sha256_update(&ctx, prefix, TEXT_LEN);
        sha256_update(&ctx, randoms + i * RANDOM_LEN, RANDOM_LEN);
        sha256_final(&ctx, digest);
        for (int j = 0; j < DIFFICULTY; j++) {
            if (digest[j] > 0) {
                solves[i] = 0;
                return;
            }
        }
/*        if ((digest[4] & 0xC0) > 0) {
            solves[i] = 0;
            return;
        }*/
        solves[i] = 1;
    }
}

__device__ void deviceRandomGen(unsigned long *x) {
    *x ^= (*x << 21);
    *x ^= (*x >> 35);
    *x ^= (*x << 4);
}

void hostRandomGen(unsigned long *x) {
    *x ^= (*x << 21);
    *x ^= (*x >> 35);
    *x ^= (*x << 4);
}

__global__ void generateStrings(BYTE *block, unsigned long baseSeed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < BLOCK_SIZE) {
        unsigned long seed = baseSeed;
        seed += (unsigned long) i;
        for (int j = 0; j < RANDOM_LEN; j++) {
            deviceRandomGen(&seed);
            int randomIdx = (int) (seed % 62);
            (block + i * RANDOM_LEN)[j] = characterSet[randomIdx];
        }
    }
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

    BYTE *rndBlock = (BYTE *) calloc(sizeof(BYTE), BLOCK_SIZE * RANDOM_LEN);
    BYTE *d_rndBlock;
    cudaMalloc(&d_rndBlock, sizeof(BYTE) * BLOCK_SIZE * RANDOM_LEN);

    BYTE *solves = (BYTE *) calloc(sizeof(BYTE), BLOCK_SIZE);
    BYTE *d_solves;
    cudaMalloc(&d_solves, sizeof(BYTE) * BLOCK_SIZE);

    unsigned long rngSeed = timems();
    hostRandomGen(&rngSeed);

    while (1) {
        generateStrings<<<BLOCK_SIZE / 256, 256>>>(d_rndBlock, rngSeed);
        hostRandomGen(&rngSeed);
        cudaDeviceSynchronize();

        hi->hashesProcessed += BLOCK_SIZE;
        sha256_cuda<<<BLOCK_SIZE / 256, 256>>>(d_prefix, d_rndBlock, d_solves);
        cudaDeviceSynchronize();

        cudaMemcpy(rndBlock, d_rndBlock, sizeof(BYTE) * BLOCK_SIZE * RANDOM_LEN, cudaMemcpyDeviceToHost);
        cudaMemcpy(solves, d_solves, sizeof(BYTE) * BLOCK_SIZE, cudaMemcpyDeviceToHost);

        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (solves[i] == 1) {
                pthread_mutex_lock(&provideSolutionLock);
                solution = rndBlock + i * RANDOM_LEN;
                pthread_mutex_unlock(&solutionLock);
                goto finish;
            }
        }
    }

finish:
    return NULL;
}

int main() {
    pthread_mutex_init(&provideSolutionLock, NULL);
    pthread_mutex_init(&solutionLock, NULL);

    pthread_mutex_lock(&solutionLock);

    pthread_t *tids = (pthread_t *) malloc(sizeof(pthread_t) * GPUS);
    unsigned long **processedPtrs = (unsigned long **) malloc(sizeof(unsigned long *) * GPUS);
    long long start = timems();
    for (int i = 0; i < GPUS; i++) {
        HandlerInput *hi = (HandlerInput *) malloc(sizeof(HandlerInput));
        hi->device = i;
        hi->hashesProcessed = 0;
        processedPtrs[i] = &hi->hashesProcessed;
        pthread_t tid;
        pthread_create(&tid, NULL, launchGPUHandlerThread, hi);
        tids[i] = tid;
        usleep(10);
    }

    pthread_mutex_lock(&solutionLock);
    long long end = timems();
    long long elapsed = end - start;

    unsigned long totalProcessed = 0;
    for (int i = 0; i < GPUS; i++) {
        totalProcessed += *(processedPtrs[i]);
        pthread_cancel(tids[i]);
    }
    printf("Solution: %.20s\n", solution);
    printf("Hashes processed: %lu\n", totalProcessed);
    printf("Time: %llu\n", elapsed);
    printf("Hashes/sec: %f\n", ((float) totalProcessed / (float) elapsed) * 1000);

    return 0;
}
