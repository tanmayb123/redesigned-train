#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include "sha256.cuh"
#include <sys/time.h>
#include "channels.c"

#define TEXT "Tanmay Bakshi"
#define TEXT_LEN 13
#define BLOCKS 5000
#define BLOCK_SIZE 50000
#define GPUS 4
#define THREADS 384 / GPUS
#define DIFFICULTY 3
#define RANDOM_LEN 20

__constant__ BYTE dev_characterSet[63];
BYTE characterSet[63] = {"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"};

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
        if ((digest[3] & 0xF0) > 0) {
            solves[i] = 0;
            return;
        }
        solves[i] = 1;
    }
}

void pre_sha256() {
    cudaMemcpyToSymbol(dev_characterSet, characterSet, sizeof(characterSet), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice);
}

void runHashCheck(BYTE *prefix, BYTE *randoms, BYTE *solves) {
    int blockSize = 4;
    int numBlocks = (BLOCK_SIZE + blockSize - 1) / blockSize;
    sha256_cuda<<<numBlocks, blockSize>>>(prefix, randoms, solves);
}

long long timems() {
    struct timeval end;
    gettimeofday(&end, NULL);
    return end.tv_sec * 1000LL + end.tv_usec / 1000;
}

unsigned long generateRandomNumber(unsigned long x) {
    x ^= (x << 21);
    x ^= (x >> 35);
    x ^= (x << 4);
    return x;
}

void insertRandomString(BYTE *buffer, int randomLength, unsigned long *rngSeed) {
    for (int i = 0; i < randomLength; i++) {
        *rngSeed = generateRandomNumber(*rngSeed);
        buffer[i] = characterSet[(int) (*rngSeed % 62)];
    }
}

struct RandomWorkerInput {
    unsigned long rng;
    BYTE **blocks;
    Channel *blockCreationChannel;
    Channel *blockUsageChannel;
};
typedef struct RandomWorkerInput RandomWorkerInput;

void *randomWorker(void *vargp) {
    RandomWorkerInput *rwi = (RandomWorkerInput *) vargp;
    while (1) {
        int freshBlockIdx = readFromChannel(rwi->blockUsageChannel);
        insertRandomString(rwi->blocks[freshBlockIdx], BLOCK_SIZE * RANDOM_LEN, &rwi->rng);
        writeToChannel(rwi->blockCreationChannel, freshBlockIdx);
    }
    return NULL;
}

struct HandlerInput {
    int device;
    unsigned long rng;
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
    BYTE *cudaPrefix;
    cudaMalloc(&cudaPrefix, TEXT_LEN);
    cudaMemcpy(cudaPrefix, cpuPrefix, TEXT_LEN, cudaMemcpyHostToDevice);

    BYTE **blocks = (BYTE **) malloc(sizeof(BYTE *) * BLOCKS);
    for (int i = 0; i < BLOCKS; i++) {
        blocks[i] = (BYTE *) malloc(sizeof(BYTE) * BLOCK_SIZE * RANDOM_LEN);
    }

    Channel *blockCreationChannel = createChannel(BLOCKS);
    Channel *blockUsageChannel = createChannel(BLOCKS);

    for (int i = 0; i < THREADS; i++) {
        RandomWorkerInput *input = (RandomWorkerInput *) malloc(sizeof(RandomWorkerInput));
        hi->rng = generateRandomNumber(hi->rng);
        input->rng = hi->rng;
        input->blocks = blocks;
        input->blockCreationChannel = blockCreationChannel;
        input->blockUsageChannel = blockUsageChannel;
        pthread_t tid;
        pthread_create(&tid, NULL, randomWorker, (void *) input);
    }

    for (int i = 0; i < BLOCKS; i++) {
        writeToChannel(blockUsageChannel, i);
    }

    BYTE *solves;
    cudaMallocManaged(&solves, BLOCK_SIZE);

    BYTE *deviceBlock;
    cudaMallocManaged(&deviceBlock, BLOCK_SIZE * RANDOM_LEN);

    while (1) {
        hi->hashesProcessed += BLOCK_SIZE;
        int freshBlockIdx = readFromChannel(blockCreationChannel);
        BYTE *block = blocks[freshBlockIdx];
        cudaMemcpy(deviceBlock, block, BLOCK_SIZE * RANDOM_LEN, cudaMemcpyHostToDevice);
        runHashCheck(cudaPrefix, deviceBlock, solves);
        cudaDeviceSynchronize();
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (solves[i] == 1) {
                pthread_mutex_lock(&provideSolutionLock);
                solution = block + i * RANDOM_LEN;
                pthread_mutex_unlock(&solutionLock);
                goto finish;
            }
        }
        writeToChannel(blockUsageChannel, freshBlockIdx);
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
        hi->rng = time(NULL);
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
