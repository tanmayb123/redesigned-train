#include <pthread.h>

struct LinkedListNode {
    int data;
    char fresh;
    struct LinkedListNode *next;
};

typedef struct LinkedListNode LinkedListNode;

LinkedListNode *createLinkedList(int length) {
    LinkedListNode *startNode = (LinkedListNode *) malloc(sizeof(LinkedListNode));
    startNode->fresh = 0;
    LinkedListNode *lastNode = startNode;
    for (int i = 1; i < length; i++) {
        LinkedListNode *nextNode = (LinkedListNode *) malloc(sizeof(LinkedListNode));
        nextNode->fresh = 0;
        lastNode->next = nextNode;
        lastNode = nextNode;
    }
    lastNode->next = startNode;
    return lastNode;
}

void insertDataInLinkedList(LinkedListNode *start, int length, int *data) {
    LinkedListNode *node = start;
    for (int i = 0; i < length; i++) {
        node->data = data[i];
        node = node->next;
    }
}

typedef struct {
    LinkedListNode *data;
    pthread_mutex_t *writeMutex;
    pthread_mutex_t *readMutex;
} Channel;

void writeToChannel(Channel *channel, int data) {
    pthread_mutex_lock(channel->writeMutex);
    LinkedListNode *node = channel->data;
    while (1) {
        if (node->fresh == 0) {
            node->data = data;
            node->fresh = 1;
            channel->data = node;
            break;
        }
        node = node->next;
    }
    pthread_mutex_unlock(channel->writeMutex);
}

int readFromChannel(Channel *channel) {
    pthread_mutex_lock(channel->readMutex);
    LinkedListNode *node = channel->data;
    int value;
    while (1) {
        if (node->fresh == 1) {
            value = node->data;
            node->fresh = 0;
            break;
        }
        node = node->next;
    }
    pthread_mutex_unlock(channel->readMutex);
    return value;
}

Channel *createChannel(int bufferSize) {
    Channel *chan = (Channel *) malloc(sizeof(Channel));
    chan->data = createLinkedList(bufferSize);
    chan->writeMutex = (pthread_mutex_t *) malloc(sizeof(pthread_mutex_t));
    chan->readMutex = (pthread_mutex_t *) malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(chan->writeMutex, NULL);
    pthread_mutex_init(chan->readMutex, NULL);
    return chan;
}
