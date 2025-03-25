// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <immintrin.h>
#include <time.h>

#include "random.h"

#define BILLION  1000000000LL

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t u8;
typedef struct timespec Timespec;

static void clock_ns(Timespec* t) {
  clock_gettime(CLOCK_MONOTONIC, t);
}

static u64 elapsed_ns(Timespec start, Timespec stop) {
  return (u64)(stop.tv_sec - start.tv_sec) * BILLION + (u64)(stop.tv_nsec - start.tv_nsec);
}

// https://arxiv.org/pdf/1504.06804
// hashes x universally into l<=64 bits using random odd seed a.
static u64 hash1(u64 x, u64 l, u64 a) {
    return (a*x) >> (64-l);
}

void rng_init(PRNG32RomuQuad* rng, u8 seed) {
    if (seed == 0) { seed += 1; }
    u32 x = seed;
    x = x | (x << 8) | (x << 16) | (x << 24);
    for (int i = 0; i < 4; i++) {
        rng->s[i] = x;
    }
    for (int i = 0; i < 10; i++) {
        (void)prng32_romu_quad(rng);
    }
}

u32 rng_u32(PRNG32RomuQuad* rng) { return prng32_romu_quad(rng); }
u32 rng_u64(PRNG32RomuQuad* rng) {
    u64 a = rng_u32(rng);
    u64 b = rng_u32(rng);
    return (a << 32) | b;
}

typedef struct {
    union {
        __m256i kv; // (k)ey (v)ector
        u32 key[8];
    };
    u64 value[8];
} YmmEntry; // 1.5 cache lines

void dump_ymm_entry(YmmEntry* e) {
    printf("k: ");
    for (size_t i = 0; i < 8; i++) { printf("%d ", e->key[i]); }
    printf("\n");
    printf("v: ");
    for (size_t i = 0; i < 8; i++) { printf("%ld ", e->value[i]); }
    printf("\n");
}

void dump_ymm_table(YmmEntry* e, size_t n_bits) {
    size_t size = ((u64)1) << n_bits;
    for (size_t i = 0; i < size; i++) {
        dump_ymm_entry(&e[i]);
    }
}

u8 ymm_entry_cmp(YmmEntry* ye, u32 key) {
    return _mm256_movemask_ps(_mm256_cmpeq_epi32(ye->kv, _mm256_set1_epi32(key)));
}

// will have 8 * 2^n_bits total entries
YmmEntry* ymm_table_create(size_t n_bits) {
    size_t size = ((u64)1) << n_bits;
    YmmEntry* ret = calloc(size, sizeof(YmmEntry));
    assert(ret != NULL);
    return ret;
}

void ymm_table_reset(YmmEntry* table, size_t n_bits) {
    size_t size = ((u64)1) << n_bits;
    for (size_t i = 0; i < size; i++) {
        table[i].kv = _mm256_setzero_si256();
    }
}

int ymm_table_insert(YmmEntry* table, size_t n_bits, u64 H, u32 key, u64 value) {
    size_t bucket = hash1(key, n_bits, H);
    u8 mask = ymm_entry_cmp(&table[bucket], key);
    if (mask != 0) { return 1; } // key existed
    mask = ymm_entry_cmp(&table[bucket], 0);
    if (mask == 0) { return 2; } // bucket full
    u8 idx = __builtin_ctz(mask);
    /*printf("idx is %d\n", idx);*/
    table[bucket].key[idx] = key;
    assert(__builtin_ctz(ymm_entry_cmp(&table[bucket], key)) == idx);
    table[bucket].value[idx] = value;
    return 0;
}

int ymm_table_get(YmmEntry* table, size_t n_bits, u64 H, u32 key, u64* value) {
    size_t bucket = hash1(key, n_bits, H);
    u8 mask = ymm_entry_cmp(&table[bucket], key);
    if (mask == 0) { return 1; } // key not found
    u8 idx = __builtin_ctz(mask);
    assert(table[bucket].key[idx] == key);
    *value = table[bucket].value[idx];
    return 0;
}

int main() {
    PRNG32RomuQuad rng;

    size_t n_bits = 8;
    size_t size = ((u64)1) << n_bits;
    size_t size_entries = size * 8;
    size_t target_size = 925;

    YmmEntry* table = ymm_table_create(n_bits);
    u32* keys = calloc(target_size, sizeof(u32));

    int success = 0;
    size_t try = 0;
    u64 H = 0;

    while (try <= 10000) {
        try += 1;
        rng_init(&rng, 42 + try);
        u64 Htry = rng_u64(&rng) * 2 + 1;
        ymm_table_reset(table, n_bits);

        /*printf("trying H=%ld\n", Htry);*/

        size_t occupancy = 0;
        while (occupancy < target_size) {
            u32 key = rng_u32(&rng);
            if (key == 0) continue;
            u64 value = key;
            int ret = ymm_table_insert(table, n_bits, Htry, key, value);
            if (ret == 1) {
                // key exists
                continue;
            } else if (ret == 2) {
                // bucket full
                /*printf("bucket full at occupancy %ld\n", occupancy);*/
                goto nexttry;
            }
            /*printf("inserted %x\n", key);*/
            assert(ret == 0);
            keys[occupancy] = key;
            occupancy += 1;
        }
        success = 1;
        H = Htry;
        break;
nexttry:
        (void)0;
    }

    if (success == 0) {
        free(table);
        free(keys);
        printf("fail\n");
        return 1;
    }

    /*dump_ymm_table(table, n_bits);*/

    printf("tries = %ld success = %d\n", try, success);
    printf("%ld buckets, %ld entries\n", size, size_entries);
    printf("%.2f occupancy\n", (double)target_size/(double)size_entries);

#ifndef NDEBUG
    for (size_t i = 0; i < target_size; i++) {
        u64 value;
        int ret = ymm_table_get(table, n_bits, H, keys[i], &value);
        assert(ret == 0);
        assert((u64)keys[i] == value);
    }
#endif

    u64 check = 0;
    u64 present = 0;
    Timespec start, stop;
    size_t rounds = 10000;

    clock_ns(&start);
    for (size_t round = 0; round < rounds; round++) {
        for (size_t i = 0; i < target_size; i++) {
            u64 value;
            int ret = ymm_table_get(table, n_bits, H, keys[i], &value);
            present |= ret;
            check += value;
        }
    }
    clock_ns(&stop);

    printf("%.2f ns/lookup %.2f ms present=%ld check=%lx\n", (double)elapsed_ns(start, stop) / (double)rounds / (double)target_size, (double)elapsed_ns(start, stop) / 1000000, present, check);


    free(keys);
    free(table);
}
