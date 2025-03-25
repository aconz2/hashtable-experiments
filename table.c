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
typedef uint16_t u16;
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

u8 ymm_entry_cmp(__m256i v, u32 key) {
    return _mm256_movemask_ps(_mm256_cmpeq_epi32(v, _mm256_set1_epi32(key)));
}

u16 ymm2_entry_cmp(__m256i v[2], u32 key) {
    u8 ma = ymm_entry_cmp(v[0], key);
    u8 mb = ymm_entry_cmp(v[1], key);
    u16 m = (((u16)mb) << 8) | (u16)ma;
    return m;
}

// ----------------------------------- Table1 -------------------------------

typedef struct {
    union {
        __m256i kv; // (k)ey (v)ector
        u32 key[8];
    };
    u64 value[8];
} Table1Bucket; // 1.5 cache lines

size_t Table1_n_buckets(size_t n_bits) { return ((u64)1) << n_bits; }
size_t Table1_n_entries(size_t n_bits) { return 8 * Table1_n_buckets(n_bits); }

void Table1Bucket_dump(Table1Bucket* e) {
    printf("k: ");
    for (size_t i = 0; i < 8; i++) { printf("%d ", e->key[i]); }
    printf("\nv: ");
    for (size_t i = 0; i < 8; i++) { printf("%ld ", e->value[i]); }
    printf("\n");
}

void Table1_dump(Table1Bucket* e, size_t n_bits) {
    for (size_t i = 0; i < Table1_n_buckets(n_bits); i++) {
        Table1Bucket_dump(&e[i]);
    }
}

Table1Bucket* Table1_create(size_t n_bits) {
    Table1Bucket* ret = calloc(Table1_n_buckets(n_bits), sizeof(Table1Bucket));
    assert(ret != NULL);
    return ret;
}

void Table1_reset(Table1Bucket* table, size_t n_bits) {
    for (size_t i = 0; i < Table1_n_buckets(n_bits); i++) {
        table[i].kv = _mm256_setzero_si256();
    }
}

int Table1_insert(Table1Bucket* table, size_t n_bits, u64 H, u32 key, u64 value) {
    size_t bucket = hash1(key, n_bits, H);
    u8 mask = ymm_entry_cmp(table[bucket].kv, key);
    if (mask != 0) { return 1; } // key existed
    mask = ymm_entry_cmp(table[bucket].kv, 0);
    if (mask == 0) { return 2; } // bucket full
    u8 idx = __builtin_ctz(mask);
    table[bucket].key[idx] = key;
    assert(__builtin_ctz(ymm_entry_cmp(table[bucket].kv, key)) == idx);
    table[bucket].value[idx] = value;
    return 0;
}

int Table1_get(Table1Bucket* table, size_t n_bits, u64 H, u32 key, u64* value) {
    size_t bucket = hash1(key, n_bits, H);
    u8 mask = ymm_entry_cmp(table[bucket].kv, key);
    if (mask == 0) { return 1; } // key not found
    u8 idx = __builtin_ctz(mask);
    assert(table[bucket].key[idx] == key);
    *value = table[bucket].value[idx];
    return 0;
}

size_t Table1_try_build(Table1Bucket* table, size_t n_bits, u32* keys, size_t target_size, size_t tries, u64* H) {
    PRNG32RomuQuad rng;
    size_t try = 0;
    while (try <= tries) {
        try += 1;
        rng_init(&rng, 42 + try);
        u64 Htry = rng_u64(&rng) * 2 + 1;
        Table1_reset(table, n_bits);

        /*printf("trying H=%ld\n", Htry);*/

        size_t occupancy = 0;
        while (occupancy < target_size) {
            u32 key = rng_u32(&rng);
            if (key == 0) continue;
            u64 value = key;
            int ret = Table1_insert(table, n_bits, Htry, key, value);
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
        *H = Htry;
        return try;
nexttry:
        (void)0;
    }
    return 0;
}

size_t Table1_bsearch_occupancy(Table1Bucket* table, size_t n_bits, u32* keys, size_t tries, u64* H) {
    size_t size = ((u64)1) << n_bits;
    size_t size_entries = size * 8;
    size_t lo = 1;
    size_t hi = size_entries - 1;
    while (hi - lo > 1) {
        size_t target_size = lo + (hi-lo)/2;
        printf("lo=%ld hi=%ld trying with target_size %ld\n", lo, hi, target_size);
        size_t try = Table1_try_build(table, n_bits, keys, target_size, tries, H);
        if (try == 0) { // didn't succeed
            hi = target_size;
        } else {
            lo = target_size;
        }
    }
    size_t try = Table1_try_build(table, n_bits, keys, lo, tries, H);
    assert(try != 0);
    return lo;
}

// ----------------------------------- Table2 -------------------------------

typedef struct {
    union {
        __m256i kv[2]; // (k)ey (v)ector
        u32 key[16];
    };
    u64 value[16];
} Table2Bucket; // 3 cache lines

size_t Table2_n_buckets(size_t n_bits) { return ((u64)1) << n_bits; }
size_t Table2_n_entries(size_t n_bits) { return 16 * Table2_n_buckets(n_bits); }

void Table2Bucket_dump(Table2Bucket* e) {
    printf("k: ");
    for (size_t i = 0; i < 16; i++) { printf("%d ", e->key[i]); }
    printf("\nv: ");
    for (size_t i = 0; i < 16; i++) { printf("%ld ", e->value[i]); }
    printf("\n");
}

void Table2_dump(Table2Bucket* e, size_t n_bits) {
    for (size_t i = 0; i < Table2_n_buckets(n_bits); i++) {
        Table2Bucket_dump(&e[i]);
    }
}

Table2Bucket* Table2_create(size_t n_bits) {
    Table2Bucket* ret = calloc(Table2_n_buckets(n_bits), sizeof(Table2Bucket));
    assert(ret != NULL);
    return ret;
}

void Table2_reset(Table2Bucket* table, size_t n_bits) {
    for (size_t i = 0; i < Table2_n_buckets(n_bits); i++) {
        table[i].kv[0] = _mm256_setzero_si256();
        table[i].kv[1] = _mm256_setzero_si256();
    }
}

int Table2_insert(Table2Bucket* table, size_t n_bits, u64 H, u32 key, u64 value) {
    size_t bucket = hash1(key, n_bits, H);
    u8 ma = ymm_entry_cmp(table[bucket].kv[0], key);
    u8 mb = ymm_entry_cmp(table[bucket].kv[1], key);
    if (ma != 0 || mb != 0) { return 1; } // key existed
    ma = ymm_entry_cmp(table[bucket].kv[0], 0);
    mb = ymm_entry_cmp(table[bucket].kv[1], 0);
    if (ma == 0 && mb == 0) { return 2; } // bucket full
    u16 m = (((u16)mb) << 8) | (u16)ma;
    u8 idx = __builtin_ctz(m);
    table[bucket].key[idx] = key;
    assert(__builtin_ctz(ymm2_entry_cmp(table[bucket].kv, key)) == idx);
    table[bucket].value[idx] = value;
    return 0;
}

int Table2_get(Table2Bucket* table, size_t n_bits, u64 H, u32 key, u64* value) {
    size_t bucket = hash1(key, n_bits, H);
    u16 mask = ymm2_entry_cmp(table[bucket].kv, key);
    if (mask == 0) { return 1; } // key not found
    u8 idx = __builtin_ctz(mask);
    assert(table[bucket].key[idx] == key);
    *value = table[bucket].value[idx];
    return 0;
}

size_t Table2_try_build(Table2Bucket* table, size_t n_bits, u32* keys, size_t target_size, size_t tries, u64* H) {
    PRNG32RomuQuad rng;
    size_t try = 0;
    while (try <= tries) {
        try += 1;
        rng_init(&rng, 42 + try);
        u64 Htry = rng_u64(&rng) * 2 + 1;
        Table2_reset(table, n_bits);

        /*printf("trying H=%ld\n", Htry);*/

        size_t occupancy = 0;
        while (occupancy < target_size) {
            u32 key = rng_u32(&rng);
            if (key == 0) continue;
            u64 value = key;
            int ret = Table2_insert(table, n_bits, Htry, key, value);
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
        *H = Htry;
        return try;
nexttry:
        (void)0;
    }
    return 0;
}

size_t Table2_bsearch_occupancy(Table2Bucket* table, size_t n_bits, u32* keys, size_t tries, u64* H) {
    size_t size = ((u64)1) << n_bits;
    size_t size_entries = size * 8;
    size_t lo = 1;
    size_t hi = size_entries - 1;
    while (hi - lo > 1) {
        size_t target_size = lo + (hi-lo)/2;
        printf("lo=%ld hi=%ld trying with target_size %ld\n", lo, hi, target_size);
        size_t try = Table2_try_build(table, n_bits, keys, target_size, tries, H);
        if (try == 0) { // didn't succeed
            hi = target_size;
        } else {
            lo = target_size;
        }
    }
    size_t try = Table2_try_build(table, n_bits, keys, lo, tries, H);
    assert(try != 0);
    return lo;
}

// ------------------------------------------------------------------------------------------------------

int main() {

    size_t tries = 1000;

    {
        size_t n_bits = 8;
        size_t n_buckets = Table1_n_buckets(n_bits);
        size_t n_entries = Table1_n_entries(n_bits);
        Table1Bucket* table = Table1_create(n_bits);
        u32* keys = calloc(n_entries, sizeof(u32));
        u64 H = 0;
        /*size_t try = try_table_build(table, n_bits, keys, target_size, tries, &H);*/
        size_t got_entries = Table1_bsearch_occupancy(table, n_bits, keys, tries, &H);
        printf("table1 got %ld num_entries\n", got_entries);
        printf("table1 %ld buckets, %ld entries\n", n_buckets, n_entries);
        printf("%.2f occupancy\n", (double)got_entries/(double)n_entries);

#ifndef NDEBUG
        for (size_t i = 0; i < got_entries; i++) {
            u64 value;
            int ret = Table1_get(table, n_bits, H, keys[i], &value);
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
            for (size_t i = 0; i < got_entries; i++) {
                u64 value;
                int ret = Table1_get(table, n_bits, H, keys[i], &value);
                present |= ret;
                check += value;
            }
        }
        clock_ns(&stop);

        printf("%.2f ns/lookup %.2f ms present=%ld check=%lx\n", (double)elapsed_ns(start, stop) / (double)rounds / (double)got_entries, (double)elapsed_ns(start, stop) / 1000000, present, check);
        free(keys);
        free(table);
    }

    {
        size_t n_bits = 7;
        size_t n_buckets = Table2_n_buckets(n_bits);
        size_t n_entries = Table2_n_entries(n_bits);
        Table2Bucket* table = Table2_create(n_bits);
        u32* keys = calloc(n_entries, sizeof(u32));
        u64 H = 0;
        size_t got_entries = Table2_bsearch_occupancy(table, n_bits, keys, tries, &H);
        printf("table1 got %ld num_entries\n", got_entries);
        printf("table1 %ld buckets, %ld entries\n", n_buckets, n_entries);
        printf("%.2f occupancy\n", (double)got_entries/(double)n_entries);

#ifndef NDEBUG
        for (size_t i = 0; i < got_entries; i++) {
            u64 value;
            int ret = Table2_get(table, n_bits, H, keys[i], &value);
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
            for (size_t i = 0; i < got_entries; i++) {
                u64 value;
                int ret = Table2_get(table, n_bits, H, keys[i], &value);
                present |= ret;
                check += value;
            }
        }
        clock_ns(&stop);

        printf("%.2f ns/lookup %.2f ms present=%ld check=%lx\n", (double)elapsed_ns(start, stop) / (double)rounds / (double)got_entries, (double)elapsed_ns(start, stop) / 1000000, present, check);
        free(keys);
        free(table);
    }


}
