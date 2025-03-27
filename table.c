// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <immintrin.h>
#include <time.h>

#include "random.h"

#define BILLION  1000000000LL

/*#define SEQKEYS*/

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
u64 hash1(u64 x, u64 l, u64 H[2]) {
    u64 a = H[0];
    u64 b = H[1];
    /*u64 a = 0x9e3779b97f4a7c15;*/
    /*return (a*x) >> (64-l);*/
    /*return (a*x+b) >> (64-l);*/
    /*return ((x + b) * a) >> (64-l);*/
    /*return (x * a) >> (64-l);*/
    return _bzhi_u64(a * x, l);
    /*return (a*x) & ((1 << l) - 1);*/
    /*(void)a;*/
    // multiplying by p turns into a shl and sub
    /*u64 p = 0xffffffffffff;*/
    // ANDing seems a bit faster with a bit more collisions (uses bzhi bit zero hi) (uses low bits)
    /*return ((p*x) ^ a) & ((1 << l) - 1);*/
    /*return (p*x) & ((1 << l) - 1);*/
    /*return (a*x) >> (64-l);*/
}

/*static u64 hash1(u32 x, u64 l, u64 a) {*/
    /*return _mm_crc32_u32((u32)a, x) >> (32-l);*/
    /*return _mm_crc32_u32((u32)a, x) & ((1 << l) - 1);*/
/*}*/

// murmur
/*u64 hash1(u64 x, u64 l, u64 a) {*/
/*  x ^= x >> 33;*/
/*  x *= 0xff51afd7ed558ccdL;*/
/*  x ^= x >> 33;*/
/*  x *= 0xc4ceb9fe1a85ec53L;*/
/*  x ^= x >> 33;*/
/*  return x >> (64 - l);*/
/*}*/

// https://github.com/skeeto/hash-prospector
/*u32 hash1(u32 x, u64 l, u64 H[2]) {*/
/*    x ^= x >> 17;*/
/*    x *= 0xed5ad4bb;*/
/*    x ^= x >> 11;*/
/*    x *= 0xac4c1b51;*/
/*    x ^= x >> 15;*/
/*    x *= 0x31848bab;*/
/*    x ^= x >> 14;*/
/*    return x >> (32 - l);*/
/*}*/
/*u32 hash1(u32 x, u64 l, u64 H[2]) {*/
/*    x ^= x >> 16;*/
/*    x *= 0x7feb352d;*/
/*    x ^= x >> 15;*/
/*    x *= 0x846ca68b;*/
/*    x ^= x >> 16;*/
/*    return x >> (32-l);*/
/*}*/

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
u64 rng_u64(PRNG32RomuQuad* rng) {
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

// by having 64 bit values, we annoyingly take up 32 + 64 bytes for each bucket
// this makes the index calculation one longer
//
// 32 bit values
//  40161c: c4 e2 cb f7 c2               	shrx	rax, rdx, rsi
//  401621: 48 c1 e0 06                  	shl	rax, 0x6
//  401625: c5 fd 76 04 07               	vpcmpeqd	ymm0, ymm0, ymmword ptr [rdi + rax]
//
// 64 bit values
//  40162c: c4 e2 cb f7 c2               	shrx	rax, rdx, rsi
//  401631: 48 8d 04 40                  	lea	rax, [rax + 2*rax]
//  401635: 48 c1 e0 05                  	shl	rax, 0x5
//  401639: c5 fd 76 04 07               	vpcmpeqd	ymm0, ymm0, ymmword ptr [rdi + rax]
//
//  we also get about 1.49ns/lookup vs 1.14 ns/lookup with 64 vs 32 bit value
//  1.30 vs 0.96 for batch 4
typedef struct {
    union {
        __m256i kv; // (k)ey (v)ector
        u32 key[8];
    };
    u32 value[8];
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
    Table1Bucket* ret = aligned_alloc(32, Table1_n_buckets(n_bits) * sizeof(Table1Bucket));
    assert(ret != NULL);
    return ret;
}

void Table1_reset(Table1Bucket* table, size_t n_bits) {
    for (size_t i = 0; i < Table1_n_buckets(n_bits); i++) {
        table[i].kv = _mm256_setzero_si256();
    }
}

int Table1_insert(Table1Bucket* table, size_t n_bits, u64 H[2], u32 key, u64 value) {
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

int Table1_get(Table1Bucket* table, size_t n_bits, u64 H[2], u32 key, u64* value) {
    size_t bucket = hash1(key, n_bits, H);
    u8 mask = ymm_entry_cmp(table[bucket].kv, key);
    if (mask == 0) { return 1; } // key not found
    u8 idx = __builtin_ctz(mask);
    assert(table[bucket].key[idx] == key);
    *value = table[bucket].value[idx];
    return 0;
}

int Table1_get_batch2(Table1Bucket* table, size_t n_bits, u64 H[2], u32 key[2], u64 value[2]) {
/*#define IMPL1*/
#ifdef IMPL1
    size_t bucket[2];
    u8 mask[2];
    for (size_t i = 0; i < 2; i++) { bucket[i] = hash1(key[i], n_bits, H); }
    for (size_t i = 0; i < 2; i++) { mask[i] = ymm_entry_cmp(table[bucket[i]].kv, key[i]); }
    int ret = 0;
    for (size_t i = 0; i < 2; i++) {
        if (mask[i] == 0) {
            ret |= (1 << i);
        } else {
            value[i] = table[bucket[i]].value[__builtin_ctz(mask[i])];
        }
    }
    return ret;

#else
    // this does worse! I think because we increase the latency to loading the first bucket
    // since we can do work for the second one while the load for the first happens
    u8 mask[2];
    Table1Bucket* bucket0 = table + hash1(key[0], n_bits, H);
    Table1Bucket* bucket1 = table + hash1(key[1], n_bits, H);
    __m256 kv0 = _mm256_set1_epi32(key[0]);
    mask[0] = _mm256_movemask_ps(_mm256_cmpeq_epi32(bucket0->kv, kv0));
    __m256 kv1 = _mm256_set1_epi32(key[1]);
    mask[1] = _mm256_movemask_ps(_mm256_cmpeq_epi32(bucket1->kv, kv1));
    int ret = 0;
    if (mask[0] == 0) {
        ret |= 1;
    } else {
        value[0] = bucket0->value[__builtin_ctz(mask[0])];
    }
    if (mask[1] == 0) {
        ret |= 2;
    } else {
        value[1] = bucket1->value[__builtin_ctz(mask[1])];
    }
    return ret;
#endif
}

int Table1_get_batch4(Table1Bucket* table, size_t n_bits, u64 H[2], u32 key[4], u64 value[4]) {
    size_t bucket[4];
    u8 mask[4];
    for (size_t i = 0; i < 4; i++) { bucket[i] = hash1(key[i], n_bits, H); }
    for (size_t i = 0; i < 4; i++) { mask[i] = ymm_entry_cmp(table[bucket[i]].kv, key[i]); }
    int ret = 0;
    for (size_t i = 0; i < 4; i++) {
        if (mask[i] == 0) {
            ret |= (1 << i);
        } else {
            value[i] = table[bucket[i]].value[__builtin_ctz(mask[i])];
        }
    }
    return ret;
}

size_t Table1_try_build(Table1Bucket* table, size_t n_bits, u32* keys, size_t target_size, size_t tries, u64* H) {
    PRNG32RomuQuad rng;
    size_t try = 0;
    while (try <= tries) {
        try += 1;
        rng_init(&rng, 42 + try);
        u64 Htry[2] = {
            rng_u64(&rng) * 2 + 1,
            rng_u64(&rng),
        };
        Table1_reset(table, n_bits);

        /*printf("trying H=%ld\n", Htry);*/

        size_t occupancy = 0;
        while (occupancy < target_size) {
#ifdef SEQKEY
            u32 key = occupancy + 1;
#else
            u32 key = rng_u32(&rng);
#endif
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
        H[0] = Htry[0];
        H[1] = Htry[1];
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
        /*printf("lo=%ld hi=%ld trying with target_size %ld\n", lo, hi, target_size);*/
        size_t try = Table1_try_build(table, n_bits, keys, target_size, tries, H);
        if (try == 0) { // didn't succeed
            hi = target_size;
        } else {
            lo = target_size;
        }
    }
    size_t try = Table1_try_build(table, n_bits, keys, lo, tries, H);
    (void)try;
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
    Table2Bucket* ret = aligned_alloc(32, Table2_n_buckets(n_bits) * sizeof(Table2Bucket));
    assert(ret != NULL);
    return ret;
}

void Table2_reset(Table2Bucket* table, size_t n_bits) {
    for (size_t i = 0; i < Table2_n_buckets(n_bits); i++) {
        table[i].kv[0] = _mm256_setzero_si256();
        table[i].kv[1] = _mm256_setzero_si256();
    }
}

int Table2_insert(Table2Bucket* table, size_t n_bits, u64 H[2], u32 key, u64 value) {
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

int Table2_get(Table2Bucket* table, size_t n_bits, u64 H[2], u32 key, u64* value) {
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
        u64 Htry[2] = {
            rng_u64(&rng) * 2 + 1,
            rng_u64(&rng),
        };
        Table2_reset(table, n_bits);

        /*printf("trying H=%ld\n", Htry);*/

        size_t occupancy = 0;
        while (occupancy < target_size) {
#ifdef SEQKEY
            u32 key = occupancy + 1;
#else
            u32 key = rng_u32(&rng);
#endif
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
        H[0] = Htry[0];
        H[1] = Htry[1];
        return try;
nexttry:
        (void)0;
    }
    return 0;
}

size_t Table2_bsearch_occupancy(Table2Bucket* table, size_t n_bits, u32* keys, size_t tries, u64 H[2]) {
    size_t size = ((u64)1) << n_bits;
    size_t size_entries = size * 8;
    size_t lo = 1;
    size_t hi = size_entries - 1;
    while (hi - lo > 1) {
        size_t target_size = lo + (hi-lo)/2;
        /*printf("lo=%ld hi=%ld trying with target_size %ld\n", lo, hi, target_size);*/
        size_t try = Table2_try_build(table, n_bits, keys, target_size, tries, H);
        if (try == 0) { // didn't succeed
            hi = target_size;
        } else {
            lo = target_size;
        }
    }
    size_t try = Table2_try_build(table, n_bits, keys, lo, tries, H);
    (void)try;
    assert(try != 0);
    return lo;
}

// ------------------------------------------------------------------------------------------------------

int main() {
/*#define NOOP*/
#ifdef NOOP
    {
        PRNG32RomuQuad rng;
        rng_init(&rng, 42);
        /*u64 H = rng_u64(&rng);*/
        u32 H = rng_u32(&rng);
        for (size_t i = 0; i < 100; i++) {
            u32 key = rng_u32(&rng);
            printf("%lx hash=%d\n", key, hash1(key, 8, H));
        }
        /*return 0;*/
    }
#endif

    printf("alignof(Table1Bucket) = %ld\n", _Alignof(Table1Bucket));
    printf("alignof(Table2Bucket) = %ld\n", _Alignof(Table2Bucket));

#ifndef NDEBUG
        const size_t rounds = 2000;
#else
        const size_t rounds = 10;
#endif

    size_t tries = 10000;
    size_t target_size = 1ll << 14;
    size_t table_1_bits = __builtin_ctz(target_size / 8);
    size_t table_2_bits = __builtin_ctz(target_size / 16);
    printf("table_1_bits=%ld table_2_bits=%ld\n", table_1_bits, table_2_bits);
    printf("-------------\n");

    {
        size_t n_bits = table_1_bits;
        size_t n_buckets = Table1_n_buckets(n_bits);
        size_t n_entries = Table1_n_entries(n_bits);
        Table1Bucket* table = Table1_create(n_bits);
        u32* keys = calloc(n_entries, sizeof(u32));
        u64 H[2];
        /*size_t try = try_table_build(table, n_bits, keys, target_size, tries, H);*/
        size_t got_entries = Table1_bsearch_occupancy(table, n_bits, keys, tries, H);
        printf("table1 got %ld num_entries\n", got_entries);
        printf("table1 %ld buckets, %ld entries\n", n_buckets, n_entries);
        printf("%.2f occupancy\n", (double)got_entries/(double)n_entries);
        printf("H = {%lx, %lx}\n", H[0], H[1]);

#ifndef NDEBUG
        for (size_t i = 0; i < got_entries; i++) {
            u64 value;
            int ret = Table1_get(table, n_bits, H, keys[i], &value);
            assert(ret == 0);
            assert((u64)keys[i] == value);
        }
#endif

        {
            u64 check = 0;
            u64 present = 0;
            Timespec start, stop;

            clock_ns(&start);
            for (size_t round = 0; round < rounds; round++) {
#pragma unroll 1
                for (size_t i = 0; i < got_entries; i++) {
                    u64 value;
                    int ret = Table1_get(table, n_bits, H, keys[i], &value);
                    present |= ret;
                    check += value;
                }
            }
            clock_ns(&stop);

            printf("%.2f ns/lookup %ld lookups (1) %.2f ms present=%ld check=%lx\n", (double)elapsed_ns(start, stop) / (double)rounds / (double)got_entries, rounds * got_entries, (double)elapsed_ns(start, stop) / 1000000, present, check);
        }

        {
            u64 check = 0;
            u64 present = 0;
            Timespec start, stop;
            size_t lookups = 0;

            clock_ns(&start);
            for (size_t round = 0; round < rounds; round++) {
#pragma unroll 1
                for (size_t i = 0; i < got_entries - 2; i += 2) {
                    lookups += 2;
                    u64 value[2];
                    u32 k[2] = {keys[i], keys[i + 1]};
                    int ret = Table1_get_batch2(table, n_bits, H, k, value);
                    present |= ret;
                    check += value[0];
                    check += value[1];
                }
            }
            clock_ns(&stop);

            printf("%.2f ns/lookup %ld lookups (2) %.2f ms present=%ld check=%lx\n", (double)elapsed_ns(start, stop) / (double)lookups, lookups, (double)elapsed_ns(start, stop) / 1000000, present, check);
        }

        {
            u64 check = 0;
            u64 present = 0;
            Timespec start, stop;
            size_t lookups = 0;

            clock_ns(&start);
            for (size_t round = 0; round < rounds; round++) {
#pragma unroll 1
                for (size_t i = 0; i < got_entries - 4; i += 4) {
                    lookups += 4;
                    u64 value[4];
                    u32 k[4] = {keys[i], keys[i + 1], keys[i + 2], keys[i + 4]};
                    int ret = Table1_get_batch4(table, n_bits, H, k, value);
                    present |= ret;
                    check += value[0];
                    check += value[1];
                    check += value[2];
                    check += value[3];
                }
            }
            clock_ns(&stop);

            printf("%.2f ns/lookup %ld lookups (4) %.2f ms present=%ld check=%lx\n", (double)elapsed_ns(start, stop) / (double)lookups, lookups, (double)elapsed_ns(start, stop) / 1000000, present, check);
        }

        free(keys);
        free(table);
    }

    printf("-------------\n");

    if (0) {
        size_t n_bits = table_2_bits;
        size_t n_buckets = Table2_n_buckets(n_bits);
        size_t n_entries = Table2_n_entries(n_bits);
        Table2Bucket* table = Table2_create(n_bits);
        u32* keys = calloc(n_entries, sizeof(u32));
        u64 H[2];
        size_t got_entries = Table2_bsearch_occupancy(table, n_bits, keys, tries, H);
        printf("table2 got %ld num_entries\n", got_entries);
        printf("table2 %ld buckets, %ld entries\n", n_buckets, n_entries);
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

        clock_ns(&start);
        for (size_t round = 0; round < rounds; round++) {
#pragma unroll 1
            for (size_t i = 0; i < got_entries; i++) {
                u64 value;
                int ret = Table2_get(table, n_bits, H, keys[i], &value);
                present |= ret;
                check += value;
            }
        }
        clock_ns(&stop);

        printf("%.2f ns/lookup %ld lookups %.2f ms present=%ld check=%lx\n", (double)elapsed_ns(start, stop) / (double)rounds / (double)got_entries, rounds * got_entries, (double)elapsed_ns(start, stop) / 1000000, present, check);
        free(keys);
        free(table);
    }


}
