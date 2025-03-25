experiments with a read only hashtable (construct once, read many), inspired by the swiss table. Each bucket has 8 32 bit keys (key 0 is the empty key) and we compare with simd; no probing so you have to try hash values until you succeed in building a table with no more than 8 collisions per bucket. Good for precomputed hash tables (especially because the hash constant and shift amount can be compile time  constants). In this test values are 64 bit. Currently occupancy is around 45% with a simple mul and shift hash. For modest size maps, I'm not sure the tradeoff in occupancy and hash complexity is worthwhile and depends on your cache size and usage pattern.

Better perf numbers and comparison coming but with 2048 entries at 0.45 occupancy I get 1.02 ns/lookup in a tight loop only looking up keys that exist.
