Experiments with a read only hashtable (construct once, read many), inspired by the swiss table. Each bucket has 8 32 bit keys (key 0 is the empty key) and we compare with simd; no probing so you have to try hash values until you succeed in building a table with no more than 8 collisions per bucket. Good for precomputed hash tables (especially because the hash constant and shift amount can be compile time  constants). In this test values are 64 bit. Currently occupancy is around 45% with a simple mul and shift hash. For modest size maps, I'm not sure the tradeoff in occupancy and hash complexity is worthwhile and depends on your cache size and usage pattern.

It is annoying with 64 bit values because then the bucket is 92 bytes, so indexing requires a multiplication by 3 which adds an instruction. You could switch to 32 bit values to get 64 byte or add 4x64 padding to get 128 byte, and maybe those padding bytes would be useful for something (like getting a 32 bit value and 64 bit value per key).

Warning the code is a total playground.

One thing I learned is that if you use version b with a hash function that promises to have the low bits zero in accordance with the size of the bucket, then you can mask once in the hash function and already have the offset ready. Whereas with version a, you have mul, mask off high bits, shift left for index calculation. Version b is mul, mask. with 16384 entries at 0.36 max occupancy (5890 entries), it is 1.34 ns/lookup vs 1.2 ns/lookup and takes up 130kb or about 25% of L1d on a 5950x.

```c
Table1Bucket* bucket = table + hash1(key, n_bits, H); // a
Table1Bucket* bucket = (Table1Bucket*)((char*)table + hash1(key, n_bits, H)); // b
```

```asm
# version a: key in ecx, rdx has hash constant, rbs ix table base pointer
imul    rdx, rcx
vmovd   xmm0, ecx
vpbroadcastd    ymm0, xmm0
# this sometimes generates a bzhi
shr     rdx, 0x35
shl     edx, 0x7
vpcmpeqd        ymm0, ymm0, ymmword ptr [rbx + rdx]

# version b; key in ecx, edx has hash constant, rbx is table base pointer
imul    edx, ecx
vmovd   xmm0, ecx
vpbroadcastd    ymm0, xmm0
and     edx, 0x3ff80
vpcmpeqd        ymm0, ymm0, ymmword ptr [rbx + rdx]
vmovmskps       esi, ymm0
```

Not entirely sure I understand when it does a 64 bit vs 32 bit multiplication, but a 32 bit constant would be nice since when you make this compile time, it means the constant in the instruction stream is 4 instead of 8 bytes.
