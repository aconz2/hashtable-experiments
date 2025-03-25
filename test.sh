#!/usr/bin/env bash

set -e

CC=${CC:-clang}

$CC -fno-sanitize-recover=all -fsanitize=undefined,address,leak -g -Wall -march=native -O2 table.c -o table-test

./table-test
