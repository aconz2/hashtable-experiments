#!/usr/bin/env bash

set -e

CC=${CC:-clang}

$CC -DNDEBUG -g -Wall -march=native -O2 table.c -o table

taskset -c 3 ./table
