#!/usr/bin/env bash

set -e

CC=${CC:-clang}

$CC -g -Wall -march=native -O2 table.c -o table

taskset -c 30 ./table
