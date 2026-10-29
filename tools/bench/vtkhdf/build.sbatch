#!/bin/bash -l
#SBATCH --constraint=GENOA
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=00:50:00
#SBATCH --output=%x_%j.out

# usage, from this directory: BUILD_SRCS="<checkout> ..." PHARE_VENV=<venv> sbatch -A <account> build.sbatch

HERE=${BENCH_DIR:-$SLURM_SUBMIT_DIR}
for PHARE_SRC in ${BUILD_SRCS:?set BUILD_SRCS to the checkouts to build}; do
export PHARE_SRC
source $HERE/env.sh
module list 2>&1 | tail -4
which python3 CC cc cmake
cd $PHARE_HOME
echo "### building $PHARE_HOME"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=CC -DCMAKE_C_COMPILER=cc \
    -DCMAKE_CXX_FLAGS="-march=native -mtune=native -DPHARE_DIAG_DOUBLES=1" \
    -Dtest=OFF -DdevMode=OFF -DwithPhlop=OFF -DSAMRAI_BUILD_SHARED_LIBS=ON -DHighFive=ON \
    -DPHARE_PERMUTATIONS=$HERE/permutations.txt \
    -DPython_EXECUTABLE=$(which python3) || exit 1

start=$(date +%s)
make -C build -j 96 2>&1 | grep -E "error|Error|warning: .*-W|Built target" | tail -60
echo "build rc=${PIPESTATUS[0]} wall=$(( $(date +%s) - start ))s"
ls -la build/pybindlibs/
done
