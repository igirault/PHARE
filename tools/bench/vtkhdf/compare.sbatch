#!/bin/bash -l
#SBATCH --constraint=GENOA
#SBATCH --exclusive
#SBATCH --ntasks-per-node=192
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --output=%x_%j.out

# usage, from this directory:
#   PHARE_PR=<checkout> PHARE_PP=<checkout> PHARE_VENV=<venv> E2E_CELLS=x,y,z E2E_DUMPS=d \
#       sbatch -A <account> -N <nodes> -t <time> -J <name> compare.sbatch <build>:<largest>[:keep[:dumps]] ...
# build is pr (checkout PHARE_PR) or pp (checkout PHARE_PP)

HERE=${BENCH_DIR:-$SLURM_SUBMIT_DIR}
OUT=${SCRATCHDIR:?set SCRATCHDIR}/vtkhdf_bench/${SLURM_JOB_ID}
NT=$((SLURM_NNODES * 192))
STRIPE=${STRIPE:-2}
DUMPS=${E2E_DUMPS:-6}
echo "nodes=$SLURM_NNODES ntasks=$NT cells=$E2E_CELLS dumps=$DUMPS stripe=$STRIPE"

for spec in "$@"; do
    IFS=: read -r build largest keep dumps <<< "$spec"
    case $build in
        pr) PHARE_SRC=${PHARE_PR:?set PHARE_PR} ;;
        pp) PHARE_SRC=${PHARE_PP:?set PHARE_PP} ;;
        *) echo "unknown build $build"; exit 1 ;;
    esac
    export PHARE_SRC E2E_LARGEST=$largest E2E_DUMPS=${dumps:-$DUMPS}
    source $HERE/env.sh
    dir=$OUT/${build}_p${largest}
    mkdir -p $dir
    lfs setstripe -c $STRIPE -S 1M $dir
    cd $dir
    echo "=== build=$build largest=$largest dumps=$E2E_DUMPS"
    start=$(date +%s)
    E2E_DIR=$dir/diags srun -n $NT -t ${STEP_MIN:-10} --cpu-bind=cores --kill-on-bad-exit=1 \
        python3 $HERE/vtkhdf_dump.py < /dev/null 2>&1 | grep -E "E2E|rror|Abort" | head -20
    rc=${PIPESTATUS[0]}
    boxes=$(python3 -c "import h5py; print(int(h5py.File('$dir/diags/EM_B.vtkhdf', 'r')['/VTKHDF/Steps/Level0/NumberOfAMRBox'][0]))" 2>/dev/null)
    echo "=== build=$build largest=$largest rc=$rc wall=$(( $(date +%s) - start ))s boxes=$boxes ranks=$NT size=$(du -sh $dir/diags | cut -f1)"
    cd $HERE
    if [ "$keep" = keep ]; then echo "=== kept $dir"; else rm -rf $dir; fi
done
