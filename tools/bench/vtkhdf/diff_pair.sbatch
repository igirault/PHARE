#!/bin/bash -l
#SBATCH --constraint=GENOA
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out

# usage: sbatch -A <account> -t <time> -J <name> diff_pair.sbatch <dir A> <dir B>
# compares every A/diags/*.vtkhdf with B/diags/, then deletes both directories
module load cray-hdf5-parallel
A=$1
B=$2
for f in $A/diags/*.vtkhdf; do
    g=$B/diags/$(basename $f)
    (h5diff $f $g > /dev/null 2>&1; echo "H5DIFF rc=$? $(basename $f)") &
done
wait
du -sh $A/diags $B/diags
rm -rf $A $B
