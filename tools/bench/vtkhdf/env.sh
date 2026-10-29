module purge >/dev/null 2>&1
module load cpe craype-x86-genoa PrgEnv-cray cray-hdf5-parallel cmake
export PHARE_HOME=${PHARE_SRC:?set PHARE_SRC to a PHARE checkout}
source ${PHARE_VENV:?set PHARE_VENV to a Python venv with PHARE requirements and h5py}/bin/activate
export PYTHONPATH=$PHARE_HOME:$PHARE_HOME/pyphare:$PHARE_HOME/build
