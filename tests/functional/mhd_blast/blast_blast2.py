#!/usr/bin/env python3
from pyphare.simulator.simulator import startMPI

from blast import run_case

if __name__ == "__main__":
    startMPI()
    run_case("blast2")
