#! /usr/bin/bash
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_Z/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_Z/log 1>&2
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_X/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_X/log 1>&2
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0.01_Current_Z/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0.01_Current_Z/log 1>&2
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0.01_Current_X/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0.01_Current_X/log 1>&2
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0_Current_Z/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0_Current_Z/log 1>&2
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0_Current_X/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0_Current_X/log 1>&2
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0_Current_Z_RHF/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0_Current_Z_RHF/log 1>&2
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0_Current_X_RHF/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0_Current_X_RHF/log 1>&2

#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_0.5X_Z/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_0.5X_Z/log 1>&2
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_0.5X_X/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_0.5X_X/log 1>&2
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_0.1X_Z/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_0.1X_Z/log 1>&2
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_0.1X_X/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_0.1X_X/log 1>&2
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_0.1XScat_Z/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_0.1XScat_Z/log 1>&2
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_0.1XScat_Z2/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_0.1XScat_Z2/log 1>&2
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_RHF_Z/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_RHF_Z/log 1>&2
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0.04_EQ/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0.04_EQ/log 1>&2
#cp Calculations/CHO_PEX/CHO_CuSmall_PE0.04_EQ/CHOfinalGn.npy Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_LR_Z/CHOfinalGn.npy
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_LR_Z/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_LR_Z/log 1>&2
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_0.04_PE0_Current_Z/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_0.04_PE0_Current_Z/log 1>&2
#OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_0_PE0.04_Current_Z/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_0_PE0.04_Current_Z/log 1>&2
OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_0_EQ/CHO_CuSmall_EQ.py  2>> Calculations/CHO_PEX/CHO_CuSmall_0_EQ/log 1>&2

