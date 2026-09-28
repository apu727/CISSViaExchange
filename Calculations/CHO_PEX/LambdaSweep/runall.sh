cp Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0/CHOfinalGn.npy Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0.2/CHOfinalGn.npy
OMP_NUM_THREADS=16 taskset 0xffff0000 python3 -u Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0.2/CHO_CuSmall_EQ.py >> Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0.2/log 2>&1

cp Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0.2/CHOfinalGn.npy Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0.4/CHOfinalGn.npy
OMP_NUM_THREADS=16 taskset 0xffff0000 python3 -u Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0.4/CHO_CuSmall_EQ.py >> Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0.4/log 2>&1

cp Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0.4/CHOfinalGn.npy Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0.6/CHOfinalGn.npy
OMP_NUM_THREADS=16 taskset 0xffff0000 python3 -u Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0.6/CHO_CuSmall_EQ.py >> Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0.6/log 2>&1

cp Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0.6/CHOfinalGn.npy Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0.8/CHOfinalGn.npy
OMP_NUM_THREADS=16 taskset 0xffff0000 python3 -u Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0.8/CHO_CuSmall_EQ.py >> Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0.8/log 2>&1

cp Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda0.8/CHOfinalGn.npy Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda1/CHOfinalGn.npy
OMP_NUM_THREADS=16 taskset 0xffff0000 python3 -u Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda1/CHO_CuSmall_EQ.py >> Calculations/CHO_PEX/LambdaSweep/CHO_CuSmall_PE0.04_Current_Z_Lambda1/log 2>&1






