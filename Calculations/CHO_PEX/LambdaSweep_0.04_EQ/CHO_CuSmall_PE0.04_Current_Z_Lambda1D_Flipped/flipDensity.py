import numpy as np
FILEPATH = f"Calculations/CHO_PEX/LambdaSweep_0.04_EQ/CHO_CuSmall_PE0.04_Current_Z_Lambda1D_Flipped/CHOfinalGn.npy"

M = np.load(FILEPATH)
Mf = np.empty_like(M)
Mf[0::2, 0::2] =  M[1::2, 1::2]     # BB -> AA
Mf[1::2, 1::2] =  M[0::2, 0::2]     # AA -> BB
Mf[0::2, 1::2] = -M[0::2, 1::2]     # AB -> -AB
Mf[1::2, 0::2] = -M[1::2, 0::2]     # BA -> -BA
np.save(FILEPATH, Mf)  # m -> -m (all components)