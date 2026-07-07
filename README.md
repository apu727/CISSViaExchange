

# Prequisites:

All scripts start in the root of the repository. 


Compile kissfft: https://github.com/mborgerding/kissfft
```
cd cppNEGF/third-party/kissfft
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install -DKISSFFT_DATATYPE=double -DKISSFFT_STATIC=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DKISSFFT_TEST=OFF ..
cmake --build . --target install --parallel N
```

Setup python venv:

```
python3 -m venv .
source bin/activate
pip install -r requirements.txt
```

Add the repository root to pythonpath

```
export PYTHONPATH=.
```

Obtain and build RevQCMagic and symlink the qcmagic folder into the repository root. 

```
ln -s /path/to/RevQCMagic/qcmagic .
```


Build the cppNEGF Code
```
cd cppNEGF
mkdir build
cd build
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release -DCOMPILE_PYTHON_LIBS=On
cmake --build . --target install --parallel N
```

This will create a dynamic .so library in the root of the repository. 

To run a calculation see the commands in Calculations/runall.sh

E.g.

```OMP_NUM_THREADS=16 taskset 0xFFFF python3 -u Calculations/CHO_PEX/CHO_CuSmall_PE0.04_Current_Z/CHO_CuSmall_EQ.py```

Note that the files 
```
Calculations/CHO_PEX/CHODAmat.dat
Calculations/CHO_PEX/CHOFAmat.dat
Calculations/CHO_PEX/CHO.fchk
Calculations/CHO_PEX/CHO.out
Calculations/CHO_PEX/CHOSmat.dat
```

Need to be copied to the respective subfolder before running. 
