PyMUMPS: A parallel sparse direct solver
=========================================


Requirements
------------

* [MUMPS](https://mumps-solver.org/)
* [mpi4py](https://code.google.com/p/mpi4py/)


Installation
------------

If necessary define `LIBRARY_PATH` and `C_INCLUDE_PATH` before running the next steps.

PyMUMPS can be installed from PyPI using pip:

```
pip install pymumps
```

Custom build flags, e.g. to specify the MUMPS installation location,
can be specified using `-C`:

```
pip install -v \
    -Cbuild.verbose=true \
    -Ccmake.define.MUMPS_ROOT=<PATH_OF_MUMPS_INSTALLATION> \
    pymumps
```

There is also conda recipe:

```
conda install -c conda-forge pymumps
```

Define `LIBRARY_PATH`/`LD_LIBRARY_PATH` 
------------
 
You can define the two variables to give to the compiler the location of your MUMPS installation:

    export LIBRARY_PATH=<...>/lib
    export C_INCLUDE_PATH=<...>/include

In the case of MUMPS installed in your Python's environment folders (`lib`/`include`), you can get the path using 

    `BASE_DATA_PYTHON=$(python -c "from sysconfig import get_paths;print(get_paths()['data'])")`

and export 
    
    export LIBRARY_PATH=$BASE_DATA_PYTHON/lib
    export C_INCLUDE_PATH=$BASE_DATA_PYTHON/include


Test if pyMUMPS is installed
------------

Test the obtained installation using

    python -c "import mumps"


Examples
--------

Centralized input & output. The sparse matrix and right hand side are
input only on the rank 0 process. The system is solved using all
available processes and the result is available on the rank 0 process.

```python
from mumps import DMumpsContext
ctx = DMumpsContext()
if ctx.myid == 0:
    ctx.set_centralized_sparse(A)
    x = b.copy()
    ctx.set_rhs(x) # Modified in place
ctx.run(job=6) # Analysis + Factorization + Solve
ctx.destroy() # Cleanup
```

Re-use symbolic or numeric factorizations.

```python
from mumps import DMumpsContext
ctx = DMumpsContext()
if ctx.myid == 0:
    ctx.set_centralized_assembled_rows_cols(A.row+1, A.col+1) # 1-based
ctx.run(job=1) # Analysis

if ctx.myid == 0:
    ctx.set_centralized_assembled_values(A.data)
ctx.run(job=2) # Factorization

if ctx.myid == 0:
    x = b1.copy()
    ctx.set_rhs(x)
ctx.run(job=3) # Solve

# Reuse factorizations by running `job=3` with new right hand sides
# or analyses by supplying new values and running `job=2` to repeat
# the factorization process.
```

Use `spsolve` function

```
import numpy as np
import scipy as sp
import mumps
# test data
dataIRN = np.array([1, 2, 4, 5, 2, 1, 5, 3, 2, 3, 1, 3],dtype=np.int32)
dataJCN = np.array([2, 3, 3, 5, 1, 1, 2, 4, 5, 2, 3, 3],dtype=np.int32)
dataVAL = np.array([3.0, -3.0, 2.0, 1.0, 3.0, 2.0, 4.0, 2.0, 6.0, -1.0, 4.0, 1.0],dtype=np.float64)
dataRHS = np.array([20.0, 24.0, 9.0, 6.0, 13.0],dtype=np.float64)
dataSOL = np.array([1.0, 2.0, 3.0, 4.0, 5.0],dtype=np.float64)

# build matrix
A = mumps.sp.sparse.coo_matrix((dataVAL, (dataIRN - 1, dataJCN - 1)), 
                        shape=(5,5))
sol = spsolve(A, dataRHS)

```

Dev
--------

Run meson to keep logs (`meson`, `ninja` and `cython` could be installed using `pip install meson ninja cython`):

    meson setup --reconfigure build
    ninja -v -C build