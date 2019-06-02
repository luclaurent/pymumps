__all__ = ['DMUMPS_STRUC_C', 'zmumps_c', 'cast_array']

########################################################################
# libzmumps / zmumps_c.h wrappers (using Cython)
########################################################################
 
MUMPS_INT_DTYPE = 'i'
ZMUMPS_REAL_DTYPE = 'd'
ZMUMPS_COMPLEX_DTYPE = 'z'

from libc.string cimport strncpy

cdef extern from "zmumps_c.h":

    ctypedef int MUMPS_INT
    ctypedef long MUMPS_INT8

    ctypedef struct mumps_double_complex:
        double r
        double i

    ctypedef mumps_double_complex ZMUMPS_COMPLEX
    ctypedef double ZMUMPS_REAL
    
    char* MUMPS_VERSION

    ctypedef struct c_ZMUMPS_STRUC_C "ZMUMPS_STRUC_C":
        MUMPS_INT      sym, par, job
        MUMPS_INT      comm_fortran    # Fortran communicator
        MUMPS_INT      icntl[40]
        MUMPS_INT      keep[500]
        ZMUMPS_REAL    cntl[15]
        ZMUMPS_REAL    dkeep[230]
        MUMPS_INT8     keep8[150]
        MUMPS_INT      n

        # used in matlab interface to decide if we
        # free + malloc when we have large variation
        MUMPS_INT      nz_alloc

        # Assembled entry
        MUMPS_INT      nz
        MUMPS_INT8     nnz
        MUMPS_INT      *irn
        MUMPS_INT      *jcn
        ZMUMPS_COMPLEX *a

        # Distributed entry
        MUMPS_INT      nz_loc
        MUMPS_INT8     nnz_loc
        MUMPS_INT      *irn_loc
        MUMPS_INT      *jcn_loc
        ZMUMPS_COMPLEX *a_loc

        # Element entry
        MUMPS_INT      nelt
        MUMPS_INT      *eltptr
        MUMPS_INT      *eltvar
        ZMUMPS_COMPLEX *a_elt

        # Ordering, if given by user
        MUMPS_INT      *perm_in

        # Orderings returned to user
        MUMPS_INT      *sym_perm    # symmetric permutation
        MUMPS_INT      *uns_perm    # column permutation

        # Scaling (input only in this version)
        ZMUMPS_REAL    *colsca
        ZMUMPS_REAL    *rowsca
        MUMPS_INT      colsca_from_mumps
        MUMPS_INT      rowsca_from_mumps

        # RHS, solution, ouptput data and statistics
        ZMUMPS_COMPLEX *rhs
        ZMUMPS_COMPLEX *redrhs
        ZMUMPS_COMPLEX *rhs_sparse
        ZMUMPS_COMPLEX *sol_loc
        MUMPS_INT      *irhs_sparse
        MUMPS_INT      *irhs_ptr
        MUMPS_INT      *isol_loc
        MUMPS_INT      nrhs
        MUMPS_INT      lrhs
        MUMPS_INT      lredrhs
        MUMPS_INT      nz_rhs
        MUMPS_INT      lsol_loc
        MUMPS_INT      schur_mloc
        MUMPS_INT      schur_nloc
        MUMPS_INT      schur_lld
        MUMPS_INT      mblock
        MUMPS_INT      nblock
        MUMPS_INT      nprow
        MUMPS_INT      npcol
        MUMPS_INT      info[40]
        MUMPS_INT      infog[40]
        ZMUMPS_REAL    rinfo[40]
        ZMUMPS_REAL    rinfog[40]

        # Null space
        MUMPS_INT      deficiency
        MUMPS_INT      *pivnul_list
        MUMPS_INT      *mapping

        # Schur
        MUMPS_INT      size_schur
        MUMPS_INT      *listvar_schur
        ZMUMPS_COMPLEX *schur

        # Internal parameters
        MUMPS_INT      instance_number
        ZMUMPS_COMPLEX *wk_user

        char *version_number
        # For out-of-core
        char *ooc_tmpdir
        char *ooc_prefix
        # To save the matrix in matrix market format
        char *write_problem
        MUMPS_INT      lwk_user
    void c_zmumps_c "zmumps_c" (c_ZMUMPS_STRUC_C *) nogil

cdef class ZMUMPS_STRUC_C:
    cdef c_ZMUMPS_STRUC_C ob

    property sym:
        def __get__(self): return self.ob.sym
        def __set__(self, value): self.ob.sym = value
    property par:
        def __get__(self): return self.ob.par
        def __set__(self, value): self.ob.par = value
    property job:
        def __get__(self): return self.ob.job
        def __set__(self, value): self.ob.job = value

    property comm_fortran:
        def __get__(self): return self.ob.comm_fortran
        def __set__(self, value): self.ob.comm_fortran = value

    property icntl:
        def __get__(self):
            cdef MUMPS_INT[:] view = self.ob.icntl
            return view
    property cntl:
        def __get__(self):
            cdef ZMUMPS_REAL[:] view = self.ob.cntl
            return view

    property n:
        def __get__(self): return self.ob.n
        def __set__(self, value): self.ob.n = value
    property nz_alloc:
        def __get__(self): return self.ob.nz_alloc
        def __set__(self, value): self.ob.nz_alloc = value

    property nz:
        def __get__(self): return self.ob.nz
        def __set__(self, value): self.ob.nz = value
    property irn:
        def __get__(self): return <long> self.ob.irn
        def __set__(self, long value): self.ob.irn = <MUMPS_INT*> value
    property jcn:
        def __get__(self): return <long> self.ob.jcn
        def __set__(self, long value): self.ob.jcn = <MUMPS_INT*> value
    property a:
        def __get__(self): return <long> self.ob.a
        def __set__(self, long value): self.ob.a = <ZMUMPS_COMPLEX*> value

    property nz_loc:
        def __get__(self): return self.ob.nz_loc
        def __set__(self, value): self.ob.nz_loc = value
    property irn_loc:
        def __get__(self): return <long> self.ob.irn_loc
        def __set__(self, long value): self.ob.irn_loc = <MUMPS_INT*> value
    property jcn_loc:
        def __get__(self): return <long> self.ob.jcn_loc
        def __set__(self, long value): self.ob.jcn_loc = <MUMPS_INT*> value
    property a_loc:
        def __get__(self): return <long> self.ob.a_loc
        def __set__(self, long value): self.ob.a_loc = <ZMUMPS_COMPLEX*> value

    property nelt:
        def __get__(self): return self.ob.nelt
        def __set__(self, value): self.ob.nelt = value
    property eltptr:
        def __get__(self): return <long> self.ob.eltptr
        def __set__(self, long value): self.ob.eltptr = <MUMPS_INT*> value
    property eltvar:
        def __get__(self): return <long> self.ob.eltvar
        def __set__(self, long value): self.ob.eltvar = <MUMPS_INT*> value
    property a_elt:
        def __get__(self): return <long> self.ob.a_elt
        def __set__(self, long value): self.ob.a_elt = <ZMUMPS_COMPLEX*> value

    property perm_in:
        def __get__(self): return <long> self.ob.perm_in
        def __set__(self, long value): self.ob.perm_in = <MUMPS_INT*> value

    property sym_perm:
        def __get__(self): return <long> self.ob.sym_perm
        def __set__(self, long value): self.ob.sym_perm = <MUMPS_INT*> value
    property uns_perm:
        def __get__(self): return <long> self.ob.uns_perm
        def __set__(self, long value): self.ob.uns_perm = <MUMPS_INT*> value

    property colsca:
        def __get__(self): return <long> self.ob.colsca
        def __set__(self, long value): self.ob.colsca = <ZMUMPS_REAL*> value
    property rowsca:
        def __get__(self): return <long> self.ob.rowsca
        def __set__(self, long value): self.ob.rowsca = <ZMUMPS_REAL*> value

    property rhs:
        def __get__(self): return <long> self.ob.rhs
        def __set__(self, long value): self.ob.rhs = <ZMUMPS_COMPLEX*> value
    property redrhs:
        def __get__(self): return <long> self.ob.redrhs
        def __set__(self, long value): self.ob.redrhs = <ZMUMPS_COMPLEX*> value
    property rhs_sparse:
        def __get__(self): return <long> self.ob.rhs_sparse
        def __set__(self, long value): self.ob.rhs_sparse = <ZMUMPS_COMPLEX*> value
    property sol_loc:
        def __get__(self): return <long> self.ob.sol_loc
        def __set__(self, long value): self.ob.sol_loc = <ZMUMPS_COMPLEX*> value


    property irhs_sparse:
        def __get__(self): return <long> self.ob.irhs_sparse
        def __set__(self, long value): self.ob.irhs_sparse = <MUMPS_INT*> value
    property irhs_ptr:
        def __get__(self): return <long> self.ob.irhs_ptr
        def __set__(self, long value): self.ob.irhs_ptr = <MUMPS_INT*> value
    property isol_loc:
        def __get__(self): return <long> self.ob.isol_loc
        def __set__(self, long value): self.ob.isol_loc = <MUMPS_INT*> value

    property nrhs:
        def __get__(self): return self.ob.nrhs
        def __set__(self, value): self.ob.nrhs = value
    property lrhs:
        def __get__(self): return self.ob.lrhs
        def __set__(self, value): self.ob.lrhs = value
    property lredrhs:
        def __get__(self): return self.ob.lredrhs
        def __set__(self, value): self.ob.lredrhs = value
    property nz_rhs:
        def __get__(self): return self.ob.nz_rhs
        def __set__(self, value): self.ob.nz_rhs = value
    property lsol_loc:
        def __get__(self): return self.ob.lsol_loc
        def __set__(self, value): self.ob.lsol_loc = value

    property schur_mloc:
        def __get__(self): return self.ob.schur_mloc
        def __set__(self, value): self.ob.schur_mloc = value
    property schur_nloc:
        def __get__(self): return self.ob.schur_nloc
        def __set__(self, value): self.ob.schur_nloc = value
    property schur_lld:
        def __get__(self): return self.ob.schur_lld
        def __set__(self, value): self.ob.schur_lld = value


    property mblock:
        def __get__(self): return self.ob.mblock
        def __set__(self, value): self.ob.mblock = value
    property nblock:
        def __get__(self): return self.ob.nblock
        def __set__(self, value): self.ob.nblock = value
    property nprow:
        def __get__(self): return self.ob.nprow
        def __set__(self, value): self.ob.nprow = value
    property npcol:
        def __get__(self): return self.ob.npcol
        def __set__(self, value): self.ob.npcol = value

    property info:
        def __get__(self):
            cdef MUMPS_INT[:] view = self.ob.info
            return view
    property infog:
        def __get__(self):
            cdef MUMPS_INT[:] view = self.ob.infog
            return view

    property rinfo:
        def __get__(self):
            cdef ZMUMPS_REAL[:] view = self.ob.rinfo
            return view
    property rinfog:
        def __get__(self):
            cdef ZMUMPS_REAL[:] view = self.ob.rinfog
            return view

    property deficiency:
        def __get__(self): return self.ob.deficiency
        def __set__(self, value): self.ob.deficiency = value
    property pivnul_list:
        def __get__(self): return <long> self.ob.pivnul_list
        def __set__(self, long value): self.ob.pivnul_list = <MUMPS_INT*> value
    property mapping:
        def __get__(self): return <long> self.ob.mapping
        def __set__(self, long value): self.ob.mapping = <MUMPS_INT*> value

    property size_schur:
        def __get__(self): return self.ob.size_schur
        def __set__(self, value): self.ob.size_schur = value
    property listvar_schur:
        def __get__(self): return <long> self.ob.listvar_schur
        def __set__(self, long value): self.ob.listvar_schur = <MUMPS_INT*> value
    property schur:
        def __get__(self): return <long> self.ob.schur
        def __set__(self, long value): self.ob.schur = <ZMUMPS_COMPLEX*> value

    property instance_number:
        def __get__(self): return self.ob.instance_number
        def __set__(self, value): self.ob.instance_number = value
    property wk_user:
        def __get__(self): return <long> self.ob.wk_user
        def __set__(self, long value): self.ob.wk_user = <ZMUMPS_COMPLEX*> value

    property version_number:
        def __get__(self):
            return (<bytes> self.ob.version_number).decode('ascii')

    property ooc_tmpdir:
        def __get__(self):
            return (<bytes> self.ob.ooc_tmpdir).decode('ascii')
        def __set__(self, char *value):
            strncpy(self.ob.ooc_tmpdir, value, sizeof(self.ob.ooc_tmpdir))
    property ooc_prefix:
        def __get__(self):
            return (<bytes> self.ob.ooc_prefix).decode('ascii')
        def __set__(self, char *value):
            strncpy(self.ob.ooc_prefix, value, sizeof(self.ob.ooc_prefix))

    property write_problem:
        def __get__(self):
            return (<bytes> self.ob.write_problem).decode('ascii')
        def __set__(self, char *value):
            strncpy(self.ob.write_problem, value, sizeof(self.ob.write_problem))

    property lwk_user:
        def __get__(self): return self.ob.lwk_user
        def __set__(self, value): self.ob.lwk_user = value

def zmumps_c(ZMUMPS_STRUC_C s not None):
    with nogil:
        c_zmumps_c(&s.ob)

__version__ = (<bytes> MUMPS_VERSION).decode('ascii')

########################################################################
# Casting routines.
########################################################################

def cast_array(arr):
    """Convert numpy array to corresponding cffi pointer.

    The user is entirely responsible for ensuring the data is contiguous
    and for holding a reference to the underlying array.
    """
    dtype = arr.dtype
    if dtype == 'i':
        return arr.__array_interface__['data'][0]
    elif dtype == 'd':
        return arr.__array_interface__['data'][0]
    elif dtype == 'c16':
        return arr.__array_interface__['data'][0]
    else:
        raise ValueError("Unknown dtype %r" % dtype)
