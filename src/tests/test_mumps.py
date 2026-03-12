from unittest import TestCase
import numpy as np
import pytest
# import sys
try:
    import mumps
except ImportError:
    pass

reasonTxt = "Requires the PyMUMPS library"

# data test
dataIRN = [1, 2, 4, 5, 2, 1, 5, 3, 2, 3, 1, 3]
dataJCN = [2, 3, 3, 5, 1, 1, 2, 4, 5, 2, 3, 3]
dataVAL = [3.0, -3.0, 2.0, 1.0, 3.0, 2.0, 4.0, 2.0, 6.0, -1.0, 4.0, 1.0]
dataRHS = [20.0, 24.0, 9.0, 6.0, 13.0]

# @pytest.mark.skipif('mumps' not in sys.modules,
#                     reason=reasonTxt)


class TestMUMPS_complex64(TestCase):
    def setUp(self):
        pytest.importorskip('mumps._cmumps')
        self.n = 5
        self.irn = np.array(dataIRN, dtype=np.int32)
        self.jcn = np.array(dataJCN, dtype=np.int32)
        self.aval = np.array(dataVAL, dtype=np.complex64)
        #
        self.rhs = np.array(dataRHS, dtype=np.complex64)
        #
        self.ctx = mumps.CMumpsContext(par=1, sym=0)
        self.ctx.set_silent()

    def test_init(self):
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.ctx.destroy()
        assert self.ctx.destroyed

    def test_rhs(self):
        self.ctx.set_shape(self.n)
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.x = self.rhs.copy()
        self.ctx.set_rhs(self.x)
        self.ctx.destroy()
        assert self.ctx.destroyed

    def test_analyze(self):
        self.ctx.set_shape(self.n)
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.ctx.run(job=1)
        assert self.ctx.id.infog[0] >= 0
        self.ctx.destroy()
        assert self.ctx.destroyed

    def test_factorize(self):
        self.ctx.set_shape(self.n)
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.ctx.run(job=4)
        assert self.ctx.id.infog[0] >= 0
        self.ctx.destroy()
        assert self.ctx.destroyed

    def test_solve(self):
        self.ctx.set_shape(self.n)
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.x = self.rhs.copy()
        self.ctx.set_rhs(self.x)
        self.ctx.run(job=6)
        assert self.ctx.id.infog[0] >= 0
        self.ctx.destroy()
        assert self.ctx.destroyed
        assert np.any(self.x == np.array([1., 2., 3., 4., 5.], dtype=complex))

###########################################################################
###########################################################################
###########################################################################
###########################################################################


class TestMUMPS_complex128(TestCase):
    def setUp(self):
        pytest.importorskip('mumps._zmumps')
        self.n = 5
        self.irn = np.array(dataIRN, dtype=np.int32)
        self.jcn = np.array(dataJCN, dtype=np.int32)
        self.aval = np.array(dataVAL, dtype=np.complex128)
        #
        self.rhs = np.array(dataRHS, dtype=np.complex128)
        #
        self.ctx = mumps.ZMumpsContext(par=1, sym=0)
        self.ctx.set_silent()

    def test_init(self):
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.ctx.destroy()
        assert self.ctx.destroyed

    def test_rhs(self):
        self.ctx.set_shape(self.n)
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.x = self.rhs.copy()
        self.ctx.set_rhs(self.x)
        self.ctx.destroy()
        assert self.ctx.destroyed

    def test_analyze(self):
        self.ctx.set_shape(self.n)
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.ctx.run(job=1)
        assert self.ctx.id.infog[0] >= 0
        self.ctx.destroy()
        assert self.ctx.destroyed

    def test_factorize(self):
        self.ctx.set_shape(self.n)
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.ctx.run(job=4)
        assert self.ctx.id.infog[0] >= 0
        self.ctx.destroy()
        assert self.ctx.destroyed

    def test_solve(self):
        self.ctx.set_shape(self.n)
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.x = self.rhs.copy()
        self.ctx.set_rhs(self.x)
        self.ctx.run(job=6)
        assert self.ctx.id.infog[0] >= 0
        self.ctx.destroy()
        assert self.ctx.destroyed
        assert np.any(self.x == np.array([1., 2., 3., 4., 5.], dtype=complex))

###########################################################################
###########################################################################
###########################################################################
###########################################################################


class TestMUMPS_real64(TestCase):
    def setUp(self):
        pytest.importorskip('mumps._dmumps')
        self.n = 5
        self.irn = np.array(dataIRN, dtype=np.int32)
        self.jcn = np.array(dataJCN, dtype=np.int32)
        self.aval = np.array(dataVAL, dtype=np.float64)
        #
        self.rhs = np.array(dataRHS, dtype=np.float64)
        #
        self.ctx = mumps.DMumpsContext(par=1, sym=0)
        self.ctx.set_silent()

    def test_init(self):
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.ctx.destroy()
        assert self.ctx.destroyed

    def test_rhs(self):
        self.ctx.set_shape(self.n)
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.x = self.rhs.copy()
        self.ctx.set_rhs(self.x)
        self.ctx.destroy()
        assert self.ctx.destroyed

    def test_analyze(self):
        self.ctx.set_shape(self.n)
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.ctx.run(job=1)
        assert self.ctx.id.infog[0] >= 0
        self.ctx.destroy()
        assert self.ctx.destroyed

    def test_factorize(self):
        self.ctx.set_shape(self.n)
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.ctx.run(job=4)
        assert self.ctx.id.infog[0] >= 0
        self.ctx.destroy()
        assert self.ctx.destroyed

    def test_solve(self):
        self.ctx.set_shape(self.n)
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.x = self.rhs.copy()
        self.ctx.set_rhs(self.x)
        self.ctx.run(job=6)
        assert self.ctx.id.infog[0] >= 0
        self.ctx.destroy()
        assert self.ctx.destroyed
        assert np.any(self.x == np.array([1., 2., 3., 4., 5.], dtype=complex))

###########################################################################
###########################################################################
###########################################################################
###########################################################################


class TestMUMPS_real32(TestCase):
    def setUp(self):
        pytest.importorskip('mumps._smumps')
        self.n = 5
        self.irn = np.array(dataIRN, dtype=np.int32)
        self.jcn = np.array(dataJCN, dtype=np.int32)
        self.aval = np.array(dataVAL, dtype=np.float32)
        #
        self.rhs = np.array(dataRHS, dtype=np.float32)
        #
        self.ctx = mumps.SMumpsContext(par=1, sym=0)
        self.ctx.set_silent()

    def test_init(self):
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.ctx.destroy()
        assert self.ctx.destroyed

    def test_rhs(self):
        self.ctx.set_shape(self.n)
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.x = self.rhs.copy()
        self.ctx.set_rhs(self.x)
        self.ctx.destroy()
        assert self.ctx.destroyed

    def test_analyze(self):
        self.ctx.set_shape(self.n)
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.ctx.run(job=1)
        assert self.ctx.id.infog[0] >= 0
        self.ctx.destroy()
        assert self.ctx.destroyed

    def test_factorize(self):
        self.ctx.set_shape(self.n)
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.ctx.run(job=4)
        assert self.ctx.id.infog[0] >= 0
        self.ctx.destroy()
        assert self.ctx.destroyed

    def test_solve(self):
        self.ctx.set_shape(self.n)
        self.ctx.set_centralized_assembled(self.irn, self.jcn, self.aval)
        self.x = self.rhs.copy()
        self.ctx.set_rhs(self.x)
        self.ctx.run(job=6)
        assert self.ctx.id.infog[0] >= 0
        self.ctx.destroy()
        assert self.ctx.destroyed
        assert np.any(self.x == np.array([1., 2., 3., 4., 5.], dtype=complex))
