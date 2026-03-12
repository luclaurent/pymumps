import numpy as np
import pytest

mumps = pytest.importorskip("mumps")

# test data
dataIRN = [1, 2, 4, 5, 2, 1, 5, 3, 2, 3, 1, 3]
dataJCN = [2, 3, 3, 5, 1, 1, 2, 4, 5, 2, 3, 3]
dataVAL = [3.0, -3.0, 2.0, 1.0, 3.0, 2.0, 4.0, 2.0, 6.0, -1.0, 4.0, 1.0]
dataRHS = [20.0, 24.0, 9.0, 6.0, 13.0]

TEST_CASES = [
    pytest.param("mumps._cmumps", mumps.CMumpsContext, np.complex64, id="complex64"),
    pytest.param("mumps._zmumps", mumps.ZMumpsContext, np.complex128, id="complex128"),
    pytest.param("mumps._dmumps", mumps.DMumpsContext, np.float64, id="real64"),
    pytest.param("mumps._smumps", mumps.SMumpsContext, np.float32, id="real32"),
]


@pytest.fixture(params=TEST_CASES)
def mumps_case(request):
    module_name, context_cls, dtype = request.param
    pytest.importorskip(module_name)

    case = {
        "n": 5,
        "irn": np.array(dataIRN, dtype=np.int32),
        "jcn": np.array(dataJCN, dtype=np.int32),
        "aval": np.array(dataVAL, dtype=dtype),
        "rhs": np.array(dataRHS, dtype=dtype),
        "ctx": context_cls(par=1, sym=0),
        "dtype": dtype,
    }
    case["ctx"].set_silent()

    yield case

    if not getattr(case["ctx"], "destroyed", False):
        case["ctx"].destroy()


def test_init(mumps_case):
    ctx = mumps_case["ctx"]
    ctx.set_centralized_assembled(mumps_case["irn"], mumps_case["jcn"], mumps_case["aval"])
    ctx.destroy()
    assert ctx.destroyed


def test_rhs(mumps_case):
    ctx = mumps_case["ctx"]
    ctx.set_shape(mumps_case["n"])
    ctx.set_centralized_assembled(mumps_case["irn"], mumps_case["jcn"], mumps_case["aval"])
    x = mumps_case["rhs"].copy()
    ctx.set_rhs(x)
    ctx.destroy()
    assert ctx.destroyed


def test_analyze(mumps_case):
    ctx = mumps_case["ctx"]
    ctx.set_shape(mumps_case["n"])
    ctx.set_centralized_assembled(mumps_case["irn"], mumps_case["jcn"], mumps_case["aval"])
    ctx.run(job=1)
    assert ctx.id.infog[0] >= 0
    ctx.destroy()
    assert ctx.destroyed


def test_factorize(mumps_case):
    ctx = mumps_case["ctx"]
    ctx.set_shape(mumps_case["n"])
    ctx.set_centralized_assembled(mumps_case["irn"], mumps_case["jcn"], mumps_case["aval"])
    ctx.run(job=4)
    assert ctx.id.infog[0] >= 0
    ctx.destroy()
    assert ctx.destroyed


def test_solve(mumps_case):
    ctx = mumps_case["ctx"]
    ctx.set_shape(mumps_case["n"])
    ctx.set_centralized_assembled(mumps_case["irn"], mumps_case["jcn"], mumps_case["aval"])
    x = mumps_case["rhs"].copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    assert ctx.id.infog[0] >= 0
    ctx.destroy()
    assert ctx.destroyed
    expected = np.array([1., 2., 3., 4., 5.], dtype=mumps_case["dtype"])
    assert np.allclose(x, expected)
