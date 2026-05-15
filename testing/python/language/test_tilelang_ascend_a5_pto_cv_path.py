import tilelang
import tilelang.language as T


def _lower_source(func, platform="A5"):
    return tilelang.lower(func, target="pto", platform=platform).kernel_source


def _ub_to_l1_kernel(threads=None):
    @T.prim_func
    def main(A: T.Tensor((128, 128), "float16")):
        with T.Kernel(1, threads=threads, is_npu=True) as (cid, vid):
            ub = T.alloc_ub((64, 128), "float16")
            l1 = T.alloc_L1((64, 128), "float16")
            l0a = T.alloc_L0A((64, 128), "float16")

            with T.Scope("V"):
                T.copy(A[vid * 64, 0], ub)
                T.copy(ub, l1)

            with T.Scope("C"):
                T.copy(l1, l0a)

    return main


def _l0c_to_ub_kernel():
    @T.prim_func
    def main(
        A: T.Tensor((64, 64), "float16"),
        B: T.Tensor((64, 64), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_l1 = T.alloc_L1((64, 64), "float16")
            b_l1 = T.alloc_L1((64, 64), "float16")
            acc_l0c = T.alloc_L0C((64, 64), "float")
            acc_ub = T.alloc_ub((64, 64), "float")

            with T.Scope("C"):
                T.copy(A, a_l1)
                T.copy(B, b_l1)
                T.gemm_v0(a_l1, b_l1, acc_l0c, init=True)
                T.copy(acc_l0c, acc_ub)

    return main


def test_a5_pto_keeps_ub_to_l1_direct():
    source = _lower_source(_ub_to_l1_kernel())

    assert "PTO_PLATFORM_A5" in source
    assert "tl::ascend_pto::copy_ub_to_l1(" in source
    assert "tl::ascend_pto::copy_ub_to_gm" not in source
    assert "tl::ascend_pto::copy_gm_to_l1" not in source
    assert "auto_gm_indices" not in source


def test_a5_pto_keeps_l0c_to_ub_direct():
    source = _lower_source(_l0c_to_ub_kernel())

    assert "PTO_PLATFORM_A5" in source
    assert "tl::ascend_pto::copy_l0c_to_ub(" in source
    assert "tl::ascend_pto::copy_l0c_to_gm" not in source
    assert "tl::ascend_pto::copy_gm_to_ub" not in source
    assert "auto_gm_indices" not in source


def test_a5_pto_threads_one_still_uses_two_vector_subblocks():
    source = _lower_source(_ub_to_l1_kernel(threads=1))

    assert "get_subblockid()" in source
    assert "#if defined(__DAV_C310_VEC__)" in source
    assert " / 2;" in source


def test_non_a5_pto_still_reduces_ub_to_l1_through_gm():
    source = _lower_source(_ub_to_l1_kernel(), platform="A3")

    assert "PTO_PLATFORM_A5" not in source
    assert "tl::ascend_pto::copy_ub_to_l1(" not in source
    assert "tl::ascend_pto::copy_ub_to_gm" in source
    assert "tl::ascend_pto::copy_gm_to_l1" in source


def test_non_a5_pto_still_reduces_l0c_to_ub_through_gm():
    source = _lower_source(_l0c_to_ub_kernel(), platform="A3")

    assert "PTO_PLATFORM_A5" not in source
    assert "tl::ascend_pto::copy_l0c_to_ub(" not in source
    assert "tl::ascend_pto::copy_l0c_to_gm" in source
    assert "tl::ascend_pto::copy_gm_to_ub" in source
