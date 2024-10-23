import pytest
import glob
import os
from _frontend.res_ast import ASTGenerator
from _frontend.ast_compiler import ASTCompiler


def get_pyres_files(path: str):
    return glob.glob(os.path.join(path, "*.pyres"))


@pytest.mark.parametrize("path", get_pyres_files("examples/frontend/src_code"))
def test_fstack(path: str):
    ast = ASTGenerator().read_and_parse(path)
    res = ASTCompiler(track_time=True).compile(ast)
    assert res is not None, "failed to resovle reservoir"
