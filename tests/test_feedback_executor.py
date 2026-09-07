import sys

import pytest

from reward_coupling.execute_tests import InfrastructureError, NativeExecutor, parse_code


def test_native_pass_failure_timeout_and_fresh_state():
    executor = NativeExecutor(sys.executable, timeout=0.2)
    assert executor.preflight()["security_sandbox"] is False
    assert executor.run_test("def f(x): return x+1", "assert f(2)==3")["verdict"] == 1
    assert executor.run_test("def f(x): return x", "assert f(2)==3")["verdict"] == 0
    assert executor.run_test("while True: pass", "assert True")["status"] == "timeout"
    assert executor.run_test("import os; os._exit(0)", "assert False")["verdict"] == 0
    assert executor.run_test("import os; os._exit(41)", "assert False")["verdict"] == 0
    with pytest.raises(InfrastructureError):
        executor.run_test("import os, signal; os.kill(os.getpid(), signal.SIGTERM)", "assert False")
    code = "count=0\ndef f():\n global count\n count+=1\n return count"
    assert all(executor.run_test(code, "assert f()==1")["verdict"] for _ in range(2))
    assert parse_code("```python\nx=1\n```\n```python\ny=2\n```")[1] == "ambiguous_code_blocks"
