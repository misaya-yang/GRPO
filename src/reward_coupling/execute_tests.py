"""Native per-test workers and an optional Docker backend."""

import ast
import hashlib
import json
import re
import subprocess
import time
import uuid


class InfrastructureError(RuntimeError):
    pass


_WORKER = (
    "import json,sys,os\np=json.load(sys.stdin)\nr,w=os.pipe()\npid=os.fork()\n"
    "if pid==0:\n os.close(r)\n scope={}\n"
    " try:\n  exec(compile(p['code'],'candidate','exec'),scope)\n"
    "  exec(compile(p['test'],'test','exec'),scope)\n"
    " except BaseException:\n  os._exit(42)\n"
    " os.write(w,b'P')\n os._exit(0)\n"
    "os.close(w)\nmarker=os.read(r,2)\n_,status=os.waitpid(pid,0)\n"
    "if os.WIFSIGNALED(status): os._exit(125)\n"
    "ok=marker==b'P' and os.WIFEXITED(status) and os.WEXITSTATUS(status)==0\n"
    "os._exit(41 if ok else 42)\n"
)


def make_executor(config):
    if config.get("executor", "native") == "native":
        return NativeExecutor(config.get("test_python"), config["test_timeout_seconds"])
    if config["executor"] == "docker":
        return DockerExecutor(config["sandbox_image"], config["test_timeout_seconds"])
    raise ValueError("Unknown executor")


def parse_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if len(blocks) > 1:
        return None, "ambiguous_code_blocks"
    code = blocks[0] if blocks else text.strip()
    try:
        ast.parse(code)
    except (SyntaxError, ValueError, MemoryError):
        return None, "syntax_error"
    return code, "parsed"


class DockerExecutor:
    def __init__(self, image, timeout=3.0, memory_mb=256):
        if not re.fullmatch(r"[\w./:-]+@sha256:[0-9a-f]{64}", image):
            raise ValueError("Pin sandbox image by repository digest")
        if not 0 < timeout <= 60 or memory_mb < 64:
            raise ValueError("Invalid sandbox limits")
        self.image, self.timeout, self.memory_mb = image, timeout, memory_mb

    def preflight(self):
        info = subprocess.run(
            ["docker", "image", "inspect", self.image], capture_output=True, text=True, timeout=20
        )
        if info.returncode:
            raise InfrastructureError("Docker unavailable or pinned image absent; no host fallback")
        return {"image": self.image, "isolation": "fresh_container_each_test", "network": "none"}

    def run_test(self, code, test):
        """Test source is trusted, pre-frozen data; code is an untrusted candidate.

        The harness detects premature exit; this is execution containment, not a
        proof against deliberate Python oracle introspection/reward hacking.
        """
        name = "feedback-" + uuid.uuid4().hex
        cmd = [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--name",
            name,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user=65534:65534",
            "--cpus=1",
            f"--memory={self.memory_mb}m",
            f"--memory-swap={self.memory_mb}m",
            "--pids-limit=32",
            "--ulimit",
            "nofile=64:64",
            "--ulimit",
            "fsize=1048576:1048576",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "--log-driver=none",
            "-i",
            self.image,
            "python",
            "-I",
            "-c",
            _WORKER,
        ]
        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                input=json.dumps({"code": code, "test": test}),
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout + 5,
            )
        except subprocess.TimeoutExpired:
            killed = subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=20)
            if killed.returncode:
                raise InfrastructureError("Unable to confirm timed-out container cleanup") from None
            return {"verdict": 0, "status": "timeout", "elapsed_seconds": time.monotonic() - start}
        # Runtime/resource/container failures must not become candidate zeros.
        if result.returncode not in (41, 42, 0):
            raise InfrastructureError(f"Sandbox exited {result.returncode}; pause bank")
        return {
            "verdict": int(result.returncode == 41),
            "status": "pass" if result.returncode == 41 else "candidate_failure",
            "elapsed_seconds": time.monotonic() - start,
        }

    def evaluate(self, text, task):
        code, status = parse_code(text)
        outcomes = (
            [self.run_test(code, test) for test in task["tests"]]
            if code is not None
            else [{"verdict": 0, "status": status, "elapsed_seconds": 0.0} for _ in task["tests"]]
        )
        return {
            "candidate_code_hash": hashlib.sha256((code or text).encode()).hexdigest(),
            "parser_status": status,
            "test_verdicts": [r["verdict"] for r in outcomes],
            "execution": outcomes,
        }

    def check_references(self, tasks):
        for task in tasks:
            result = self.evaluate(task["reference_code"], task)
            if not all(result["test_verdicts"]):
                raise InfrastructureError(f"Reference failed for {task['prompt_id']}")
        return {"status": "PASS", "prompts": len(tasks), **self.preflight()}


class NativeExecutor(DockerExecutor):
    """Ordinary native Python worker in a fresh temporary working directory.

    This is process separation, not a security sandbox. It is intended for the
    user-authorized dedicated research host; Docker remains an optional backend.
    """

    def __init__(self, python=None, timeout=3.0, memory_mb=256):
        import sys

        self.python = python or sys.executable
        if not 0 < timeout <= 60:
            raise ValueError("Invalid test timeout")
        self.timeout = timeout

    def preflight(self):
        probe = subprocess.run(
            [self.python, "-I", "-c", "import sys; print(sys.version)"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if probe.returncode:
            raise InfrastructureError("Native Python unavailable")
        return {
            "execution": "native_subprocess_fresh_workdir",
            "security_sandbox": False,
            "python": probe.stdout.strip(),
        }

    def run_test(self, code, test):
        import os
        import signal
        import tempfile

        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="feedback-test-") as work:
            process = subprocess.Popen(
                [self.python, "-I", "-c", _WORKER],
                cwd=work,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
                env={"PATH": os.path.dirname(self.python), "LANG": "C.UTF-8"},
            )
            try:
                process.communicate(json.dumps({"code": code, "test": test}), timeout=self.timeout)
                if process.returncode not in (0, 41, 42):
                    raise InfrastructureError(
                        f"Native worker exited {process.returncode}; inspect infrastructure"
                    )
                return {
                    "verdict": int(process.returncode == 41),
                    "status": "pass" if process.returncode == 41 else "candidate_failure",
                    "elapsed_seconds": time.monotonic() - start,
                }
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                return {
                    "verdict": 0,
                    "status": "timeout",
                    "elapsed_seconds": time.monotonic() - start,
                }
            finally:
                # Remove child processes left behind by a test, including after normal exit.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
