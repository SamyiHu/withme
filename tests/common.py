"""测试公共工具：让每个测试文件既能被 pytest 收集，也能 `python tests/test_xxx.py` 直跑。"""
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def repo_config_dir() -> str:
    return os.path.join(ROOT, "config")


def make_tmpdir() -> str:
    """系统临时目录优先；受限沙箱下 mkdtemp 目录不可写时回落到仓库内 .test_tmp。

    背景：DSH workspace-write 沙箱会拒绝写入 mkdtemp 创建的目录（Windows ACL），
    回落路径只在探测失败时启用，不改变任何测试语义。
    """
    try:
        d = tempfile.mkdtemp(prefix="withme_test_")
        probe = os.path.join(d, ".probe")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("x")
        os.remove(probe)
        return d
    except OSError:
        base = os.path.join(ROOT, ".test_tmp")
        os.makedirs(base, exist_ok=True)
        d = os.path.join(base, "t_%d_%d" % (os.getpid(), int(time.time() * 1000) % 1000000))
        os.makedirs(d, exist_ok=True)
        return d


def rmtree(path: str):
    shutil.rmtree(path, ignore_errors=True)


def copy_repo_configs(tmpdir: str, config_text: str = None, persona_text: str = None):
    """把仓库 config/ 拷进临时目录；传入文本则覆盖，便于构造非法配置。"""
    dst = os.path.join(tmpdir, "config")
    os.makedirs(dst, exist_ok=True)
    for name, text_override in (("config.yaml", config_text), ("persona.yaml", persona_text)):
        src = os.path.join(repo_config_dir(), name.replace(".yaml", ".example.yaml"))
        text = text_override if text_override is not None else open(src, encoding="utf-8").read()
        with open(os.path.join(dst, name), "w", encoding="utf-8") as f:
            f.write(text)
    return dst


def run_tests(module_globals, name: str) -> int:
    failed = []
    for fname, fn in sorted(module_globals.items()):
        if fname.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {fname}")
            except AssertionError as e:
                failed.append((fname, f"AssertionError: {e}"))
                print(f"  FAIL {fname}: {e}")
            except Exception as e:  # noqa: BLE001
                failed.append((fname, repr(e)))
                print(f"  ERROR {fname}: {e!r}")
    status = "OK" if not failed else "FAILED"
    print(f"[{name}] {status} ({len([k for k in module_globals if k.startswith('test_')])} tests)")
    return 1 if failed else 0
