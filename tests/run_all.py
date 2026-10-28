"""运行 tests/ 下全部测试并生成 reports/test_report.md。"""
import glob
import importlib.util
import io
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_module(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py")))
    total = passed = 0
    report_lines = ["# 测试报告", "",
                    "运行命令：`python tests/run_all.py`", ""]
    overall_ok = True
    for path in files:
        mod = load_module(path)
        buf = io.StringIO()
        sys.stdout = buf
        t0 = time.time()
        try:
            code = 1
            if hasattr(mod, "run_tests"):
                pass
            # 每个测试文件自带 main；这里直接内联执行其 test_ 函数
            failed = []
            n = 0
            for fname, fn in sorted(vars(mod).items()):
                if fname.startswith("test_") and callable(fn):
                    n += 1
                    try:
                        fn()
                        print(f"  PASS {fname}")
                    except Exception as e:  # noqa: BLE001
                        failed.append((fname, repr(e)))
                        print(f"  FAIL {fname}: {e!r}")
            code = 1 if failed else 0
        finally:
            sys.stdout = sys.__stdout__
        dt = (time.time() - t0) * 1000
        ok = code == 0
        overall_ok = overall_ok and ok
        total += n
        passed += n - len(failed) if not failed else n - len(failed)
        report_lines.append(f"## {os.path.basename(path)} — {'PASS' if ok else 'FAIL'}（{n} 项，{dt:.0f}ms）")
        report_lines.append("```")
        report_lines.append(buf.getvalue().rstrip())
        report_lines.append("```")
        report_lines.append("")
        print(f"{os.path.basename(path):32s} {'PASS' if ok else 'FAIL'}  {n} tests  {dt:.0f}ms")
        if failed:  # 失败明细同时回显 stdout：CI 日志里直接可见，不用翻报告文件
            for fname, err in failed:
                print(f"    ↳ {fname}: {err}")
    failed_total = total - passed
    report_lines.insert(3, f"**结果：{passed}/{total} 通过，{failed_total} 失败**")
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "test_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n总计 {passed}/{total} 通过；报告已写入 reports/test_report.md")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
