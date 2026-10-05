"""极简 pytest 垫片（仅本沙箱验证用，非交付物）。

支持：mark.skip、raises、fixture、appox、test 函数自动发现执行。
fixture 仅处理 tests/conftest.py 中的 client/pool_initialized（在本沙箱
无 Postgres，集成测试会被 skip）。
"""
from __future__ import annotations

import math
import re as _re
import sys
import traceback
import types as _types


class Approx:
    def __init__(self, expected, rel=1e-9, abs=1e-12):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, other):
        if self.expected is None or other is None:
            return self.expected is None and other is None
        tol = max(self.abs, self.rel * abs(self.expected))
        return abs(other - self.expected) <= tol

    def __repr__(self):
        return f"approx({self.expected})"


def approx(expected, rel=1e-9, abs=1e-12):
    return Approx(expected, rel, abs)


class _Raises:
    def __init__(self, exc):
        self.exc = exc
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, etype, e, tb):
        if etype is None:
            raise AssertionError(f"DID_NOT_RAISE {self.exc}")
        assert issubclass(etype, self.exc), f"expected {self.exc}, got {etype}"
        self.value = e
        return True


def raises(exc):
    return _Raises(exc)


class Mark:
    @staticmethod
    def skip(reason=""):
        raise _Skip(reason)


class _Skip(Exception):
    pass


mark = _types.SimpleNamespace(skip=lambda reason="": _skip_marker(reason))


def _skip_marker(reason):
    return _types.SimpleNamespace(reason=reason)


# fixture 装饰器：本垫片下仅记录，运行时在 runner 里特判 conftest
def fixture(*args, **kwargs):
    def deco(fn):
        fn._is_fixture = True
        return fn

    if args and callable(args[0]):
        args[0]._is_fixture = True
        return args[0]
    return deco


def _run_file(path):
    import importlib.util

    # 让 conftest 可 import
    sys.path.insert(0, path.rsplit("/", 1)[0])
    spec = importlib.util.spec_from_file_location("conftest", path.rsplit("/", 1)[0] + "/conftest.py")
    conftest = None
    try:
        conftest_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conftest_mod)
        conftest = conftest_mod
    except Exception:  # noqa: BLE001
        conftest = None

    spec2 = importlib.util.spec_from_file_location("mod_under_test", path)
    mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod)

    passed = failed = skipped = 0
    failures = []
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        fn = getattr(mod, name)
        if not callable:
            continue
        try:
            # 收集夹具参数
            need = list(fn.__code__.co_varnames[: fn.__code__.co_argcount])
            kwargs = {}
            for arg in need:
                if arg == "client":
                    # 沙箱无 Postgres：跳过
                    raise _Skip("no postgres in sandbox")
                if conftest and hasattr(conftest, arg):
                    fx = getattr(conftest, arg)
                    if hasattr(fx, "_is_fixture"):
                        kwargs[arg] = next(fx())
            if getattr(fn, "_pytest_skip", False):
                raise _Skip(fn._pytest_skip_reason)
            fn(**kwargs)
            passed += 1
            print(f"PASS {name}")
        except _Skip as s:
            skipped += 1
            print(f"SKIP {name}: {s}")
        except Exception:  # noqa: BLE001
            failed += 1
            failures.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()
    return passed, failed, skipped


def main():
    tp = tf = ts = 0
    for p in sys.argv[1:]:
        p0, f0, s0 = _run_file(p)
        tp += p0
        tf += f0
        ts += s0
    print(f"\n{tp} passed, {tf} failed, {ts} skipped")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
