"""仅用于本沙箱（无 numpy、无网络）验证 app/did.py 的最小 numpy 垫片。

非交付物：Docker 镜像里安装真正的 numpy 2.2.1。
用 PYTHONPATH=/workspace/scripts python -m pytest ... 时优先命中本文件。
覆盖 app/did.py 实际用到的子集：稠密嵌套数组、广播四则、花式/布尔索引、
matmul、高斯消元 solve/inv、随机数。
"""
from __future__ import annotations

import math as _math
import random as _random

nan = float("nan")
inf = float("inf")


class LinAlgError(Exception):
    pass


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def shape(x):
    s = []
    cur = x
    while isinstance(cur, list):
        s.append(len(cur))
        cur = cur[0] if cur else None
    return tuple(s)


def deepcopy(x):
    if isinstance(x, list):
        return [deepcopy(v) for v in x]
    return x


def flatten(x):
    out = []

    def rec(v):
        if isinstance(v, list):
            for z in v:
                rec(z)
        else:
            out.append(v)

    rec(x)
    return out


def unflatten(flat, shp):
    if len(shp) == 1:
        return list(flat)
    step = 1
    for s in shp[1:]:
        step *= s
    return [unflatten(flat[i * step:(i + 1) * step], shp[1:]) for i in range(shp[0])]


def cast(x, dtype):
    if isinstance(x, list):
        return [cast(v, dtype) for v in x]
    if x is None:
        return False if dtype is bool else nan
    if dtype is bool:
        return bool(x)
    if dtype is object:
        return x
    if dtype is int:
        return int(x)
    return float(x)


def make(shp, fill, dtype=float):
    if len(shp) == 1:
        if dtype is bool:
            return [bool(fill)] * shp[0]
        if dtype is object:
            return [deepcopy(fill) for _ in range(shp[0])]
        if dtype is int:
            return [int(fill)] * shp[0]
        return [float(fill)] * shp[0]
    return [make(shp[1:], fill, dtype) for _ in range(shp[0])]


def broadcast(a, b, op):
    """逐元素 op，支持右对齐广播（b 维度更少或某维长度为 1）。"""
    if isinstance(a, list) or isinstance(b, list):
        la, lb = isinstance(a, list), isinstance(b, list)
        if la and lb:
            if len(a) == len(b):
                return [broadcast(x, y, op) for x, y in zip(a, b)]
            if len(a) == 1:
                return [broadcast(a[0], y, op) for y in b]
            if len(b) == 1:
                return [broadcast(x, b[0], op) for x in a]
            raise ValueError("cannot broadcast shapes")
        if la:
            return [broadcast(x, b, op) for x in a]
        return [broadcast(a, y, op) for y in b]
    return op(a, b)


# --------------------------------------------------------------------------- #
# ndarray
# --------------------------------------------------------------------------- #
class ndarray:
    def __init__(self, data, dtype=float):
        self.data = data
        self.dtype = dtype

    # ---- 基础 ----
    @property
    def shape(self):
        return shape(self.data)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def copy(self):
        return ndarray(deepcopy(self.data), self.dtype)

    def astype(self, dtype):
        return ndarray(cast(self.data, dtype), dtype)

    def tolist(self):
        return deepcopy(self.data)

    def reshape(self, *shp):
        if len(shp) == 1 and isinstance(shp[0], (tuple, list)):
            shp = tuple(shp[0])
        flat = flatten(self.data)
        if -1 in shp:
            known = 1
            for s in shp:
                if s != -1:
                    known *= s
            shp = tuple(self.size // known if s == -1 else s for s in shp)
        return ndarray(unflatten(flat, shp), self.dtype)

    # ---- 索引 ----
    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        val = _get(self.data, idx)
        if isinstance(val, list):
            return ndarray(val, self.dtype)
        return val

    def __setitem__(self, idx, value):
        if not isinstance(idx, tuple):
            idx = (idx,)
        v = value.data if isinstance(value, ndarray) else value
        _set(self.data, idx, v, self.dtype)

    # ---- 运算 ----
    def _b(self, other, op):
        b = other.data if isinstance(other, ndarray) else other
        return ndarray(broadcast(self.data, b, op), self.dtype if self.dtype is not bool else float)

    def __add__(self, o):
        return self._b(o, lambda a, b: a + b)

    def __radd__(self, o):
        return self._b(o, lambda a, b: b + a)

    def __sub__(self, o):
        return self._b(o, lambda a, b: a - b)

    def __rsub__(self, o):
        return self._b(o, lambda a, b: b - a)

    def __mul__(self, o):
        return self._b(o, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self._b(o, lambda a, b: a / b)

    def __neg__(self):
        return ndarray(broadcast(self.data, 0.0, lambda a, _: -a), self.dtype)

    def __matmul__(self, o):
        return matmul(self, o)

    def __eq__(self, o):
        b = o.data if isinstance(o, ndarray) else o
        return ndarray(broadcast(self.data, b, lambda a, b: bool(a == b)), bool)

    def __ne__(self, o):
        b = o.data if isinstance(o, ndarray) else o
        return ndarray(broadcast(self.data, b, lambda a, b: bool(a != b)), bool)

    def __lt__(self, o):
        b = o.data if isinstance(o, ndarray) else o
        return ndarray(broadcast(self.data, b, lambda a, b: a < b), bool)

    def __le__(self, o):
        b = o.data if isinstance(o, ndarray) else o
        return ndarray(broadcast(self.data, b, lambda a, b: a <= b), bool)

    def __gt__(self, o):
        b = o.data if isinstance(o, ndarray) else o
        return ndarray(broadcast(self.data, b, lambda a, b: a > b), bool)

    def __ge__(self, o):
        b = o.data if isinstance(o, ndarray) else o
        return ndarray(broadcast(self.data, b, lambda a, b: a >= b), bool)

    def __and__(self, o):
        b = o.data if isinstance(o, ndarray) else o
        return ndarray(broadcast(self.data, b, lambda a, b: bool(a) and bool(b)), bool)

    def __or__(self, o):
        b = o.data if isinstance(o, ndarray) else o
        return ndarray(broadcast(self.data, b, lambda a, b: bool(a) or bool(b)), bool)

    def __invert__(self):
        return ndarray(broadcast(self.data, False, lambda a, _: not bool(a)), bool)

    def __hash__(self):
        return id(self)

    # ---- 归约 ----
    def sum(self, axis=None):
        if axis is None:
            return sum(flatten(self.data))
        if axis == 0:
            if not isinstance(self.data[0], list):
                return sum(self.data)
            cols = len(self.data[0])
            return ndarray(
                [sum(self.data[i][j] for i in range(len(self.data))) for j in range(cols)],
                self.dtype,
            )
        return ndarray([_sum_axis(x, axis - 1) for x in self.data], self.dtype)

    def all(self, axis=None):
        if axis is None:
            return all(flatten(cast(self.data, bool)))
        if axis == 0:
            if not isinstance(self.data[0], list):
                return ndarray([all(cast(self.data, bool))], bool)
            cols = len(self.data[0])
            return ndarray(
                [all(bool(self.data[i][j]) for i in range(len(self.data))) for j in range(cols)],
                bool,
            )
        return ndarray([_all_axis(x, axis - 1) for x in self.data], bool)

    def any(self, axis=None):
        if axis is None:
            return any(flatten(cast(self.data, bool)))
        if axis == 0:
            if not isinstance(self.data[0], list):
                return ndarray([any(cast(self.data, bool))], bool)
            cols = len(self.data[0])
            return ndarray(
                [any(bool(self.data[i][j]) for i in range(len(self.data))) for j in range(cols)],
                bool,
            )
        return ndarray([_any_axis(x, axis - 1) for x in self.data], bool)

    @property
    def T(self):
        a = self.data
        return ndarray(
            [[a[i][j] for i in range(len(a))] for j in range(len(a[0]))], self.dtype
        )

    def __float__(self):
        return float(flatten(self.data)[0])

    def __iter__(self):
        for v in self.data:
            yield ndarray(v, self.dtype) if isinstance(v, list) else v


def _sum_axis(x, axis):
    if axis == 0:
        if not isinstance(x[0], list):
            return sum(x)
        cols = len(x[0])
        return [sum(x[i][j] for i in range(len(x))) for j in range(cols)]
    return [_sum_axis(v, axis - 1) for v in x]


def _all_axis(x, axis):
    if axis == 0:
        if not isinstance(x[0], list):
            return all(bool(v) for v in x)
        cols = len(x[0])
        return [all(bool(x[i][j]) for i in range(len(x))) for j in range(cols)]
    return [_all_axis(v, axis - 1) for v in x]


def _any_axis(x, axis):
    if axis == 0:
        if not isinstance(x[0], list):
            return any(bool(v) for v in x)
        cols = len(x[0])
        return [any(bool(x[i][j]) for i in range(len(x))) for j in range(cols)]
    return [_any_axis(v, axis - 1) for v in x]


def _norm_idx(i):
    if isinstance(i, ndarray):
        i = i.data
    return i


def _get(data, idx):
    if not idx:
        return data
    i = _norm_idx(idx[0])
    rest = idx[1:]
    if isinstance(i, list):
        if i and isinstance(i[0], bool):
            return [_get(data[k], rest) for k, flag in enumerate(i) if flag]
        return [_get(data[k], rest) for k in i]
    if isinstance(i, slice):
        return [_get(data[k], rest) for k in range(*i.indices(len(data)))]
    return _get(data[i], rest)


def _set(data, idx, value, dtype):
    """支持：整数标量索引、花式整数列表、一维布尔掩码、切片（可多级组合）；右值可广播。"""
    if not idx:
        if isinstance(value, list):
            data[:] = cast(deepcopy(value), dtype)
        else:
            raise RuntimeError("cannot assign scalar to leaf without index")
        return
    i = _norm_idx(idx[0])
    rest = idx[1:]
    if isinstance(i, list):
        bool_mask = bool(i) and isinstance(i[0], bool)
        keys = [k for k, flag in enumerate(i) if flag] if bool_mask else i
        if not rest:
            if isinstance(value, list):
                for k, v in zip(keys, value):
                    data[k] = cast(v, dtype) if isinstance(v, list) else _scalar(v, dtype)
            else:
                for k in keys:
                    data[k] = _scalar(value, dtype)
                    if isinstance(data[k], list) and not isinstance(value, list):
                        # 不应发生（形状由调用方保证）
                        raise RuntimeError("shape mismatch in assignment")
        else:
            for k in keys:
                _set(data[k], rest, value, dtype)
        return
    if isinstance(i, slice):
        keys = list(range(*i.indices(len(data))))
        if not rest:
            if isinstance(value, list):
                for k, v in zip(keys, value):
                    data[k] = cast(v, dtype) if isinstance(v, list) else _scalar(v, dtype)
            else:
                for k in keys:
                    data[k] = _scalar(value, dtype)
        else:
            for k in keys:
                _set(data[k], rest, value, dtype)
        return
    if rest:
        _set(data[i], rest, value, dtype)
    else:
        data[i] = cast(deepcopy(value), dtype) if isinstance(value, list) else _scalar(value, dtype)


def _scalar(v, dtype):
    if dtype is bool:
        return bool(v)
    if dtype is object:
        return v
    if dtype is int:
        return int(v)
    return float(v)


# --------------------------------------------------------------------------- #
# 构造与函数
# --------------------------------------------------------------------------- #
def array(x, dtype=None):
    dt = dtype if dtype is not None else (object if _needs_object(x) else float)
    return ndarray(cast(deepcopy(x), dt), dt)


asarray = array


def _needs_object(x):
    f = flatten(x)
    return bool(f) and not isinstance(f[0], (int, float, bool))


def zeros(shp, dtype=float):
    if isinstance(shp, int):
        shp = (shp,)
    return ndarray(make(shp, 0, dtype), dtype)


def ones(shp, dtype=float):
    if isinstance(shp, int):
        shp = (shp,)
    return ndarray(make(shp, 1, dtype), dtype)


def full(shp, fill, dtype=None):
    if isinstance(shp, int):
        shp = (shp,)
    dt = dtype if dtype is not None else (object if not isinstance(fill, (int, float, bool)) else float)
    return ndarray(make(shp, fill, dt), dt)


def zeros_like(a):
    return zeros(a.shape, a.dtype)


def sqrt(x):
    if isinstance(x, ndarray):
        return ndarray(broadcast(x.data, 0.0, lambda a, _: _math.sqrt(a)), float)
    return _math.sqrt(x)


def abs(x):
    if isinstance(x, ndarray):
        return ndarray(broadcast(x.data, 0.0, lambda a, _: -a if a < 0 else a), float)
    return -x if x < 0 else x


def max(x, axis=None):
    if isinstance(x, ndarray):
        if axis is None:
            return builtins_max(flatten(x.data))
        raise NotImplementedError
    return builtins_max(x)


def mean(x):
    if isinstance(x, ndarray):
        f = flatten(x.data)
        return sum(f) / len(f)
    return sum(x) / len(x)


def min(x):
    if isinstance(x, ndarray):
        return builtins_min(flatten(x.data))
    return builtins_min(x)


import builtins as _builtins

builtins_max = _builtins.max


def clip(x, lo, hi):
    upper = inf if hi is None else hi
    lower = -inf if lo is None else lo
    if isinstance(x, ndarray):
        return ndarray(
            broadcast(x.data, 0.0, lambda a, _: builtins_max(lower, builtins_min(a, upper))),
            float,
        )
    return builtins_max(lower, builtins_min(x, upper))


builtins_min = _builtins.min


def isfinite(x):
    if isinstance(x, ndarray):
        return ndarray(broadcast(x.data, 0.0, _isfin), bool)
    return _isfin(x, None)


def _isfin(v, _):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (inf, -inf)


def diag(M):
    return ndarray([M.data[i][i] for i in range(len(M.data))], float)


def ix_(*idxs):
    result = []
    for gi, idx in enumerate(idxs):
        vals = list(idx.data if isinstance(idx, ndarray) else idx)
        shp = [1] * len(idxs)
        shp[gi] = len(vals)
        result.append(ndarray(unflatten(vals, tuple(shp)), int))
    return tuple(result)


def outer(a, b):
    a = a.data if isinstance(a, ndarray) else list(a)
    b = b.data if isinstance(b, ndarray) else list(b)
    return ndarray([[x * y for y in b] for x in a], float)


def vstack(arrs):
    rows = []
    for a in arrs:
        d = a.data if isinstance(a, ndarray) else a
        rows.extend(d if isinstance(d[0], list) else [d])
    return ndarray(rows, float)


def concatenate(arrs, axis=0):
    out = []
    dt = float
    for a in arrs:
        if isinstance(a, ndarray):
            dt = a.dtype
        d = a.data if isinstance(a, ndarray) else a
        out.extend(d)
    return ndarray(out, dt)


def add_at(a, idx, b):
    idx = idx.data if isinstance(idx, ndarray) else idx
    b = b.data if isinstance(b, ndarray) else b
    if isinstance(b, list):
        for k, v in zip(idx, b):
            a.data[k] += v
    else:
        for k in idx:
            a.data[k] += b


def flatnonzero(a):
    d = a.data if isinstance(a, ndarray) else a
    return ndarray([i for i, v in enumerate(d) if v], int)


def where(cond):
    d = cond.data if isinstance(cond, ndarray) else cond
    return (ndarray([i for i, v in enumerate(d) if v], int),)


def argwhere(a):
    d = a.data
    out = []

    def rec(pre, x):
        if x and isinstance(x[0], list):
            for i, v in enumerate(x):
                rec(pre + [i], v)
        else:
            for i, v in enumerate(x):
                if v:
                    out.append(pre + [i])

    rec([], d)
    return ndarray(out, int)


def unique(a):
    d = list(a.data if isinstance(a, ndarray) else a)
    seen = []
    for v in d:
        if v not in seen:
            seen.append(v)
    return ndarray(seen, a.dtype if isinstance(a, ndarray) else object)


def matmul(A, B):
    a = A.data if isinstance(A, ndarray) else A
    b = B.data if isinstance(B, ndarray) else B
    a2 = isinstance(a[0], list)
    b2 = isinstance(b[0], list)
    if not a2 and not b2:
        return sum(x * y for x, y in zip(a, b))
    if a2 and b2:
        n, m, p = len(a), len(b), len(b[0])
        out = [[0.0] * p for _ in range(n)]
        for i in range(n):
            ai = a[i]
            oi = out[i]
            for k in range(m):
                aik = ai[k]
                if aik:
                    bk = b[k]
                    for j in range(p):
                        oi[j] += aik * bk[j]
        return ndarray(out, float)
    if not a2:  # 1d @ 2d
        p = len(b[0])
        out = [0.0] * p
        for k, av in enumerate(a):
            if av:
                bk = b[k]
                for j in range(p):
                    out[j] += av * bk[j]
        return ndarray(out, float)
    # 2d @ 1d
    out = [0.0] * len(a)
    for i, ai in enumerate(a):
        s = 0.0
        for k in range(len(b)):
            s += ai[k] * b[k]
        out[i] = s
    return ndarray(out, float)


class linalg:
    LinAlgError = LinAlgError

    @staticmethod
    def solve(A, b):
        return ndarray(_gauss(A.data, b.data if isinstance(b, ndarray) else b, False), float)

    @staticmethod
    def inv(A):
        return ndarray(_gauss(A.data, None, True), float)


def _gauss(A, b, invert):
    n = len(A)
    M = [row[:] for row in A]
    RHS = (
        [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        if invert
        else [[float(v)] for v in b]
    )
    for col in range(n):
        piv = _builtins.max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-13:
            raise LinAlgError("singular")
        M[col], M[piv] = M[piv], M[col]
        RHS[col], RHS[piv] = RHS[piv], RHS[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        RHS[col] = [v / pv for v in RHS[col]]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col]
            if f:
                M[r] = [x - f * y for x, y in zip(M[r], M[col])]
                RHS[r] = [x - f * y for x, y in zip(RHS[r], RHS[col])]
    if invert:
        return RHS
    return [row[0] for row in RHS]


class _Add:
    @staticmethod
    def at(a, idx, b):
        add_at(a, idx, b)


add = _Add()


class _RNG:
    def __init__(self, seed):
        self.r = _random.Random(seed)
        self._spare = None

    def normal(self, loc=0.0, scale=1.0):
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return loc + scale * v
        while True:
            u = self.r.random() * 2 - 1
            vv = self.r.random() * 2 - 1
            s = u * u + vv * vv
            if 0 < s < 1:
                break
        m = ((-2 * _math.log(s)) / s) ** 0.5
        self._spare = vv * m
        return loc + scale * (u * m)

    def shuffle(self, xs):
        d = xs.data if isinstance(xs, ndarray) else xs
        self.r.shuffle(d)


def default_rng(seed=None):
    return _RNG(seed)


class _RandomNS:
    default_rng = staticmethod(default_rng)


random = _RandomNS()
