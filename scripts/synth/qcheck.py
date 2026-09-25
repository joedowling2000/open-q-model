"""Run q solutions against a Python reference on generated inputs.

Shared by bank verification, distillation and RL rewards, so every stage judges
q the same way. Three details the first versions got wrong:

* Inputs are bound to globals (`A0:...;`) before the call. Inlining a large
  literal inside the protected lambda hits q's `limit` error, which reads as a
  solution failure.
* `\\P 17` before serialising. `.j.j` prints floats at the display precision
  (7 significant digits), so 137827.74 comes back as 137827.7.
* Dict keys go in as symbols when they look like identifiers, else as strings;
  a solution is judged under whichever encoding it was written for.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import tempfile

Q = "/home/joedowling/.kx/bin/q"
QENV = {"QHOME": "/home/joedowling/.kx", "QLIC": "/home/joedowling/.kx",
        "PATH": "/home/joedowling/.kx/bin:/usr/bin:/bin", "HOME": "/home/joedowling"}


def to_q(v, keymode: str = "sym") -> str:
    if isinstance(v, bool):
        return "1b" if v else "0b"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        r = repr(v)
        return r if any(c in r for c in ".en") else r + ".0"
    if isinstance(v, str):
        lit = '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
        # A one-character string is a char atom in q; the list needs enlist.
        return f"enlist {lit}" if len(v) == 1 else lit
    if isinstance(v, (list, tuple)):
        if not v:
            return "()"
        if len(v) == 1:
            return f"enlist {to_q(v[0], keymode)}"
        return "(" + ";".join(to_q(x, keymode) for x in v) + ")"
    if isinstance(v, dict):
        if not v:
            return "()!()"
        ks, vs = list(v), list(v.values())
        if keymode == "sym" and all(isinstance(k, str) and re.fullmatch(r"[A-Za-z]\w*", k)
                                    for k in ks):
            kq = "`" + "`".join(ks) if len(ks) > 1 else "enlist`" + ks[0]
        else:
            kq = to_q(ks, keymode) if len(ks) > 1 else "enlist " + to_q(ks[0], keymode)
        vq = to_q(vs, keymode) if len(vs) > 1 else "enlist " + to_q(vs[0], keymode)
        return f"({kq})!{vq}"
    raise TypeError(f"no q literal for {type(v).__name__}")


def run_q(code: str, inputs: list[dict], keymode: str = "sym",
          timeout: float = 300) -> list | None:
    """One q process for all inputs. Returns parsed outputs, {'__error__': msg}
    for a case that raised, or None if the script never produced a result."""
    lines = ["\\P 17", code, "out:();"]
    for kw in inputs:
        names = []
        for n, x in enumerate(kw.values()):
            lines.append(f"A{n}:{to_q(x, keymode)};")
            names.append(f"A{n}")
        lines.append(f'out,:enlist @[{{.j.j solve[{";".join(names)}]}};::;'
                     f'{{"__error__:",x}}];')
    lines += ['-1 "@@R@@",.j.j out;', "exit 0;"]
    with tempfile.NamedTemporaryFile("w", suffix=".q", delete=False) as fh:
        fh.write("\n".join(lines))
        path = fh.name
    try:
        # -w caps the workspace (MB): a runaway solution raises wsfull instead of
        # taking the machine down (a teacher attempt grew to 35 GB on 25 Sep).
        r = subprocess.run([Q, path, "-q", "-w", "4000"], capture_output=True, text=True,
                           timeout=timeout, env=QENV)
        for line in (r.stdout + r.stderr).splitlines():
            if line.startswith("@@R@@"):
                return [{"__error__": x} if x.startswith("__error__") else json.loads(x)
                        for x in json.loads(line[5:])]
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return None
    finally:
        os.unlink(path)


def agrees(py, qv) -> bool:
    """Python result vs q's JSON: numbers within 1e-6 relative, dicts unordered,
    a Python string may come back as a list of chars, booleans as 0/1."""
    if isinstance(qv, dict) and "__error__" in qv:
        return False
    if isinstance(py, bool) or isinstance(qv, bool):
        return (py in (0, 1, True, False) and qv in (0, 1, True, False)
                and bool(py) == bool(qv))
    if isinstance(py, (int, float)) and isinstance(qv, (int, float)):
        return abs(py - qv) <= 1e-6 * max(1, abs(py))
    if isinstance(py, dict) and isinstance(qv, dict):
        return (set(map(str, py)) == set(map(str, qv))
                and all(agrees(py[k], qv.get(k, qv.get(str(k)))) for k in py))
    if isinstance(py, (list, tuple)) and isinstance(qv, list):
        return len(py) == len(qv) and all(agrees(a, b) for a, b in zip(py, qv))
    if isinstance(py, str) and isinstance(qv, list):
        return py == "".join(map(str, qv))
    if isinstance(py, (list, tuple)) and len(py) == 1 and not isinstance(qv, list):
        return agrees(py[0], qv)
    if isinstance(py, str) and isinstance(qv, str):
        return py == qv
    if py is None:
        return qv is None or qv == [] or qv == ""
    return False


def arg_order(rec: dict) -> list[str]:
    if "signature" in rec:
        return re.findall(r"\w+", rec["signature"].split("[", 1)[1])
    return rec["argnames"]


def gen_inputs(python_src: str, order: list[str], seeds) -> tuple[list[dict], list]:
    """Draw inputs from the problem's own generator; return (inputs, expected)."""
    g: dict = {}
    exec(python_src, g)
    inputs, expected = [], []
    for s in seeds:
        x = g["gen_input"](random.Random(s))
        if not isinstance(x, dict):
            x = {order[0]: x} if len(order) == 1 else dict(zip(order, x))
        x = {k: x[k] for k in order}
        inputs.append(x)
        expected.append(json.loads(json.dumps(g["solve"](**json.loads(json.dumps(x))))))
    return inputs, expected


def dominant_share(expected: list) -> float:
    """Fraction of cases sharing the most common answer — the discriminating check."""
    keys = [json.dumps(e, sort_keys=True) for e in expected]
    return max(keys.count(k) for k in set(keys)) / len(keys)


def check(code: str, python_src: str, order: list[str], seeds) -> dict:
    """Verify one q solution. Tries symbol-keyed dicts first, then string keys."""
    inputs, expected = gen_inputs(python_src, order, seeds)
    best = None
    for keymode in ("sym", "str"):
        got = run_q(code, inputs, keymode)
        if got is None:
            continue
        bad = [i for i, (e, q) in enumerate(zip(expected, got)) if not agrees(e, q)]
        if best is None or len(bad) < best["n_fail"]:
            best = {"keymode": keymode, "n_fail": len(bad),
                    "first_fail": ({"input": inputs[bad[0]], "expected": expected[bad[0]],
                                    "got": got[bad[0]]} if bad else None)}
        if not bad:
            break
    if best is None:
        return {"ok": False, "reason": "did not load", "n_cases": len(inputs)}
    return {"ok": best["n_fail"] == 0, "n_cases": len(inputs),
            "dominant_share": dominant_share(expected), **best}
