#!/usr/bin/env python3
"""Exact average of LoRA adapters (amendment 15), by rank concatenation.

For adapters i = 1..k with the same scale s = alpha/r, each update is s * B_i A_i.
Stacking A_i along the rank and B_i/k along the rank gives
s * [B_1/k ... B_k/k] [A_1; ...; A_k] = (1/k) sum_i s * B_i A_i: the mean
update, exactly. The result has rank k*r and alpha k*alpha, so the scale stays s.

    python scripts/rl/average_adapters.py --out out/rl_avg/top3 A1 A2 A3
    python scripts/rl/average_adapters.py --selftest
"""
import argparse, json, shutil, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file


def average(dirs: list[Path], out: Path) -> None:
    cfgs = [json.loads((d / "adapter_config.json").read_text()) for d in dirs]
    assert len({(c["r"], c["lora_alpha"], tuple(sorted(c["target_modules"]))) for c in cfgs}) == 1, "adapters differ"
    k, ws = len(dirs), [load_file(str(d / "adapter_model.safetensors")) for d in dirs]
    assert all(w.keys() == ws[0].keys() for w in ws)
    merged = {}
    for key in ws[0]:
        if key.endswith("lora_A.weight"):
            merged[key] = torch.cat([w[key] for w in ws], dim=0).contiguous()          # (k r, in)
        elif key.endswith("lora_B.weight"):
            merged[key] = torch.cat([w[key] / k for w in ws], dim=1).contiguous()      # (out, k r)
        else:
            raise SystemExit(f"unexpected tensor {key}")
    out.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out / "adapter_model.safetensors"))
    cfg = dict(cfgs[0]); cfg["r"] = k * cfgs[0]["r"]; cfg["lora_alpha"] = k * cfgs[0]["lora_alpha"]
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    (out / "SOURCES").write_text("\n".join(map(str, dirs)) + "\n")


def selftest() -> None:
    torch.manual_seed(0)
    r, i, o, k = 16, 64, 48, 3
    As = [torch.randn(r, i) for _ in range(k)]; Bs = [torch.randn(o, r) for _ in range(k)]
    A = torch.cat(As, 0); B = torch.cat([b / k for b in Bs], 1)
    want = sum(b @ a for a, b in zip(As, Bs)) / k
    assert torch.allclose(B @ A, want, atol=1e-5)
    print("selftest ok: concatenated adapter equals the mean update exactly")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest(); sys.exit(0)
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("dirs", nargs="+")
    a = ap.parse_args(); average([Path(d) for d in a.dirs], Path(a.out))
