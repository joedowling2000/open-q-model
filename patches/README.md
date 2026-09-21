# Patches to KX's q-evaluation-harness

`q-evaluation-harness.diff` applies to the upstream commit recorded in
`q-evaluation-harness.base-commit`:

```bash
git clone https://github.com/KxSystems/q-evaluation-harness
cd q-evaluation-harness && git checkout $(cat ../patches/q-evaluation-harness.base-commit)
git apply ../patches/q-evaluation-harness.diff
```

Six changes, none of which touch grading:

1. `src/models/factory.py` — vLLM imported at module scope made every other
   backend unusable where vLLM will not install (aarch64). Now lazy.
2. `src/utils/hardware_profiler.py` — same import, same fix.
3. `src/evaluation/q_python_executor.py` — `del os.environ['QHOME']` raised
   KeyError at import and took down the CLI, including generation, which needs
   no q. Also: any non-x86_64 Linux was mapped to `l32`, a 32-bit path that does
   not exist on ARM64 (it is `l64arm`), and KDB-X installs `bin/q` instead.
4. `src/models/model_config.py` — an unrecognised model name falls back to
   `max_concurrent=1`, serialising the whole run; a self-hosted endpoint has no
   remote rate limit to respect. Added a `QEVAL_MAX_CONCURRENT` override.
5. `src/models/litellm_model.py` — a request that times out (litellm's 600 s
   default covers all n samples at once) comes back as n empty completions,
   which grade as failures. Added a `QEVAL_REQUEST_TIMEOUT` override.
6. `src/cli.py` — `QEVAL_ONLY_TASKS` (comma-separated task ids) generates a
   subset, to redo problems a run lost without redoing the whole benchmark.

Test execution and pass@k computation are untouched, so scores produced here are
comparable with the upstream leaderboard (modulo quantisation, which we state).
