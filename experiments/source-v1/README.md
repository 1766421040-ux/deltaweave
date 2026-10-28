# Exploratory source snapshot boundary

This directory was copied **after** the initial unindexed v1 run, following a
read-only mode/store and receipt-contract hardening change. Its `engine.py`
SHA256 (`607953bbb7c428bd185ae0a1a86f2e6f2c00a853952872e98228202fed35a585`)
does **not** match the v1 run's recorded engine hash
(`1ecf71088fab7c24e4b7e4b86a1f68bcd13c03839fa2b1d222f8961087e3b8f7`).

Do not claim this directory is an exact replay snapshot for `updates-v1.json`.
That run is retained as exploratory evidence only. The matched indexed and
unindexed v2 runs were then repeated on the same source; `source-v2/manifest.json`
records their exact core source files. Final claims use the v3 report, its source
hashes, and the current core implementation.
