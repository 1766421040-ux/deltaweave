# Sources and attribution

The DeltaWeave implementation is original project code under MIT. Its general
incremental-computation mechanisms have prior art; the research documents credit
Build Systems à la Carte, Adapton, Salsa, database predicate dependencies and
other relevant systems. No algorithmic priority is claimed.

The optional base is the official
[DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B),
revision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`. The upstream
[DeepSeek-R1 repository](https://github.com/deepseek-ai/DeepSeek-R1) releases its
code and model weights under MIT and documents the Qwen2.5-Math-1.5B ancestry
under Apache 2.0. Preserve the original model licenses when obtaining or
redistributing weights. The 3.55 GB base is downloaded from the fixed official
revision and is not bundled in this repository. Local adapters are derivative
fine-tuning artifacts; they are not a new independently pretrained base model.

`data/public/pypi-catalog.json` contains selected public, file-level package
metadata facts, source URLs, observed times and SHA256 digests. Package code and
wheel binaries are not redistributed. Source metadata belongs to its upstream
publishers; inclusion is not endorsement or a claim of authorship. Synthetic
policies, withdrawals and update sequences are marked as experiment overlays,
not allegations about the actual packages.

LightRAG, Graphiti, Pathway and Debezium were studied as references; their full
systems were not benchmarked here. Pathway's core licensing is not represented
as MIT. See [research/prior-work.md](research/prior-work.md) for reading scope and
mechanism comparisons.
