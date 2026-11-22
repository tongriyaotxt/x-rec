# x-rec

**An unofficial, mechanism-level reimplementation of X-Rec** 鈥?recommendation as conditional generation in
the continuous item-embedding space, via Riemannian flow matching, anchor conditioning and a
late-interaction diffusion Transformer.

[![ci](https://github.com/tongriyaotxt/x-rec/actions/workflows/ci.yml/badge.svg)](https://github.com/tongriyaotxt/x-rec/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-2f6f4f)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-4b5563)](pyproject.toml)

**English** 路 [涓枃](README.zh-CN.md) 路 [research report (涓枃, 闀挎枃)](docs/x-rec-research-zh.md)

> **Unofficial.** Not affiliated with, endorsed by, or reviewed by ByteDance, TikTok or the paper's
> authors. Written from the paper's equations; no vendor code is included.
> **Not a reproduction.** The fixture is synthetic and small on purpose; the numbers below are not
> comparable to the paper's production results. See [Limitations](#limitations).

---

## Why this exists

X-Rec replaces the retrieval head with a generative one: instead of pooling a user into one vector, it
learns `p(x | c)` over item embeddings and *generates* K trigger vectors, then hits an ANN index with
all of them. The paper reports 10.76% Recall@20 at 575.7 QPS against a discrete-ID baseline at 10.23%
and 166.4 QPS.

Two things a reader cannot check from outside: **are the equations implemented correctly**, and **does
the ordering survive an independent, equal-budget comparison**. This repository answers both, and the
answers are not the ones the abstract suggests:

- the RFM math reproduces to 3.0e-4, late interaction is genuinely cheaper, multi-trigger generation
  covers multi-mode interests;
- but on a 6k-item fixture **every learned variant loses to "return the 20 most popular items"**, and
  the SID-AR control wins by *collapsing* into a popularity prior rather than by modelling better.

That is what the rest of this file documents, with the numbers and the code that produced them.

## Results at a glance

3 seeds, `n_eval=4096`, candidate budget = 20 in every row (`scripts/run_experiments.py`):

| variant | Recall@20 (20脳1) | Recall@20 (1脳20) | mode coverage | trigger cosine | ms/request |
|---|---|---|---|---|---|
| **X-Rec** | **0.88%** [0.85鈥?.90] | **1.13%** [0.98鈥?.34] | 99.0% | 0.197 | 0.34 |
| w/o anchor conditioning | 0.81% [0.56鈥?.15] | 0.83% | 99.3% | 0.163 | 0.33 |
| w/o RFM (Euclidean rectified flow) | 0.68% [0.44鈥?.88] | 0.85% | 99.3% | 0.162 | 0.33 |
| w/o pretraining | 0.44% [0.42鈥?.46] | 0.44% | 97.1% | 0.249 | 0.32 |
| legacy random-prefix pretraining | 0.91% [0.85鈥?.98] | 1.18% | 99.7% | 0.234 | 0.32 |
| U2I (single vector) | 0.02% | 0.71% | 91.0% | 1.000 | 0.04 |
| U2I-K (ComiRec-SA, K=20) | 0.07% | 0.90% | 96.2% | 0.972 | 0.04 |
| SID-AR (4脳256 RQ, entity model) | 2.80% [2.47鈥?.03] | 1.58% | 100% | 0.146 | 0.10 |

**Survives 3 seeds:** pretraining is the dominant factor (0.88% 鈫?0.44%, the only non-overlapping
interval); X-Rec beats both U2I controls at equal budget; multi-interest U2I is only *superficially*
diverse (cosine 0.972), exactly as the paper argues.

**Does not survive:** anchor conditioning (0.81 卤 0.30) and RFM (0.68 卤 0.22) sit inside seed noise; the
Fig. 3(a) all-prefix mask gives no measurable gain over the legacy random-prefix recipe.

## The part worth reading: calibration

Same 20-candidate budget, same test set:

| reference | Recall@20 | what it means |
|---|---|---|
| uniform | 0.33% | floor |
| **popularity@20** | **3.93%** | the 20 most frequent training items, same list for every request |
| oracle-mode popularity@20 | **14.53%** | top-20 *inside the user's true interest mode* 鈥?still ignores everything else |

Every learned variant is below the trivial baseline. X-Rec's 99% mode coverage says the triggers land in
the right interest cluster; it says nothing about ranking inside it (0.88% vs a 14.53% within-cluster
ceiling). Mechanically: in this fixture popularity is encoded only in an item's *radius from the cluster
centre*, and the RFM objective regresses the target's direction 鈥?the model learns **which** cluster, not
**how deep** inside it.

`scripts/diagnose_prior.py` shows what SID-AR's apparent win actually is:

| variant | Recall@20 | mean popularity percentile | share of returned items in global top-20 | distinct items returned |
|---|---|---|---|---|
| X-Rec | 1.17% | 0.746 | 5.7% | **3680** |
| SID-AR | 1.46% | 0.986 | 56.8% | **23** |

SID-AR returns **23 distinct items across 4096 requests** 鈥?it is reciting the marginal, which is why its
recall lands just under popularity. X-Rec spreads over 3680 items: it does the personalisation, and still
places candidates badly.

This is a statement about *this fixture and this training budget* (鈮?M params, 200 + 300 steps), not about
SID-AR at billion-item scale, where a SID is shared by many items and cannot be recited. It is the reason
this repo does not claim "the three ideas work".

## Quickstart

```bash
pip install -r requirements.txt          # torch + numpy + scikit-learn; run from the repo root

python xrec_ref/rfm.py --selftest        # 6 numerical identities, no training
pytest -q                                # 41 tests: math, ANN, log parsing, serving, HTTP

# train + ablations + servable checkpoint
python xrec_ref/train.py --out runs/xrec.json --sweep --ckpt-out runs/xrec.pt

# real-log data path (MovieLens format; --fixture writes one for smoke tests)
python xrec_ref/dataset.py --fixture data/fixture.dat
python xrec_ref/train.py --data data/fixture.dat --out runs/xrec_data.json

# retrieval + serving
python xrec_ref/index.py                                 # IVF recall vs QPS
python xrec_ref/serve.py --ckpt runs/xrec.pt --self-test  # real socket round-trip

# efficiency + the full comparison matrix
python xrec_ref/bench.py                                 # model / decode / ANN / analytic cost
python scripts/run_experiments.py --seeds 0 1 2          # the table above, ~15 min
```

Everything is regenerated by `scripts/reproduce.ps1` (PowerShell) and its bash twin in the same directory.
There is no `pip install .` step: modules import each other flat (`from rfm import ...`) and run from the
repo root.

## What's inside

| module | what it does | paper |
|---|---|---|
| `xrec_ref/rfm.py` | geodesic interpolation, conditional velocity, tangent projection, Riemannian Euler, CFG merge | eq.20鈥?4, 31 |
| `xrec_ref/model.py` | anchor predictor + late-interaction DiT velocity field, all-prefix pretraining mask, U2I / U2I-K controls | eq.25鈥?1, 34鈥?6, Fig.3(a) |
| `xrec_ref/sid.py` | residual k-means quantiser (semantic IDs), recon cosine 0.938 | 搂A.1 |
| `xrec_ref/models_sid.py` | SID-AR entity control: autoregressive decoding of 4 SID tokens | 搂A.2 |
| `xrec_ref/data.py` | synthetic multi-mode interest stream, K-means anchors, Recall@(K脳R) | 搂4.1.1, 搂4.4 |
| `xrec_ref/dataset.py` | real-log ingestion (MovieLens-1M/100K/CSV) 鈫?PPMI + truncated SVD 鈫?anchors 鈫?leave-one-out splits | 搂3.1, 搂4.2 |
| `xrec_ref/index.py` | dependency-free IVF-flat ANN, recall-vs-QPS sweep | retrieval |
| `xrec_ref/baselines.py` | popularity / oracle-mode / uniform references + retrieval-profile diagnostics | calibration |
| `xrec_ref/train.py` | two-stage training, ablations, chunked sampling, servable checkpoints | 搂3.4, 搂4.4.2 |
| `xrec_ref/serve.py` | HTTP `/health` `/metrics` `/recommend`, index built at startup | serving |
| `xrec_ref/bench.py` | late-interaction vs full-layer denoising, X-Rec vs SID-AR decode, ANN, analytic cost | 搂4.3.3 |

## Design notes

Deliberate engineering choices, all reversible:

- **Late interaction, implemented literally.** Condition tokens are encoded by layers `1..L-1`, their
  last-layer K/V are cached once per request, and every denoising step runs exactly one Transformer layer
  for all K triggers. Measured 2.74脳 cheaper than full-layer denoising at `L=12`.
- **All-prefix pretraining** (`loss_all_prefix`): one causal encoder pass, then N鈭? next-item targets, each
  with its own last-layer pass. Replaces the random-prefix recipe; no measurable quality gain here, kept
  because it is what the paper describes.
- **Chunked sampling** (`n_eval_chunk`): the late-interaction cache is replicated K脳 per request, so a
  4096-row eval batch would materialise gigabytes. Both `evaluate()` and the serving path chunk.
- **No faiss.** `index.py` is an IVF-flat index in ~200 lines of torch: vectors sorted by coarse list,
  `search(q, k, nprobe)` with `nprobe == nlist` degenerating to exact search 鈥?which is what the tests
  assert against.

## Deviations from the paper

| | paper | here |
|---|---|---|
| backbone | Qwen3-0.6B, d=1024, L=28, N=200 | 3 layers, d=160, N=48 (CPU/GPU-minutes, not hours) |
| anchors | learned cluster index / RQ codebook | k-means over item embeddings |
| item vectors | multimodal content tower | synthetic geometry, or PPMI+SVD for real logs |
| SID quantiser | RQ-VAE | residual k-means (no VAE), recon cosine 0.938 vs 0.94 |
| ANN | production index | IVF-flat, no PQ/HNSW |
| scale | ~1e9 items, 0.46B params | 6k items, 鈮?M params |

## Limitations

- Absolute recall is far below a popularity baseline on this fixture; treat every table as a *relative*
  comparison under one protocol, not as evidence that the method works.
- Single-seed cells except the headline table; no significance tests beyond intervals over 3 seeds.
- The real-data path is exercised on a log in MovieLens format, not on MovieLens itself (network egress is
  blocked in the build environment). Pointing `--data` at a real `ratings.dat` is a one-flag change.
- Serving is single-process, in-memory, no auth/TLS 鈥?a reference path, not a deployment.

## Citation

If this code helps you, cite the paper it implements:

```bibtex
@article{xrec2026,
  title   = {X-Rec Technical Report},
  author  = {Shen, Chenglei and Huang, Chenzhe and Jiang, Dong and Gao, Hongjie and Zhang, Jue and Xu, Kun and others},
  journal = {arXiv preprint arXiv:2609.29180},
  year    = {2026}
}
```

## License and attribution

MIT for the code (see [LICENSE](LICENSE)). The paper is 漏 2026 ByteDance and is **not** redistributed
here; [NOTICE](NOTICE) records the attribution, the absence of affiliation, and the fact that no
third-party dataset is bundled.
