#!/usr/bin/env bash
# Full reproduction of every number quoted in the README (~20 min on CPU, faster with one GPU).
# Every step writes into runs/ and data/; nothing here needs network access.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs data
export PYTHONIOENCODING=utf-8

step() { echo "== $1 =="; }

step "1/12 RFM numerical identities"
python xrec_ref/rfm.py --selftest

step "2/12 unit tests (math, model, data path, ANN, SID-AR, serving)"
python -m pytest -q

step "3/12 X-Rec (full) + anchor-temperature sweep + servable checkpoint"
python xrec_ref/train.py --eval-n 4096 --out runs/xrec.json --sweep --ckpt-out runs/xrec.pt

step "4/12 ablation: no anchor conditioning"
python xrec_ref/train.py --eval-n 4096 --no-anchor --out runs/xrec_noAC.json

step "5/12 ablation: no RFM (Euclidean rectified flow)"
python xrec_ref/train.py --eval-n 4096 --no-rfm --out runs/xrec_noRFM.json

step "6/12 ablation: no pretraining"
python xrec_ref/train.py --eval-n 4096 --no-pretrain --out runs/xrec_noPT.json

step "7/12 ablation: legacy random-prefix pretraining mask"
python xrec_ref/train.py --eval-n 4096 --pretrain-mask last --out runs/xrec_lastprefix.json

step "8/12 control: U2I single vector"
python xrec_ref/train.py --eval-n 4096 --arch u2i --out runs/u2i.json

step "9/12 control: U2I multi-interest (ComiRec-SA)"
python xrec_ref/train.py --eval-n 4096 --arch u2i_k --out runs/u2i_k.json

step "10/12 control: SID-AR (residual-quantised semantic IDs)"
python xrec_ref/train.py --eval-n 4096 --arch sid_ar --out runs/sid_ar.json

step "11/12 real-format interaction log (MovieLens layout)"
python xrec_ref/dataset.py --fixture data/fixture.dat
python xrec_ref/train.py --data data/fixture.dat --eval-n 1024 --out runs/xrec_data.json

step "12/12 efficiency: model / decode / ANN / analytic"
python xrec_ref/bench.py

step "serving self-test (ephemeral port, one real request)"
python xrec_ref/serve.py --ckpt runs/xrec.pt --self-test

step "seed sweep: mean[min-max] over 3 seeds (optional, ~15 min)"
python scripts/run_experiments.py --seeds 0 1 2 --eval-n 4096

step "done: artifacts in runs/ (summary in runs/exp/summary.json)"
