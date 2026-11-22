# Full reproduction of every number quoted in the README (~20 min on CPU, faster with one GPU).
# Every step writes into runs/ and data/; nothing here needs network access.
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force runs, data | Out-Null

Write-Host "== 1/12 RFM numerical identities =="
python -X utf8 xrec_ref/rfm.py --selftest

Write-Host "== 2/12 unit tests (math, model, data path, ANN, SID-AR, serving) =="
python -X utf8 -m pytest -q

Write-Host "== 3/12 X-Rec (full) + anchor-temperature sweep + servable checkpoint =="
python -X utf8 xrec_ref/train.py --eval-n 4096 --out runs/xrec.json --sweep --ckpt-out runs/xrec.pt

Write-Host "== 4/12 ablation: no anchor conditioning =="
python -X utf8 xrec_ref/train.py --eval-n 4096 --no-anchor --out runs/xrec_noAC.json

Write-Host "== 5/12 ablation: no RFM (Euclidean rectified flow) =="
python -X utf8 xrec_ref/train.py --eval-n 4096 --no-rfm --out runs/xrec_noRFM.json

Write-Host "== 6/12 ablation: no pretraining =="
python -X utf8 xrec_ref/train.py --eval-n 4096 --no-pretrain --out runs/xrec_noPT.json

Write-Host "== 7/12 ablation: legacy random-prefix pretraining mask =="
python -X utf8 xrec_ref/train.py --eval-n 4096 --pretrain-mask last --out runs/xrec_lastprefix.json

Write-Host "== 8/12 control: U2I single vector =="
python -X utf8 xrec_ref/train.py --eval-n 4096 --arch u2i --out runs/u2i.json

Write-Host "== 9/12 control: U2I multi-interest (ComiRec-SA) =="
python -X utf8 xrec_ref/train.py --eval-n 4096 --arch u2i_k --out runs/u2i_k.json

Write-Host "== 10/12 control: SID-AR (residual-quantised semantic IDs) =="
python -X utf8 xrec_ref/train.py --eval-n 4096 --arch sid_ar --out runs/sid_ar.json

Write-Host "== 11/12 real-format interaction log (MovieLens layout) =="
python -X utf8 xrec_ref/dataset.py --fixture data/fixture.dat
python -X utf8 xrec_ref/train.py --data data/fixture.dat --eval-n 1024 --out runs/xrec_data.json

Write-Host "== 12/12 efficiency: model / decode / ANN / analytic =="
python -X utf8 xrec_ref/bench.py

Write-Host "== serving self-test (ephemeral port, one real request) =="
python -X utf8 xrec_ref/serve.py --ckpt runs/xrec.pt --self-test

Write-Host "== seed sweep: mean[min-max] over 3 seeds (optional, ~15 min) =="
python -X utf8 scripts/run_experiments.py --seeds 0 1 2 --eval-n 4096

Write-Host "== done; artifacts in runs/ (summary in runs/exp/summary.json) =="
