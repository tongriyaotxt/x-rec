# x-rec

**X-Rec 鐨勭嫭绔嬮噸瀹炵幇锛堟満鍒剁骇锛?* 鈥斺€?鎶婃帹鑽愬彫鍥炲啓鎴愩€屽湪 item 杩炵画鍚戦噺绌洪棿閲屽仛鏉′欢鐢熸垚銆嶏細
榛庢浖娴佸尮閰?+ anchor 鏉′欢鍖?+ late-interaction 鎵╂暎 Transformer銆?
[![ci](https://github.com/tongriyaotxt/x-rec/actions/workflows/ci.yml/badge.svg)](https://github.com/tongriyaotxt/x-rec/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-2f6f4f)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-4b5563)](pyproject.toml)

[English](README.md) 路 **涓枃** 路 [瀹屾暣鐮旂┒鎶ュ憡锛堟満鍒舵帹瀵?/ 璁烘枃鍏ㄩ儴鏁板瓧 / 寮€婧愮敓鎬侊級](docs/x-rec-research-zh.md)

> **闈炲畼鏂?*锛氫笌瀛楄妭璺冲姩 / TikTok 鍙婅鏂囦綔鑰呮棤浠讳綍鍏宠仈銆佹湭鑾疯儗涔︺€佹湭缁忓闃咃紱浠ｇ爜鎸夎鏂囧叕寮忕嫭绔嬪疄鐜帮紝
> 涓嶅惈浠讳綍鍘傚晢浠ｇ爜銆?*涓嶆槸澶嶇幇**锛氬熀鍑嗘槸鍒绘剰鍋氬皬鐨勫悎鎴愭暟鎹紝涓嬮潰鐨勬暟瀛椾笌璁烘枃鐨勭敓浜ф寚鏍囦笉鍙瘮銆?
---

## 涓轰粈涔堝仛杩欎釜

X-Rec 鎶婂彫鍥炴崲鎴愮敓鎴愶細涓嶅啀鎶婄敤鎴锋睜鍖栨垚涓€涓悜閲忥紝鑰屾槸瀛?`p(x | c)`锛岀洿鎺?*鐢熸垚** K 涓Е鍙戝悜閲忓幓鏌?ANN銆?璁烘枃鎶ョ殑鏄?Recall@20 = 10.76%銆?75.7 QPS锛屽鐓х殑绂绘暎 ID 鏂规鏄?10.23% / 166.4 QPS銆?
澶栭儴璇昏€呯湡姝ｈ兘楠岃瘉鐨勫彧鏈変袱浠朵簨锛?*鍏紡瀹炵幇瀵逛笉瀵?*銆?*鍚岄绠椾笅鐨勬帓搴忚繕鎴愪笉鎴愮珛**銆傝繖涓粨搴撴妸涓や欢浜嬮兘鍋氫簡锛?缁撹鍜屾憳瑕佺粰浜虹殑鍗拌薄骞朵笉涓€鑷达細

- RFM 鏁板瀵瑰埌 3.0e-4锛宭ate interaction 纭疄鏇寸渷锛屽瑙﹀彂纭疄瑕嗙洊澶氬嘲锛?- 浣嗗湪 6k item 鐨勫悎鎴愬簱涓婏紝**鎵€鏈夊涔犲瀷鍙樹綋閮芥墦涓嶈繃銆岀洿鎺ヨ繑鍥炴渶鐑?20 涓€?*锛涜€?SID-AR 瀵圭収涔嬫墍浠ヨ耽锛?  鏄洜涓哄畠**宕╁鎴愮儹搴﹀厛楠?*锛屼笉鏄洜涓哄缓妯℃洿濂姐€?
涓嬮潰鏄暟瀛楀拰浜х敓杩欎簺鏁板瓧鐨勪唬鐮併€?
## 缁撴灉涓€瑙?
3 seed锛宍n_eval=4096`锛屾瘡涓€琛岀殑鍊欓€夐绠楅兘鏄?20锛坄scripts/run_experiments.py`锛夛細

| 鍙樹綋 | Recall@20锛?0脳1锛?| Recall@20锛?脳20锛?| 妯″紡瑕嗙洊 | 瑙﹀彂涓や袱浣欏鸡 | ms/璇锋眰 |
|---|---|---|---|---|---|
| **X-Rec** | **0.88%** [0.85鈥?.90] | **1.13%** [0.98鈥?.34] | 99.0% | 0.197 | 0.34 |
| 鍘绘帀 anchor 鏉′欢鍖?| 0.81% [0.56鈥?.15] | 0.83% | 99.3% | 0.163 | 0.33 |
| 鍘绘帀 RFM锛堟姘?rectified flow锛?| 0.68% [0.44鈥?.88] | 0.85% | 99.3% | 0.162 | 0.33 |
| 鍘绘帀棰勮缁?| 0.44% [0.42鈥?.46] | 0.44% | 97.1% | 0.249 | 0.32 |
| 鏃ч殢鏈哄墠缂€棰勮缁冿紙瀵圭収锛?| 0.91% [0.85鈥?.98] | 1.18% | 99.7% | 0.234 | 0.32 |
| U2I锛堝崟鍚戦噺鍙屽锛?| 0.02% | 0.71% | 91.0% | 1.000 | 0.04 |
| U2I-K锛圕omiRec-SA锛孠=20锛?| 0.07% | 0.90% | 96.2% | 0.972 | 0.04 |
| SID-AR锛?脳256 RQ锛屽疄浣撴ā鍨嬶級 | 2.80% [2.47鈥?.03] | 1.58% | 100% | 0.146 | 0.10 |

**璺?3 seed 绔欏緱浣忕殑**锛氶璁粌鏄敮涓€闈為噸鍙犵殑宸窛锛?.88% 鈫?0.44%锛夛紱鍚岄绠椾笅 X-Rec 浼樹簬涓や釜 U2I 瀵圭収锛?澶氬叴瓒?U2I 鍙槸**琛ㄩ潰鍒嗘暎**锛堜綑寮?0.972锛夆€斺€旇繖涓夌偣涓庤鏂囦竴鑷淬€?
**绔欎笉浣忕殑**锛歛nchor 鏉′欢鍖栵紙0.81 卤 0.30锛変笌 RFM锛?.68 卤 0.22锛夎惤鍦?seed 鍣０閲岋紱Fig.3(a) 鐨?all-prefix mask 鐩稿鏃х殑闅忔満鍓嶇紑璺緞娌℃湁鍙祴鏀剁泭銆?
## 鏈€璇ョ湅鐨勪竴鑺傦細鏍″噯

鍚屾牱 20 涓€欓€夈€佸悓涓€涓祴璇曢泦锛?
| 鍙傜収 | Recall@20 | 鍚箟 |
|---|---|---|
| 鍧囧寑 | 0.33% | 鍦版澘 |
| **鐑害@20** | **3.93%** | 璁粌闆嗘渶楂橀鐨?20 涓?item锛屾墍鏈夎姹傝繑鍥炲悓涓€浠?|
| 妯″紡鍐呯儹搴20 | **14.53%** | 鐢ㄦ埛**鐪熷疄**鐩爣妯″紡鍐呮渶楂橀鐨?20 涓紝浠嶇劧蹇界暐鍏朵綑淇℃伅 |

鎵€鏈夊涔犲瀷鍙樹綋閮戒綆浜庤繖鏉℃渶鏈寸礌鐨勫熀绾裤€俋-Rec 鐨?99% 妯″紡瑕嗙洊鍙鏄庤Е鍙戣惤瀵逛簡鍏磋叮绨囷紝瀹屽叏涓嶄唬琛ㄧ皣鍐呮帓搴?锛?.88% vs 14.53% 鐨勭皣鍐呬笂鐣岋級銆傛満鍒朵笂锛氬悎鎴愪笘鐣岄噷 item 鐨勭儹搴﹀彧缂栫爜鍦ㄣ€岃窛绨囧績鐨勫崐寰勩€嶄笂锛岃€?RFM 鐨勫洖褰?鐩爣鏄洰鏍?item 鐨?*鏂瑰悜**鈥斺€旀ā鍨嬪浼氫簡銆岃惤鍦ㄥ摢涓皣銆嶏紝娌″浼氥€岃惤鍦ㄧ皣鍐呭娣便€嶃€?
`scripts/diagnose_prior.py` 鎻ず浜?SID-AR 鐨勩€岃耽銆嶅埌搴曟槸浠€涔堬細

| 鍙樹綋 | Recall@20 | 骞冲潎鐑害鍒嗕綅 | 钀藉湪鍏ㄥ眬 top-20 鐨勬瘮渚?| 杩斿洖杩囩殑涓嶅悓 item |
|---|---|---|---|---|
| X-Rec | 1.17% | 0.746 | 5.7% | **3680** |
| SID-AR | 1.46% | 0.986 | 56.8% | **23** |

SID-AR 鍦?**4096 娆¤姹傞噷鍙悙鍑?23 涓笉鍚?item**鈥斺€斿畠鍦ㄨ儗鐑害鍏堥獙锛屾墍浠ュ彫鍥炲垰濂借创鍦ㄧ儹搴﹀熀绾夸笅鏂广€?X-Rec 閾哄紑鍒?3680 涓?item锛氫釜鎬у寲鏄湡鐨勶紝鍙槸鍊欓€変綅缃病鏀惧ソ銆?
杩欐槸**鏈熀鍑?+ 鏈缁冮绠?*锛堢害 1M 鍙傛暟銆?00 + 300 姝ワ級鐨勭粨璁猴紝涓嶆槸瀵瑰崄浜跨骇瑙勬ā涓?SID-AR 鐨勫垽鏂€斺€?閭ｉ噷涓€涓?SID 琚ぇ閲?item 鍏变韩锛岃儗涓嶅嚭鏉ャ€備篃姝ｆ槸鍥犱负杩欎竴鐐癸紝鏈粨搴撲笉鍐欍€屼笁涓?idea 閮芥垚绔嬨€嶃€?
## 蹇€熷紑濮?
```bash
pip install -r requirements.txt          # torch + numpy + scikit-learn锛屼粠浠撳簱鏍圭洰褰曡繍琛?
python xrec_ref/rfm.py --selftest        # 6 椤规暟鍊兼亽绛夊紡锛屼笉璁粌
pytest -q                                # 41 椤规祴璇曪細鏁板 / ANN / 鏃ュ織瑙ｆ瀽 / 鏈嶅姟 / HTTP

# 璁粌 + 娑堣瀺 + 鍙湇鍔″寲 ckpt
python xrec_ref/train.py --out runs/xrec.json --sweep --ckpt-out runs/xrec.pt

# 鐪熷疄鏃ュ織鏁版嵁閫氳矾锛圡ovieLens 鏍煎紡锛?-fixture 鍙敓鎴愬悓鏍煎紡鏃ュ織鍋氬啋鐑燂級
python xrec_ref/dataset.py --fixture data/fixture.dat
python xrec_ref/train.py --data data/fixture.dat --out runs/xrec_data.json

# 妫€绱笌鏈嶅姟
python xrec_ref/index.py                                 # IVF 鐨?recall / QPS 鏇茬嚎
python xrec_ref/serve.py --ckpt runs/xrec.pt --self-test  # 鐪?socket 鎵撲竴娆¤姹?
# 鏁堢巼涓庡畬鏁村鐓х煩闃?python xrec_ref/bench.py                                 # model / decode / ANN / 瑙ｆ瀽鎴愭湰
python scripts/run_experiments.py --seeds 0 1 2          # 涓婇潰閭ｅ紶琛紝绾?15 鍒嗛挓
```

鎵€鏈夋暟瀛楃敱 `scripts/reproduce.ps1`锛堝強鍚岀洰褰曠殑 bash 鐗堬級涓€閿噸璺戙€傛病鏈?`pip install .` 杩欎竴姝ワ細
妯″潡涔嬮棿鏄墎骞冲鍏ワ紙`from rfm import ...`锛夛紝浠庝粨搴撴牴鐩綍鐩存帴璺戙€?
## 鐩綍缁撴瀯

| 妯″潡 | 鍐呭 | 瀵瑰簲璁烘枃 |
|---|---|---|
| `xrec_ref/rfm.py` | 鐞冮潰娴嬪湴绾挎彃鍊笺€佹潯浠堕€熷害鍦恒€佸垏绌洪棿鎶曞奖銆侀粠鏇?Euler銆丆FG 鍚堝苟 | eq.20鈥?4, 31 |
| `xrec_ref/model.py` | anchor 棰勬祴澶?+ late-interaction DiT锛沘ll-prefix 棰勮缁?mask锛沀2I / U2I-K 瀵圭収 | eq.25鈥?1, 34鈥?6, Fig.3(a) |
| `xrec_ref/sid.py` | 娈嬪樊 k-means 閲忓寲鍣紙璇箟 ID锛夛紝閲嶅缓浣欏鸡 0.938 | 搂A.1 |
| `xrec_ref/models_sid.py` | SID-AR 瀹炰綋瀵圭収锛氳嚜鍥炲綊瑙ｇ爜 4 涓?SID token | 搂A.2 |
| `xrec_ref/data.py` | 鍚堟垚澶氬嘲鍏磋叮娴併€並-means anchor銆丷ecall@(K脳R) | 搂4.1.1, 搂4.4 |
| `xrec_ref/dataset.py` | 鐪熷疄鏃ュ織锛坢l-1m / ml-100k / CSV锛夆啋 PPMI+SVD item 鍚戦噺 鈫?anchor 鈫?鐣欎竴鍒掑垎 | 搂3.1, 搂4.2 |
| `xrec_ref/index.py` | 鏃?faiss 鐨?IVF-flat ANN + recall/QPS 鎵弿 | 妫€绱㈠眰 |
| `xrec_ref/baselines.py` | 鐑害 / 妯″紡鍐呯儹搴?/ 鍧囧寑鍩虹嚎 + 妫€绱㈢敾鍍忚瘖鏂?| 鏍″噯 |
| `xrec_ref/train.py` | 涓ら樁娈佃缁冦€佹秷铻嶃€佸垎鍧楅噰鏍枫€佸彲鏈嶅姟鍖?ckpt | 搂3.4, 搂4.4.2 |
| `xrec_ref/serve.py` | HTTP `/health` `/metrics` `/recommend`锛屽惎鍔ㄥ嵆寤虹储寮?| 鏈嶅姟鍖?|
| `xrec_ref/bench.py` | late-interaction vs 鍏ㄥ眰鍘诲櫔銆乆-Rec vs SID-AR 瑙ｇ爜銆丄NN銆佽В鏋愭垚鏈?| 搂4.3.3 |

## 鍑犱釜瀹炵幇涓婄殑閫夋嫨

- **late interaction 鎸夊瓧闈㈠疄鐜?*锛歝ondition 璧板畬 `1..L-1` 灞傦紝鏈€鍚庝竴灞?K/V 姣忚姹傚彧绠椾竴娆★紝姣忎釜鍘诲櫔姝?  鍙窇涓€灞?Transformer锛圞 涓Е鍙戜竴璧凤級銆侺=12 鏃跺疄娴嬫瘮鍏ㄥ眰鍘诲櫔鐪?2.74脳銆?- **all-prefix 棰勮缁?*锛坄loss_all_prefix`锛夛細涓€閬嶅洜鏋滅紪鐮佸櫒鍓嶅悜锛岀劧鍚?N鈭? 涓綅缃悇鑷娴嬩笅涓€椤癸紝
  姣忎釜浣嶇疆鍙璺戞渶鍚庝竴灞傘€傛浛浠ｉ殢鏈哄墠缂€鏂规锛涜繖閲屾病鏈夋祴鍑鸿川閲忔敹鐩婏紝淇濈暀鏄洜涓鸿鏂囪繖涔堝啓銆?- **鍒嗗潡閲囨牱**锛坄n_eval_chunk`锛夛細late-interaction 鐨?K/V 浼氭寜 K 鍊嶅鍒讹紝4096 琛屼竴璧烽噰鏍蜂細鍚冩帀鍑?GB锛?  `evaluate()` 鍜屾湇鍔¤矾寰勯兘鍒嗗潡銆?- **涓嶇敤 faiss**锛歚index.py` 鏄害 200 琛?torch 鐨?IVF-flat鈥斺€斿悜閲忔寜绮楄仛绫绘帓搴忓瓨鏀撅紝`nprobe == nlist`
  閫€鍖栦负绮剧‘妫€绱紝娴嬭瘯灏辩敤杩欐潯鎬ц川鍋氱┓涓剧瓑浠锋柇瑷€銆?
## 涓庤鏂囩殑鍋忕

| | 璁烘枃 | 鏈粨搴?|
|---|---|---|
| 涓诲共 | Qwen3-0.6B锛宒=1024锛孡=28锛孨=200 | 3 灞傦紝d=160锛孨=48锛堝垎閽熺骇锛屼笉鏄皬鏃剁骇锛?|
| anchor | 瀛︿範寮忕皣绱㈠紩 / RQ 鐮佹湰 | item 鍚戦噺涓婄殑 k-means |
| item 鍚戦噺 | 澶氭ā鎬佸唴瀹瑰 | 鍚堟垚鍑犱綍锛涚湡瀹炴棩蹇楄蛋 PPMI+SVD |
| SID 閲忓寲鍣?| RQ-VAE | 娈嬪樊 k-means锛堟棤 VAE锛夛紝閲嶅缓 0.938 vs 璁烘枃 0.94 |
| ANN | 鐢熶骇绱㈠紩 | IVF-flat锛屾棤 PQ/HNSW |
| 瑙勬ā | 鍗佷嚎绾?item銆?.46B 鍙傛暟 | 6000 item銆佺害 1M 鍙傛暟 |

## 闄愬埗

- 缁濆鍙洖杩滀綆浜庣儹搴﹀熀绾匡紱鎵€鏈夎〃鏍奸兘鍙槸**鍚屼竴鍗忚涓嬬殑鐩稿姣旇緝**锛屼笉鑳藉綋浣溿€屾柟娉曟湁鏁堛€嶇殑璇佹嵁銆?- 闄や富琛ㄥ閮芥槸鍗?seed锛涙樉钁楁€у彧缁欎簡 3 seed 鐨勫尯闂达紝娌℃湁鍋氭楠屻€?- 鐪熷疄鏁版嵁閫氳矾鍙湪銆孧ovieLens 鏍煎紡鐨勬棩蹇椼€嶄笂璺戦€氾紝娌″湪 MovieLens 鏈韩涓婅窇锛堟瀯寤虹幆澧冪缃戯級銆?  `--data` 鎸囧悜鐪熷疄 `ratings.dat` 鏄竴琛屾敼鍔ㄣ€?- 鏈嶅姟鏄崟杩涚▼銆佸唴瀛樻€併€佹棤閴存潈/鏃?TLS鈥斺€斿弬鑰冨疄鐜帮紝涓嶆槸鍙儴缃叉柟妗堛€?
## 寮曠敤

```bibtex
@article{xrec2026,
  title   = {X-Rec Technical Report},
  author  = {Shen, Chenglei and Huang, Chenzhe and Jiang, Dong and Gao, Hongjie and Zhang, Jue and Xu, Kun and others},
  journal = {arXiv preprint arXiv:2609.29180},
  year    = {2026}
}
```

## 璁稿彲涓庡綊灞?
浠ｇ爜 MIT锛堣 [LICENSE](LICENSE)锛夈€傝鏂囩増鏉冨綊 ByteDance 鎵€鏈夛紝鏈粨搴?*涓嶈浆杞?*锛?[NOTICE](NOTICE) 璁板綍浜嗗綊灞炪€佹棤鍏宠仈澹版槑锛屼互鍙娿€屾湭鎵撳寘浠讳綍绗笁鏂规暟鎹泦銆嶃€?