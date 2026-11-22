# xrec_ref — X-Rec 最小可运行参考实现

对应论文：**X-Rec Technical Report**（ByteDance / TikTok，arXiv:2609.29180v1，2026-09-24）。
官方未发布代码，本目录是按论文公式独立实现的可运行版本：CPU 单机、约 1M 参数、全流程 <2 分钟（GPU 更快）。

## 文件

| 文件 | 内容 | 对应论文 |
|---|---|---|
| `rfm.py` | 球面测地线插值、条件速度场、RFM 损失目标、切空间投影、Riemannian Euler、CFG 合并 | eq.20–24, eq.31 |
| `config.py` | 超参：`paper` 生产值 vs `demo` CPU 值 | §4.1.4 |
| `data.py` | 合成多峰兴趣流（8 个语义簇、双模式马尔可夫链、Zipf 热度）、K-means anchor、Recall@(K×R) | §4.1.1, §4.4 |
| `dataset.py` | 真实交互日志 → PPMI 共现 → 截断 SVD item 向量 → K-means anchor → 留一划分（ml-1m / ml-100k / CSV 自动识别） | §3.1, §4.2 |
| `model.py` | anchor 预测头 + late-interaction DiT 速度场；U2I / U2I-K 对照；**all-prefix 预训练 mask** | eq.25–31, eq.34–36, Fig.3(a) |
| `sid.py` | 残差 k-means 量化器（= 论文语义 ID 的构造，去掉 VAE） | §A.1 |
| `models_sid.py` | **SID-AR 实体对照**：同一骨干、同一 item 空间，改为自回归解码 4 个 SID token | §A.2 |
| `index.py` | 无依赖 IVF-flat ANN（build / save / load / search + recall-vs-QPS 扫描） | §4.2 检索 |
| `baselines.py` | 热度基线、模式内热度上界、均匀基线、检索画像（是否在背热度先验） | 评测校准 |
| `train.py` | 两阶段训练（pretrain → streaming SFT）+ 消融 + 评测 + 温度扫描 + 可服务化 ckpt | §3.4, §4.4.2 |
| `serve.py` | HTTP 服务：`/health` `/metrics` `/recommend`，启动时建 ANN，ckpt 自带 item 表 | 部署路径 |
| `bench.py` | late-interaction vs 全层去噪实测、X-Rec vs SID-AR 解码实测、ANN 曲线、论文尺寸解析成本模型 | §4.3.3 |

## 运行

```bash
python rfm.py --selftest                                    # 数学自检（6 项恒等式）
python train.py --out runs/xrec.json --sweep --ckpt-out runs/xrec.pt
python train.py --no-anchor   --out runs/xrec_noAC.json
python train.py --no-rfm      --out runs/xrec_noRFM.json
python train.py --no-pretrain --out runs/xrec_noPT.json
python train.py --pretrain-mask last --out runs/xrec_lastprefix.json   # 旧的随机前缀路径
python train.py --arch u2i    --out runs/u2i.json
python train.py --arch u2i_k  --out runs/u2i_k.json
python train.py --arch sid_ar --out runs/sid_ar.json        # SID-AR 实体对照

python dataset.py --fixture data/fixture.dat                # 造一份 MovieLens 格式的日志
python train.py --data data/fixture.dat --out runs/xrec_data.json

python index.py                                             # IVF 自测（recall / QPS）
python serve.py --ckpt runs/xrec.pt --self-test             # 真 socket 打一次请求
python bench.py                                             # model / decode / ann / analytic
```

Windows 下用 `pwsh` + `python -X utf8`，或设 `$env:PYTHONIOENCODING="utf-8"`。

## 关键实现约定

一致：
- 速度场回归目标 = 测地线的解析导数 `ω/sinω · (cos(tω)·x₁ − cos((1−t)ω)·ε)`；
- 每个 denoise step 只跑 **最后一层** Transformer，condition 的该层 K/V 只算一次并复用；
- 触发向量逐点投影到切空间，Euler 步用指数映射，保证全程在单位球面上；
- anchor 用 CE 监督，anchor 采样温度控制触发多样性；推理用 CFG（训练时 5% 上下文丢弃）；
- SFT 每目标 8 个噪声副本，副本之间互不可见（论文的 diagonal mask 语义）；
- **预训练改为 Figure 3(a) 的 all-prefix**：一遍编码器前向，每个位置预测下一项（`loss_all_prefix`）；
- **SID-AR 对照是实体模型**，不是成本模型：4×256 残差量化 + 自回归 beam 解码，重建余弦 0.938。

偏离（规模与工程简化，不影响机制验证）：
- 主干 `d_m=160 / L=3`（论文 Qwen3-0.6B：`d_m=1024 / L=28`），因此吞吐倍率远小于论文；
- anchor 由 item embedding 的 K-means 得到（论文用语义簇索引 / RQ 码本第一层）；
- 合成数据只有 6000 item、8 个兴趣簇（论文为十亿级真实视频流）；
- 真实数据通路的 item 向量用 PPMI+SVD（论文是多模态内容塔）；`ratings` 的取值与时间间隔被当作两个
  sub-attribute 的代理（真实日志里没有 action/duration 字段）；
- ANN 是 IVF-flat，没有 PQ/HNSW（`faiss` 不在依赖里）。

## 已验证的数值事实

| 检查 | 结果 |
|---|---|
| 条件速度场 vs 测地线有限差分（`rfm.py --selftest`） | 3.0e-4 |
| 中间态 / 速度的球面约束 | 6e-8 / 5e-8 |
| Euler(200 步) 复现 x₁ | 2.3e-6 |
| 欧氏直线插值偏离球面 | 0.42 |
| RQ 4×256 重建余弦（6000 item） | 0.938（论文 0.94） |
| RQ 4×256 下 SID 唯一率 | 86.8%（|I|=6000） |
| IVF 50k×128, nlist=256 | recall@10 97.7% @0.14 ms/query；100% @7.4 ms |
| 热度基线 Recall@20（同一预算） | **3.93%**；模式内热度上界 14.53%；均匀 0.33% |

## 已知限制（读数字前先看这里）

1. **绝对召回低于热度基线。** X-Rec 3 seed 均值 0.88%，热度@20 是 3.93%，模式内热度上界 14.53%。
   多触发把候选放进了正确的兴趣簇（模式覆盖 99%），但没学会在簇内按热度排序。
2. **SID-AR 的 2.80% 不是「离散优于连续」。** `scripts/diagnose_prior.py` 显示它在 4096 个请求上只吐出
   **23 个不同 item**、57% 落在全局 top-20，即 beam 崩塌成热度先验；X-Rec 吐出 3680 个不同 item。
3. **anchor conditioning 与 RFM 的单项增益在噪声内**（0.81±0.30 / 0.68±0.22 vs 0.88），
   预训练是唯一跨 seed 稳定的因素（0.44±0.02）。
4. **all-prefix mask 没有实测收益**（0.88 vs 旧路径 0.91，重叠），实现保留，结论如实写。
5. 合成世界的 Zipf 指数是 0.5（`train.py --zipf`），换指数会整体移动所有绝对数字。
