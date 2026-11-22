# 宣传片（抖音竖屏）

`xrec-promo-1080x1920.mp4` — 85.1 s，1080×1920，H.264 / AAC，3.1 MB。

| 项 | 值 |
|---|---|
| 发布 | 抖音，2026-11-22 22:22，标题「抖音新推荐算法 X-Rec，我照论文重写了一遍」 |
| 分镜 | 9 个镜头：论文现状 → 三步机制 → 三项数值自检 → 消融 → 热度基线 → SID-AR 崩塌 → 工程侧 → 仓库 |
| 口播 | edge-tts `zh-CN-YunjianNeural`，-6%，83.0 s，约 259 字/分 |
| 画面 | 纸白底 + 青/红强调，1920×1080 设计后改竖屏；底部 180 px 留字幕 |
| 已知问题 | 该版由旧切点逻辑渲染，片尾画面比音频约快 3.8 s（字幕对，画面略超前）；修复已进 toolchain，重渲即好 |

重渲方式（工具链在 `F:\shit\toolchain\video`）：

```powershell
$P = "F:\shit\toolchain\video\projects\xrec-promo"
python F:\shit\toolchain\video\tts_edge.py --project $P --rate=-6%
python $P\build_scenes.py --render
python F:\shit\toolchain\video\make_srt.py --sent $P\_work\voice\sentences.json --out $P\_work\sub.srt
python F:\shit\toolchain\video\make_bgm.py --dur 88 --out $P\_work\bgm.wav
python F:\shit\toolchain\video\assemble.py --project $P --name xrec-成片 --size 1080x1920
```
