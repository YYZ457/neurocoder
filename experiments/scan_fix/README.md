# Selective scan 修复验证

## 问题
旧版 `_selective_scan` 用闭式解 `h_t = p_t * cumsum(b / p)`，其中 `p_t = cumprod(a)`。
`a_t` 最小约为 `exp(-8)`，几个 token 后 `p_t` 就低于 1e-12 的 clamp，`b / p` 失真，
之后所有状态都是错的。修复改用 Hillis-Steele 并行前缀扫描，只有乘法和加法。

## 数值检查（`check_scan.py`，对比逐步循环参考实现）
| L | 旧版最大误差 | 新版最大误差 |
|---|---|---|
| 8 | 6.7 | 1.9e-6 |
| 2048 | 26.2 | 3.8e-6 |

（输出幅值约 13~30，旧版误差与信号同量级。）

## 训练对比（`exp.py`）
字节级 LM，Python 标准库源码，d_model=96、4 blocks、8 experts、B=8、L=128，
同种子同数据，CPU，各 600 步。验证 loss：

| 步数 | 旧 | 新 |
|---|---|---|
| 100 | 2.730 | 2.723 |
| 300 | 1.907 | 1.862 |
| 600 | 1.690 | 1.625 |

按位置分段（第 600 步）：0–7 位置差 0.013，8–127 位置差约 0.065–0.077，
即差距集中在需要前文记忆的位置。

局限：单种子、短序列、600 步、无误差范围；新版 CPU 上每步慢约 35%。

## 复现
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
mkdir -p old && git show 4894c9e:model.py > old/model.py && cp ../../config.py old/
python3 check_scan.py
python3 exp.py new 600 new.json
python3 exp.py old 600 old.json
```
