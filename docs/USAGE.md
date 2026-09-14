# STAGE · Adaptive-global

仅保留核心模型、预处理和推理，不含训练流程、其他推演方法或临床分类器。

## 处理流程

原始 Lead II（mV）→ 重采样至 500 Hz → 0.5 Hz 高通去基线与质量筛查 → R 峰检测 → 600 点截搏及形态筛查 → 中位搏 → 100 Hz 低通与 60 Hz 陷波 → 重采样至 400 Hz → 两端各补 16 个零 → 512 点输入。

输入保留 mV 幅度，不归一化、不做非线性时间拉伸；R 位于 Python 索引 208。原始 ECG 预处理需要 MATLAB Signal Processing Toolbox；ADC→mV 转换和 Lead II 选择由数据读取端完成。滤波和质量阈值见 `preprocessing/matlab/pipeline_config.json`。

```matlab
addpath('preprocessing/matlab');
result = stage_preprocess(leadII_mV, originalFs); % 50 Hz 陷波时传入第三参数 50
assert(result.quality_pass, result.exclusion_reason);
save('beat.mat', 'result', '-v7');
```

## 运行

从本目录执行，Python ≥ 3.10：

```bash
pip install -r requirements.txt
python -m stage.prepare --mat beat.mat --age 55 --output source.npz
# 仅转换自己可信的原始 checkpoint；转换后无需重复执行
python -m stage.export /path/to/model_final.pth weights.pt --trust-original
python -m stage.infer --checkpoint weights.pt --input source.npz --output forecast.npz --horizons 1 3 5 --gain 1.2
python examples/plot.py --input forecast.npz --output plots
```

模型流程：Source → encoder → Evolution 8D / Invariant 40D → 全局年龄方向 × 年龄间隔 × 自适应强度 → 保持 Invariant 不变并解码。G1/G3/G5 均从同一 Source 生成，不递归；年龄满足 `18 ≤ source_age ≤ target_age ≤ 95`。

输入 NPZ：`source [N,1,512]`、`age [N]`；输出另含 `forecast [N,H,1,512]`、`horizons`、`gain`。权重需自行转换或另行提供。
