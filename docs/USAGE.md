# STAGE · Adaptive-global

This package contains the core model, preprocessing and inference utilities. Training workflows, alternative forecasting methods and clinical classifiers are outside its scope.

## Preprocessing

Raw Lead II (mV) → resample to 500 Hz → remove baseline wander with a 0.5 Hz high-pass filter and screen signal quality → detect R peaks → extract and screen 600-sample beats → compute the median beat → apply a 100 Hz low-pass filter and a 60 Hz notch filter → resample to 400 Hz → pad with 16 zeros on each side → obtain a 512-sample input.

Input amplitudes remain in mV, without normalization or nonlinear time warping. The R peak is at Python index 208. Raw ECG preprocessing requires MATLAB Signal Processing Toolbox. ADC-to-mV conversion and Lead II selection are performed by the data reader. Filter settings and quality thresholds are defined in `preprocessing/matlab/pipeline_config.json`.

```matlab
addpath('preprocessing/matlab');
result = stage_preprocess(leadII_mV, originalFs); % Pass 50 as the third argument for a 50 Hz notch filter
assert(result.quality_pass, result.exclusion_reason);
save('beat.mat', 'result', '-v7');
```

## Usage

Run from the repository root with Python ≥ 3.10:

```bash
pip install -r requirements.txt
python -m stage.prepare --mat beat.mat --age 55 --output source.npz
# Convert only your own trusted original checkpoint; this step is needed once.
python -m stage.export /path/to/model_final.pth weights.pt --trust-original
python -m stage.infer --checkpoint weights.pt --input source.npz --output forecast.npz --horizons 1 3 5 --gain 1.2
python examples/plot.py --input forecast.npz --output plots
```

Model flow: Source → encoder → Evolution 8D / Invariant 40D → global age direction × age interval × adaptive intensity → decode while preserving the invariant representation. G1/G3/G5 are generated from the same Source without recursive rollout. Ages must satisfy `18 ≤ source_age ≤ target_age ≤ 95`.

Input NPZ fields: `source [N,1,512]` and `age [N]`. Output additionally includes `forecast [N,H,1,512]`, `horizons` and `gain`.
