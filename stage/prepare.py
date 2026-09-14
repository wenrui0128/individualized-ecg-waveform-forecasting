"""Validate prepared median beats and save the portable NPZ input contract.

This is a format adapter, NOT a replacement R detector/median-beat pipeline.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat


def validate(source, age):
    source = np.asarray(source, dtype=np.float32)
    age = np.asarray(age, dtype=np.float32).reshape(-1)
    if source.shape != (len(age), 1, 512) or len(age) == 0:
        raise ValueError('Expected source [N,1,512] and age [N].')
    if not np.isfinite(source).all() or not np.isfinite(age).all():
        raise ValueError('Nonfinite beat/age; do not silently impute invalid ECGs.')
    if (age < 18).any() or (age > 95).any():
        raise ValueError('Age must lie in [18,95].')
    if not (np.all(source[:, :, :16] == 0) and np.all(source[:, :, -16:] == 0)):
        raise ValueError('Expected 16 exact zero padding samples at each edge.')
    if np.any(np.std(source[:,:,16:-16], axis=-1) < 0.005):
        raise ValueError('Near-constant input beat.')
    return source, age


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--mat', type=Path, help='Single MATLAB stage_preprocess output saved as variable result (MAT v7, not v7.3)')
    mode.add_argument('--beats', type=Path, help='Already prepared floating NPY [N,1,512], mV')
    p.add_argument('--ages', type=Path, help='NPY ages [N], required with --beats')
    p.add_argument('--age', type=float, help='Scalar years, required with --mat')
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('Output exists; choose a new path.')
    if a.mat:
        if a.age is None: p.error('--age is required with --mat')
        result=loadmat(a.mat, simplify_cells=True)['result']
        if not result['quality_pass']:
            raise ValueError('Preprocessing rejected ECG: '+str(result['exclusion_reason']))
        source=np.asarray(result['waveform']).reshape(1,1,512);age=[a.age]
    else:
        if a.ages is None: p.error('--ages is required with --beats')
        source=np.load(a.beats, allow_pickle=False);age=np.load(a.ages, allow_pickle=False)
    source,age=validate(source,age)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('wb') as f:
        np.savez_compressed(f,source=source,age=age,sampling_rate_hz=400,r_peak_index=208,units='mV')
    print(a.output)

if __name__=='__main__': main()
