"""python -m stage.infer --checkpoint weights.pt --input source.npz --output forecast.npz"""
import argparse
from pathlib import Path
import numpy as np
import torch
from .model import STAGE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--input', required=True, type=Path, help='NPZ keys: source [N,1,512] in mV; age [N] in years')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--horizons', type=float, nargs='+', default=[1, 3, 5])
    p.add_argument('--gain', type=float, default=1.0)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists; choose a new file.')
    if args.batch_size < 1:
        p.error('batch-size must be positive')
    model = STAGE.load(args.checkpoint, args.device)
    with np.load(args.input, allow_pickle=False) as data:
        source = np.asarray(data['source'], dtype=np.float32)
        age = np.asarray(data['age'], dtype=np.float32)
    if source.ndim != 3 or source.shape[1:] != (1, 512) or age.shape != (len(source),) or len(source)==0:
        p.error('Invalid source/age shapes or empty input.')
    forecasts = []
    with torch.inference_mode():
        for horizon in args.horizons:
            pieces=[]
            for start in range(0, len(source), args.batch_size):
                x=torch.from_numpy(source[start:start+args.batch_size]).to(args.device)
                a=torch.from_numpy(age[start:start+args.batch_size]).to(args.device)
                pieces.append(model(x, a, a+horizon, gain=args.gain)['forecast'].cpu().numpy())
            forecasts.append(np.concatenate(pieces))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # [N,H,1,512], in the explicitly stored horizon order; no recursive rollout.
    with args.output.open('wb') as f:
        np.savez_compressed(f, source=source, age=age, horizons=np.asarray(args.horizons),
                            forecast=np.stack(forecasts, axis=1), gain=np.asarray(args.gain))
    print(args.output)

if __name__ == '__main__':
    main()
