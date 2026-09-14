"""Convert a trusted original STAGE checkpoint; discard all unrelated weights."""
import argparse
from dataclasses import fields
import hashlib
from pathlib import Path
import torch
from .model import ModelConfig, STAGE


def convert(source, output, *, trusted=False):
    if not trusted:
        raise ValueError('Original training checkpoints require explicit --trust-original approval.')
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    values = checkpoint['config']
    config = {f.name: values[f.name] for f in fields(ModelConfig) if f.name in values}
    # Historical fallback is used only when no explicit adaptive base was set.
    if config.get('adaptive_base_intensity') is None:
        config['adaptive_base_intensity'] = values['proto_intensity']
    model = STAGE(ModelConfig(**config))
    state = {k: v for k, v in checkpoint['model'].items()
             if k.startswith(('backbone.', 'proto.', 'adaptive_intensity.'))}
    model.load_state_dict(state, strict=True)
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    model.save(output, metadata={'source_checkpoint_sha256': digest,
                                'scope': 'backbone, Gaussian memory, Adaptive-global only'})
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--trust-original', action='store_true', help='Only use with your own trusted original checkpoint; pickle can execute code.')
    args = parser.parse_args()
    convert(args.source, args.output, trusted=args.trust_original)
    print(args.output)

if __name__ == '__main__':
    main()
