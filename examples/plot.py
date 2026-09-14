"""Plot Source and forecasts from stage.infer output, without changing amplitudes."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_file(input_path, output_path):
    with np.load(input_path, allow_pickle=False) as data:
        source, forecast = data['source'], data['forecast']
        horizons = data['horizons']
    if (source.ndim != 3 or source.shape[1:] != (1, 512)
            or forecast.shape != (len(source), len(horizons), 1, 512)
            or not np.isfinite(source).all() or not np.isfinite(forecast).all()):
        raise ValueError('Expected finite Source [N,1,512] and forecast [N,H,1,512].')
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    time = (np.arange(16, 496) - 208) * 2.5
    colors = ['#4472C4', '#4C956C', '#E88A37']
    for i in range(len(source)):
        fig, ax = plt.subplots(figsize=(7.4, 3.3))
        ax.plot(time, source[i, 0, 16:-16], color='#333333', lw=1.6, label='Source')
        for j, h in enumerate(horizons):
            ax.plot(time, forecast[i, j, 0, 16:-16], color=colors[j % len(colors)],
                    lw=1.3, label=f'G{h:g}')
        ax.set(xlabel='Time from R peak (ms)', ylabel='Amplitude (mV)',
               xlim=(time[0], time[-1]))
        ax.set_title(f'Case {i+1:02d}', loc='left', fontsize=11)
        ax.legend(loc='lower right', bbox_to_anchor=(1, 1.01),
                  ncol=len(horizons)+1, frameon=False, fontsize=9, borderaxespad=0)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(output_path / f'case_{i+1:02d}.png', dpi=200)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot_file(args.input, args.output)
