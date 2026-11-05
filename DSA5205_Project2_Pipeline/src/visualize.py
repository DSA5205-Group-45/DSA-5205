from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def visualize(screened, audit, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    audit.sort_values('decision_date').plot(x='decision_date', y=['eligible_top300','selected'], ax=ax)
    ax.set_title('Eligible universe and selected candidates')
    ax.set_ylabel('Stocks')
    fig.tight_layout(); fig.savefig(outdir/'screening_coverage.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    screened.groupby(['momentum_group','volatility_group']).size().unstack(fill_value=0).plot(kind='bar', ax=ax)
    ax.set_title('Selected stocks by momentum / volatility bucket')
    fig.tight_layout(); fig.savefig(outdir/'bucket_counts.png', dpi=160); plt.close(fig)
    if screened['target_forward_return'].notna().any():
        fig, ax = plt.subplots(figsize=(7,4))
        screened['target_forward_return'].dropna().hist(bins=55, ax=ax)
        ax.set_title('Future holding-period returns (decimal)')
        fig.tight_layout(); fig.savefig(outdir/'forward_return_distribution.png', dpi=160); plt.close(fig)
