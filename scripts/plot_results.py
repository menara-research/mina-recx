#!/usr/bin/env python3
"""
High-quality publication plots for MinSTS-Retrieval paper.
Plot 1: Performance comparison (radar-style grouped bars, focused metrics)
Plot 2: Training loss curves (epoch and temperature ablations)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# ─── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'legend.framealpha': 0.92,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'mathtext.fontset': 'stix',
})

RESULTS_DIR = Path('results')

def load_json(path):
    with open(path) as f:
        return json.load(f)

def extract_metrics(d):
    return {
        'mono_r1': d.get('retrieval_monolingual', {}).get('Recall@1', 0),
        'mono_r10': d.get('retrieval_monolingual', {}).get('Recall@10', 0),
        'mono_mrr': d.get('retrieval_monolingual', {}).get('MRR@10', 0),
        'mono_ndcg': d.get('retrieval_monolingual', {}).get('nDCG@10', 0),
        'cross_r1': d.get('retrieval_cross_en', {}).get('Recall@1', 0),
        'cross_r10': d.get('retrieval_cross_en', {}).get('Recall@10', 0),
        'cross_mrr': d.get('retrieval_cross_en', {}).get('MRR@10', 0),
        'sts': d.get('sts', {}).get('Spearman', 0),
        'en_acc': d.get('cross_lingual', {}).get('min_en', {}).get('Accuracy@1', 0),
        'id_acc': d.get('cross_lingual', {}).get('min_id', {}).get('Accuracy@1', 0),
        'cs_cos': d.get('codeswitch', {}).get('avg_cosine_similarity', 0),
        'cs_sp': d.get('codeswitch', {}).get('Spearman', 0),
    }


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Performance Comparison — two-panel
# (a) High-range metrics (bitext, STS, code-switch)
# (b) Low-range metrics (retrieval, zoomed in)
# ═══════════════════════════════════════════════════════════════════════

def plot_performance_comparison():
    # Load baseline model results
    models = [
        ('mE5-small', load_json(RESULTS_DIR / 'baseline_intfloat_multilingual-e5-small.json')),
        ('Indo-e5-small', load_json(RESULTS_DIR / 'baseline_LazarusNLP_all-indo-e5-small-v4.json')),
        ('NusaBERT', load_json(RESULTS_DIR / 'baseline_LazarusNLP_all-NusaBERT-base-v4.json')),
        ('ModernBERT', load_json(RESULTS_DIR / 'baseline_answerdotai_ModernBERT-base.json')),
        ('Jina v5-nano\n(base)', load_json(RESULTS_DIR / 'ablation_baseline.json')),
        ('Minang-Embedder\n(ours)', load_json(RESULTS_DIR / 'ablation_temp_0.2.json')),
    ]
    
    model_names = [m[0] for m in models]
    model_metrics = [extract_metrics(m[1]) for m in models]
    n = len(models)
    
    # Colors: 5 baselines in muted tones, ours in bold orange
    colors = ['#0072B2', '#009E73', '#56B4E9', '#D55E00', '#CC79A7', '#E69F00']
    hatches = ['', '', '//', 'xx', '..', '']
    edge_widths = [0.6, 0.6, 0.6, 0.6, 0.6, 1.4]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), 
                                     gridspec_kw={'width_ratios': [3, 1.2]})
    
    # ─── Panel (a): High-range metrics ───
    metrics_a = [
        ('sts', 'STS ρ'),
        ('en_acc', 'Bitext\n(min→en)'),
        ('id_acc', 'Bitext\n(min→id)'),
        ('cs_cos', 'Code-switch\nCosine'),
    ]
    
    x = np.arange(len(metrics_a))
    width = 0.12
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width
    
    for i in range(n):
        vals = [model_metrics[i][mk] for mk, _ in metrics_a]
        bars = ax1.bar(x + offsets[i], vals, width * 0.88,
                       color=colors[i], edgecolor='#222',
                       linewidth=edge_widths[i], hatch=hatches[i],
                       alpha=0.88, label=model_names[i])
        # Add value labels on top of bars for our model only
        if i == n - 1:  # ours
            for j, v in enumerate(vals):
                ax1.text(x[j] + offsets[i], v + 0.015, f'{v:.2f}',
                        ha='center', va='bottom', fontsize=7.5,
                        fontweight='bold', color='#E69F00')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[1] for m in metrics_a])
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1.08)
    ax1.legend(loc='lower right', ncol=2, fontsize=8.5, handlelength=1.5)
    ax1.set_title('(a) Semantic Similarity & Cross-lingual Transfer', fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.25, linewidth=0.5)
    
    # ─── Panel (b): Retrieval metrics (zoomed) ───
    metrics_b = [
        ('mono_r10', 'Mono.\nR@10'),
        ('mono_mrr', 'Mono.\nMRR@10'),
        ('mono_ndcg', 'Mono.\nnDCG@10'),
    ]
    
    x2 = np.arange(len(metrics_b))
    
    for i in range(n):
        vals = [model_metrics[i][mk] for mk, _ in metrics_b]
        ax2.bar(x2 + offsets[i], vals, width * 0.88,
                color=colors[i], edgecolor='#222',
                linewidth=edge_widths[i], hatch=hatches[i],
                alpha=0.88)
        if i == n - 1:
            for j, v in enumerate(vals):
                ax2.text(x2[j] + offsets[i], v + 0.002, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=6.5,
                        fontweight='bold', color='#E69F00')
    
    ax2.set_xticks(x2)
    ax2.set_xticklabels([m[1] for m in metrics_b])
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 0.075)
    ax2.set_title('(b) Monolingual Retrieval\n(zoomed)', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.25, linewidth=0.5)
    
    fig.suptitle('MinSTS-Retrieval Benchmark: Model Comparison', 
                 fontsize=15, fontweight='bold', y=1.03)
    fig.tight_layout()
    fig.savefig('figures/performance_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig('figures/performance_comparison.pdf', bbox_inches='tight')
    print("Saved: figures/performance_comparison.png/pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Ablation Study — 2×2 grid
# ═══════════════════════════════════════════════════════════════════════

def plot_ablation_results():
    epoch_runs = {}
    for e in [3, 5, 7, 10]:
        epoch_runs[e] = extract_metrics(load_json(RESULTS_DIR / f'ablation_epochs_{e}.json'))
    
    temp_runs = {}
    for t in ['0.02', '0.1', '0.2']:
        temp_runs[float(t)] = extract_metrics(load_json(RESULTS_DIR / f'ablation_temp_{t}.json'))
    temp_runs[0.05] = extract_metrics(load_json(RESULTS_DIR / 'ablation_epochs_5.json'))
    
    baseline = extract_metrics(load_json(RESULTS_DIR / 'ablation_baseline.json'))
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    
    # ─── (a) Epoch ablation: STS + Cross-lingual ───
    ax = axes[0, 0]
    epochs = sorted(epoch_runs.keys())
    
    l1, = ax.plot(epochs, [epoch_runs[e]['sts'] for e in epochs], 
                  'o-', color='#E69F00', lw=2.2, ms=8, label='STS Spearman ρ', zorder=5)
    l2, = ax.plot(epochs, [epoch_runs[e]['en_acc'] for e in epochs],
                  's--', color='#0072B2', lw=2, ms=7, label='Bitext Acc@1 (min→en)')
    l3, = ax.plot(epochs, [epoch_runs[e]['id_acc'] for e in epochs],
                  '^--', color='#009E73', lw=2, ms=7, label='Bitext Acc@1 (min→id)')
    
    # Baseline references
    ax.axhline(baseline['sts'], color='#E69F00', ls=':', alpha=0.4, lw=1.2)
    ax.axhline(baseline['en_acc'], color='#0072B2', ls=':', alpha=0.4, lw=1.2)
    ax.axhline(baseline['id_acc'], color='#009E73', ls=':', alpha=0.4, lw=1.2)
    ax.text(10.3, baseline['sts'], 'base', fontsize=7, color='#E69F00', alpha=0.6, va='center')
    ax.text(10.3, baseline['en_acc'], 'base', fontsize=7, color='#0072B2', alpha=0.6, va='center')
    
    # Annotate the tradeoff
    ax.annotate('↗ improves', xy=(8, 0.79), fontsize=8, color='#E69F00', alpha=0.7)
    ax.annotate('↘ degrades', xy=(8, 0.82), fontsize=8, color='#0072B2', alpha=0.7)
    
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Score')
    ax.set_title('(a) Epoch Ablation: STS vs. Cross-lingual', fontweight='bold')
    ax.legend(fontsize=8, loc='center right')
    ax.set_xticks(epochs)
    ax.set_ylim(0.45, 1.02)
    ax.grid(alpha=0.2, lw=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ─── (b) Epoch ablation: Retrieval + Code-switching ───
    ax = axes[0, 1]
    
    ax.plot(epochs, [epoch_runs[e]['mono_r10'] for e in epochs],
            'o-', color='#D55E00', lw=2.2, ms=8, label='Mono. R@10')
    ax2t = ax.twinx()
    ax2t.plot(epochs, [epoch_runs[e]['cs_cos'] for e in epochs],
              's--', color='#56B4E9', lw=2, ms=7, label='Code-switch Cosine')
    
    ax.axhline(baseline['mono_r10'], color='#D55E00', ls=':', alpha=0.4, lw=1.2)
    ax2t.axhline(baseline['cs_cos'], color='#56B4E9', ls=':', alpha=0.4, lw=1.2)
    
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Mono. R@10', color='#D55E00')
    ax2t.set_ylabel('Code-switch Cosine', color='#56B4E9')
    ax.tick_params(axis='y', labelcolor='#D55E00')
    ax2t.tick_params(axis='y', labelcolor='#56B4E9')
    ax.set_title('(b) Epoch Ablation: Retrieval & Code-switch', fontweight='bold')
    ax.set_xticks(epochs)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2t.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')
    
    ax.set_ylim(0.03, 0.055)
    ax2t.set_ylim(0.55, 0.85)
    ax.grid(alpha=0.2, lw=0.5)
    ax.spines['top'].set_visible(False)
    
    # ─── (c) Temperature ablation: STS + Cross-lingual ───
    ax = axes[1, 0]
    temps = sorted(temp_runs.keys())
    
    ax.plot(temps, [temp_runs[t]['sts'] for t in temps],
            'o-', color='#E69F00', lw=2.2, ms=8, label='STS Spearman ρ', zorder=5)
    ax.plot(temps, [temp_runs[t]['en_acc'] for t in temps],
            's--', color='#0072B2', lw=2, ms=7, label='Bitext Acc@1 (min→en)')
    ax.plot(temps, [temp_runs[t]['id_acc'] for t in temps],
            '^--', color='#009E73', lw=2, ms=7, label='Bitext Acc@1 (min→id)')
    
    ax.axhline(baseline['sts'], color='#E69F00', ls=':', alpha=0.4, lw=1.2)
    ax.axhline(baseline['en_acc'], color='#0072B2', ls=':', alpha=0.4, lw=1.2)
    ax.axhline(baseline['id_acc'], color='#009E73', ls=':', alpha=0.4, lw=1.2)
    
    # Highlight sweet spot
    ax.axvspan(0.1, 0.25, alpha=0.08, color='#E69F00', label='_sweet spot')
    ax.text(0.17, 0.48, 'sweet\nspot', fontsize=8, color='#E69F00', 
            ha='center', style='italic', alpha=0.8)
    
    ax.set_xlabel('Temperature (τ)')
    ax.set_ylabel('Score')
    ax.set_title('(c) Temperature Ablation: STS vs. Cross-lingual', fontweight='bold')
    ax.legend(fontsize=8, loc='center right')
    ax.set_xticks(temps)
    ax.set_xticklabels([f'{t:.2f}' for t in temps])
    ax.set_ylim(0.45, 1.02)
    ax.grid(alpha=0.2, lw=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ─── (d) Temperature ablation: Retrieval + Code-switching ───
    ax = axes[1, 1]
    
    ax.plot(temps, [temp_runs[t]['mono_r10'] for t in temps],
            'o-', color='#D55E00', lw=2.2, ms=8, label='Mono. R@10')
    ax2t = ax.twinx()
    ax2t.plot(temps, [temp_runs[t]['cs_cos'] for t in temps],
              's--', color='#56B4E9', lw=2, ms=7, label='Code-switch Cosine')
    
    ax.axhline(baseline['mono_r10'], color='#D55E00', ls=':', alpha=0.4, lw=1.2)
    ax2t.axhline(baseline['cs_cos'], color='#56B4E9', ls=':', alpha=0.4, lw=1.2)
    
    ax.axvspan(0.1, 0.25, alpha=0.08, color='#D55E00')
    
    ax.set_xlabel('Temperature (τ)')
    ax.set_ylabel('Mono. R@10', color='#D55E00')
    ax2t.set_ylabel('Code-switch Cosine', color='#56B4E9')
    ax.tick_params(axis='y', labelcolor='#D55E00')
    ax2t.tick_params(axis='y', labelcolor='#56B4E9')
    ax.set_title('(d) Temperature Ablation: Retrieval & Code-switch', fontweight='bold')
    ax.set_xticks(temps)
    ax.set_xticklabels([f'{t:.2f}' for t in temps])
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2t.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')
    
    ax.set_ylim(0.03, 0.06)
    ax2t.set_ylim(0.55, 0.95)
    ax.grid(alpha=0.2, lw=0.5)
    ax.spines['top'].set_visible(False)
    
    fig.suptitle('Ablation Study: Training Epochs & Contrastive Temperature', 
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('figures/ablation_study.png', dpi=300, bbox_inches='tight')
    fig.savefig('figures/ablation_study.pdf', bbox_inches='tight')
    print("Saved: figures/ablation_study.png/pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Training Loss Curves
# ═══════════════════════════════════════════════════════════════════════

def plot_training_loss():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    
    # ─── (a) Loss by epoch count ───
    ax = axes[0]
    epoch_colors = {3: '#0072B2', 5: '#009E73', 7: '#E69F00', 10: '#D55E00'}
    
    for epochs in [3, 5, 7, 10]:
        lh = load_json(f'models/ablations/epochs_{epochs}/loss_history.json')
        steps = np.array(lh['step_losses']['steps'])
        losses = np.array(lh['step_losses']['losses'])
        
        # Smooth with EMA
        alpha = 0.15
        smooth = np.zeros_like(losses)
        smooth[0] = losses[0]
        for i in range(1, len(losses)):
            smooth[i] = alpha * losses[i] + (1 - alpha) * smooth[i-1]
        
        ax.plot(steps, smooth, lw=2.2, color=epoch_colors[epochs],
                label=f'{epochs} epochs', alpha=0.9)
        
        # Mark epoch boundaries
        spe = lh['config']['steps_per_epoch']
        for ep in range(1, epochs + 1):
            step_ep = ep * spe
            if step_ep <= steps[-1]:
                ax.axvline(x=step_ep, color=epoch_colors[epochs], 
                          ls=':', alpha=0.15, lw=0.8)
    
    # Also plot raw for 10-epoch (faint)
    lh10 = load_json('models/ablations/epochs_10/loss_history.json')
    steps10 = np.array(lh10['step_losses']['steps'])
    losses10 = np.array(lh10['step_losses']['losses'])
    ax.plot(steps10, losses10, lw=0.4, color='#D55E00', alpha=0.3, label='_raw')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('MNR Loss (EMA smoothed)')
    ax.set_title('(a) Loss Curves by Epoch Count', fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.2, lw=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(losses10) * 1.05)
    
    # ─── (b) Loss by temperature ───
    ax = axes[1]
    temp_data = [
        ('0.02', '#0072B2', 'τ=0.02'),
        ('0.05', '#009E73', 'τ=0.05'),
        ('0.1', '#E69F00', 'τ=0.10'),
        ('0.2', '#D55E00', 'τ=0.20'),
    ]
    
    for temp_str, color, label in temp_data:
        if temp_str == '0.05':
            lh = load_json('models/ablations/epochs_5/loss_history.json')
        else:
            lh = load_json(f'models/ablations/temp_{temp_str}/loss_history.json')
        steps = np.array(lh['step_losses']['steps'])
        losses = np.array(lh['step_losses']['losses'])
        
        alpha_ema = 0.15
        smooth = np.zeros_like(losses)
        smooth[0] = losses[0]
        for i in range(1, len(losses)):
            smooth[i] = alpha_ema * losses[i] + (1 - alpha_ema) * smooth[i-1]
        
        ax.plot(steps, smooth, lw=2.2, color=color, label=label, alpha=0.9)
        
        # Mark 5-epoch boundary
        spe = lh['config']['steps_per_epoch']
        ax.axvline(x=5*spe, color=color, ls=':', alpha=0.15, lw=0.8)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('MNR Loss (EMA smoothed)')
    ax.set_title('(b) Loss Curves by Temperature (τ)', fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.2, lw=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.suptitle('Training Dynamics: MultipleNegativesRankingLoss', 
                 fontsize=14, fontweight='bold', y=1.04)
    fig.tight_layout()
    fig.savefig('figures/training_loss.png', dpi=300, bbox_inches='tight')
    fig.savefig('figures/training_loss.pdf', bbox_inches='tight')
    print("Saved: figures/training_loss.png/pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Summary heatmap — all models × all metrics
# ═══════════════════════════════════════════════════════════════════════

def plot_summary_heatmap():
    """Heatmap comparing all models across all metrics."""
    
    models_order = [
        ('mE5-small', 'baseline_intfloat_multilingual-e5-small.json'),
        ('Indo-e5-small', 'baseline_LazarusNLP_all-indo-e5-small-v4.json'),
        ('NusaBERT', 'baseline_LazarusNLP_all-NusaBERT-base-v4.json'),
        ('ModernBERT', 'baseline_answerdotai_ModernBERT-base.json'),
        ('Jina v5-nano (base)', 'ablation_baseline.json'),
        ('Ours (T=0.2, 5ep)', 'ablation_temp_0.2.json'),
    ]
    
    metrics_order = [
        ('mono_r10', 'Mono. R@10'),
        ('cross_r10', 'Cross R@10'),
        ('sts', 'STS ρ'),
        ('en_acc', 'Bitext (en)'),
        ('id_acc', 'Bitext (id)'),
        ('cs_cos', 'Code-switch'),
    ]
    
    data = np.zeros((len(models_order), len(metrics_order)))
    for i, (_, path) in enumerate(models_order):
        m = extract_metrics(load_json(RESULTS_DIR / path))
        for j, (mk, _) in enumerate(metrics_order):
            data[i, j] = m[mk]
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(range(len(metrics_order)))
    ax.set_xticklabels([m[1] for m in metrics_order], fontsize=10)
    ax.set_yticks(range(len(models_order)))
    ax.set_yticklabels([m[0] for m in models_order], fontsize=10)
    
    # Annotate cells
    for i in range(len(models_order)):
        for j in range(len(metrics_order)):
            val = data[i, j]
            text_color = 'white' if val > 0.7 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   fontsize=9, color=text_color, fontweight='bold' if i == len(models_order)-1 else 'normal')
    
    # Highlight our model row
    ax.axhline(y=len(models_order)-1.5, color='#E69F00', lw=3)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Score', fontsize=11)
    
    ax.set_title('MinSTS-Retrieval: Cross-Model Performance Heatmap', 
                 fontsize=14, fontweight='bold', pad=12)
    
    fig.tight_layout()
    fig.savefig('figures/performance_heatmap.png', dpi=300, bbox_inches='tight')
    fig.savefig('figures/performance_heatmap.pdf', bbox_inches='tight')
    print("Saved: figures/performance_heatmap.png/pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════

Path('figures').mkdir(exist_ok=True)

print("Generating plots...")
plot_performance_comparison()
plot_ablation_results()
plot_training_loss()
plot_summary_heatmap()
print("Done!")
