import os
import glob
import json
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Path & Configuration
# ==========================================
DATASET_DIR = r"C:\Users\USER\VScode\neonate\dataset\RE_baby"
SAVE_DIR = r"D:\neonate\Test8_final_media_ref_pp_color"
CACHE_DIR = r"D:\neonate\Test8_final_media_ref_pp_color\.cache"

FIXED_THRESHOLD = 0.9

# ==========================================
# 2. 6-Fold Specifications (dsp_6fold_spec.md)
# ==========================================
"""
OLD 5-FOLD SPLIT LOGIC (PRESERVED AS COMMENT PER USER REQUEST):
# def pure_numpy_kfold_split(n_samples, n_splits=5, seed=42):
#     np.random.seed(seed)
#     indices = np.arange(n_samples)
#     np.random.shuffle(indices)
#     folds = np.array_split(indices, n_splits)
#     splits = []
#     for i in range(n_splits):
#         test_idx = folds[i]
#         train_idx = np.hstack([folds[j] for j in range(n_splits) if j != i])
#         splits.append((train_idx, test_idx))
#     return splits
"""

# DSP Official 6-Fold Split (dsp_6fold_spec.md)
FOLD_TEST = {
    1: ["S01", "S02"],
    2: ["S05", "S11"],
    3: ["S06", "S08"],
    4: ["S04", "S09", "S12"],
    5: ["S03", "S07", "S13"],
    6: ["S10", "S14", "S15"],
}

FOLD_VALID = {
    1: ["S03", "S12"],
    2: ["S13", "S15"],
    3: ["S09", "S10"],
    4: ["S01", "S08"],
    5: ["S04", "S11"],
    6: ["S02", "S07"],
}

ALL_SUBJECTS = [f"S{i:02d}" for i in range(1, 16)]

def extract_main_subject_id(subject_folder_name):
    match = re.match(r'^(S\d+)', subject_folder_name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return subject_folder_name

def calculate_icc_31(y_true, y_pred):
    n = len(y_true)
    if n < 2:
        return 0.0
    Y = np.vstack([y_true, y_pred]).T
    mean_target = np.mean(Y, axis=1)
    grand_mean = np.mean(Y)
    
    ss_total = np.sum((Y - grand_mean) ** 2)
    ss_between = 2 * np.sum((mean_target - grand_mean) ** 2)
    ss_within = ss_total - ss_between
    
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / n
    
    icc31 = (ms_between - ms_within) / (ms_between + ms_within + 1e-8)
    return max(0.0, min(1.0, float(icc31)))

def save_csv_robust(csv_path, headers, rows, summary_stats_dict=None):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        if summary_stats_dict:
            f.write('\n')
            f.write('# SUMMARY STATISTICS (OVERALL OUT-OF-FOLD TEST)\n')
            for k, v in summary_stats_dict.items():
                f.write(f'# {k},{v}\n')

def main():
    print("=" * 90)
    print("      DSP Official 6-Fold Subject-Wise Cross-Validation Pipeline (dsp_6fold_spec.md)")
    print("=" * 90)
    
    # 1. Load Caches
    cache_files = glob.glob(os.path.join(CACHE_DIR, "*_sweep_stats_3x3.json"))
    print(f"Found {len(cache_files)} cache JSON files in {CACHE_DIR}")
    
    results = []
    for cf in cache_files:
        try:
            with open(cf, 'r', encoding='utf-8') as fp:
                d = json.load(fp)
            if d.get('status') == 'success':
                results.append(d)
        except Exception:
            pass
            
    valid_results = [r for r in results if r.get('status') == 'success']
    print(f"Loaded {len(valid_results)} successful video evaluations out of {len(cache_files)}.\n")
    
    for r in valid_results:
        r['main_subject'] = extract_main_subject_id(r['subject'])
        
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    th_str = str(FIXED_THRESHOLD)
    
    fold_summaries = []
    out_of_fold_test_predictions = []
    
    print("=" * 90)
    print("EXECUTING 6-FOLD SUBJECT-WISE CROSS-VALIDATION")
    print("  -> Hyperparameter tuning on Train + Valid (13 subjects)")
    print("  -> Unbiased Evaluation on Held-out Test (2~3 subjects)")
    print("=" * 90 + "\n")

    """
    Dataset 분리해서 처리하는 코드
    """
    for fold in range(1, 7):
        test_subs = FOLD_TEST[fold]
        valid_subs = FOLD_VALID[fold]
        train_subs = sorted(set(ALL_SUBJECTS) - set(test_subs) - set(valid_subs))
        
        # dsp_6fold_spec.md Directive 2: Tuning done on Train + Valid (13 subjects)
        tuning_subs = train_subs + valid_subs
        
        tuning_samples = [r for r in valid_results if r['main_subject'] in tuning_subs]
        test_samples = [r for r in valid_results if r['main_subject'] in test_subs]
        
        # 1. Hyperparameter Search (Weight_Flow) on Train+Valid 13 Subjects
        best_tuning_config = None
        best_tuning_mae = float('inf')
        
        for w_flow in weights:
            w_key = f"w_{w_flow:.1f}"
            t_errors = []
            
            for r in tuning_samples:
                th_res = r['threshold_results'].get(th_str) or r['threshold_results'].get(float(FIXED_THRESHOLD))
                if th_res and th_res['status'] == 'success':
                    k_res_dict = th_res['k_results'].get(w_key)
                    if k_res_dict:
                        best_k_key = str(min([int(k) for k in k_res_dict.keys()]))
                        k_res = k_res_dict[best_k_key]
                        t_errors.append(abs(k_res['est_bpm'] - r['gt_bpm']))
                        
            if len(t_errors) > 0:
                avg_t_mae = np.mean(t_errors)
                if avg_t_mae < best_tuning_mae:
                    best_tuning_mae = avg_t_mae
                    best_tuning_config = {'w_flow': w_flow, 'w_color': round(1.0 - w_flow, 1)}
                    
        if best_tuning_config is None:
            best_tuning_config = {'w_flow': 0.9, 'w_color': 0.1}
            
        best_w_key = f"w_{best_tuning_config['w_flow']:.1f}"
        
        # 2. Evaluate Held-out Test Set (2~3 Subjects)
        test_est_bpms = []
        test_gt_bpms = []
        test_snrs = []
        test_selected_cells = []
        
        for r in test_samples:
            th_res = r['threshold_results'].get(th_str) or r['threshold_results'].get(float(FIXED_THRESHOLD))
            if th_res and th_res['status'] == 'success':
                k_res_dict = th_res['k_results'].get(best_w_key)
                if k_res_dict:
                    best_k_key = str(min([int(k) for k in k_res_dict.keys()]))
                    k_res = k_res_dict[best_k_key]
                    
                    est_bpm = k_res['est_bpm']
                    gt_bpm = r['gt_bpm']
                    snr_val = k_res['snr']
                    valid_top_cells = th_res['top_cells']
                    
                    test_est_bpms.append(est_bpm)
                    test_gt_bpms.append(gt_bpm)
                    if not np.isnan(snr_val):
                        test_snrs.append(snr_val)
                    test_selected_cells.append(len(valid_top_cells))
                    
                    out_of_fold_test_predictions.append({
                        'fold': fold,
                        'sample_id': r['sample_id'],
                        'video': r['video'],
                        'subject': r['subject'],
                        'main_subject': r['main_subject'],
                        'posture': r['posture'],
                        'selected_w_flow': best_tuning_config['w_flow'],
                        'selected_w_color': best_tuning_config['w_color'],
                        'threshold': FIXED_THRESHOLD,
                        'selected_cells_count': len(valid_top_cells),
                        'est_bpm': est_bpm,
                        'gt_bpm': gt_bpm,
                        'error': abs(est_bpm - gt_bpm),
                        'snr': snr_val
                    })
                    
        test_errors = np.abs(np.array(test_est_bpms) - np.array(test_gt_bpms))
        test_mae = np.mean(test_errors)
        test_rmse = np.sqrt(np.mean(test_errors**2))
        test_pcc = np.corrcoef(test_est_bpms, test_gt_bpms)[0, 1] if len(test_est_bpms) > 1 and np.std(test_est_bpms) > 0 and np.std(test_gt_bpms) > 0 else 0.0
        test_icc = calculate_icc_31(test_est_bpms, test_gt_bpms)
        test_sr3 = np.mean(test_errors <= 3.0) * 100.0
        test_sr5 = np.mean(test_errors <= 5.0) * 100.0
        test_avg_snr = np.mean(test_snrs) if len(test_snrs) > 0 else 0.0
        test_avg_cells = np.mean(test_selected_cells) if len(test_selected_cells) > 0 else 0.0
        
        fold_summaries.append({
            'Fold': fold,
            'Test_Subjects': ", ".join(test_subs),
            'Valid_Subjects': ", ".join(valid_subs),
            'Tuning_Samples': len(tuning_samples),
            'Test_Samples': len(test_samples),
            'Selected_W_Flow': best_tuning_config['w_flow'],
            'Selected_W_Color': best_tuning_config['w_color'],
            'Threshold': FIXED_THRESHOLD,
            'Avg_Selected_Cells': round(test_avg_cells, 2),
            'Tuning_MAE': round(best_tuning_mae, 4),
            'Test_MAE': round(test_mae, 4),
            'Test_RMSE': round(test_rmse, 4),
            'Test_PCC': round(test_pcc, 4),
            'Test_ICC': round(test_icc, 4),
            'Test_Avg_SNR': round(test_avg_snr, 2),
            'Test_SR_3BPM': f"{round(test_sr3, 2)}%",
            'Test_SR_5BPM': f"{round(test_sr5, 2)}%"
        })
        
        print(f"Fold {fold}/6 | Test: {test_subs} ({len(test_samples)} clips) | Valid: {valid_subs}")
        print(f"  -> Best Tuning Config (13 subs): W_Flow={best_tuning_config['w_flow']} | Threshold={FIXED_THRESHOLD} (Tuning MAE: {best_tuning_mae:.4f} BPM)")
        print(f"  -> Unbiased Test Performance   : MAE = {test_mae:.4f} BPM | RMSE = {test_rmse:.4f} | PCC = {test_pcc:.4f} | SR(<5BPM) = {test_sr5:.2f}%\n")
        
    # Overall Out-of-Fold Aggregated Test Metrics
    oof_est_bpms = np.array([p['est_bpm'] for p in out_of_fold_test_predictions])
    oof_gt_bpms = np.array([p['gt_bpm'] for p in out_of_fold_test_predictions])
    oof_errors = np.abs(oof_est_bpms - oof_gt_bpms)
    
    oof_mae = np.mean(oof_errors)
    oof_rmse = np.sqrt(np.mean(oof_errors**2))
    oof_pcc = np.corrcoef(oof_est_bpms, oof_gt_bpms)[0, 1] if len(oof_est_bpms) > 1 and np.std(oof_est_bpms) > 0 and np.std(oof_gt_bpms) > 0 else 0.0
    oof_icc = calculate_icc_31(oof_est_bpms, oof_gt_bpms)
    oof_sr3 = np.mean(oof_errors <= 3.0) * 100.0
    oof_sr5 = np.mean(oof_errors <= 5.0) * 100.0
    valid_oof_snrs = [p['snr'] for p in out_of_fold_test_predictions if not np.isnan(p['snr'])]
    oof_avg_snr = np.mean(valid_oof_snrs) if len(valid_oof_snrs) > 0 else 0.0
    oof_avg_cells = np.mean([p['selected_cells_count'] for p in out_of_fold_test_predictions])
    
    # Save CSVs
    fold_summary_csv = os.path.join(SAVE_DIR, "kfold_6fold_cross_validation_folds_summary.csv")
    fold_headers = ['Fold', 'Test_Subjects', 'Valid_Subjects', 'Tuning_Samples', 'Test_Samples', 'Selected_W_Flow', 'Selected_W_Color', 'Threshold', 'Avg_Selected_Cells', 'Tuning_MAE', 'Test_MAE', 'Test_RMSE', 'Test_PCC', 'Test_ICC', 'Test_Avg_SNR', 'Test_SR_3BPM', 'Test_SR_5BPM']
    save_csv_robust(fold_summary_csv, fold_headers, fold_summaries)
    
    oof_predictions_csv = os.path.join(SAVE_DIR, "kfold_6fold_out_of_fold_predictions.csv")
    oof_headers = ['fold', 'sample_id', 'video', 'subject', 'main_subject', 'posture', 'selected_w_flow', 'selected_w_color', 'threshold', 'selected_cells_count', 'est_bpm', 'gt_bpm', 'error', 'snr']
    oof_stats_dict = {
        'Overall 6-Fold Out-of-Fold MAE': round(oof_mae, 4),
        'Overall 6-Fold Out-of-Fold RMSE': round(oof_rmse, 4),
        'Overall 6-Fold Out-of-Fold PCC': round(oof_pcc, 4),
        'Overall 6-Fold Out-of-Fold ICC': round(oof_icc, 4),
        'Overall Avg Selected Cells': round(oof_avg_cells, 2),
        'Success Rate (<3.0 BPM)': f"{round(oof_sr3, 2)}%",
        'Success Rate (<5.0 BPM)': f"{round(oof_sr5, 2)}%",
        'Average SNR': f"{round(oof_avg_snr, 2)} dB"
    }
    save_csv_robust(oof_predictions_csv, oof_headers, out_of_fold_test_predictions, oof_stats_dict)
    
    # Generate 6-Fold Plots (Scatter & Bland-Altman)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    # Scatter Plot
    ax1.scatter(oof_gt_bpms, oof_est_bpms, alpha=0.6, color='#1f77b4', edgecolors='k', linewidth=0.5, s=35, label='OOF Test Predictions (N=374)')
    min_bpm = float(min(np.min(oof_gt_bpms), np.min(oof_est_bpms)) - 2)
    max_bpm = float(max(np.max(oof_gt_bpms), np.max(oof_est_bpms)) + 2)
    ax1.plot([min_bpm, max_bpm], [min_bpm, max_bpm], 'r--', linewidth=1.5, label='Identity Line (y=x)')
    
    cov = np.cov(oof_gt_bpms, oof_est_bpms)[0, 1]
    var_x = np.var(oof_gt_bpms, ddof=1)
    slope = float(cov / var_x)
    intercept = float(np.mean(oof_est_bpms) - slope * np.mean(oof_gt_bpms))
    ax1.plot(np.array([min_bpm, max_bpm]), slope * np.array([min_bpm, max_bpm]) + intercept, color='#2ca02c', linewidth=1.5, label=f'Linear Fit (y = {slope:.2f}x + {intercept:.2f})')
    
    ax1.set_xlim(min_bpm, max_bpm)
    ax1.set_ylim(min_bpm, max_bpm)
    ax1.set_title("(a) Scatter Plot of Ground Truth vs. Estimated RR", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Ground Truth RR (BPM)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Estimated RR (BPM)", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    text_str = (f'MAE  = {oof_mae:.4f} BPM\n'
                f'RMSE = {oof_rmse:.4f} BPM\n'
                f'PCC  = {oof_pcc:.4f}')
    ax1.text(0.04, 0.95, text_str, transform=ax1.transAxes, fontsize=8.5,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='square,pad=0.6', facecolor='white', edgecolor='#cccccc', alpha=0.9, linewidth=1.0),
             fontfamily='monospace')
    
    ax1.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    
    # Bland-Altman Plot
    diffs = oof_est_bpms - oof_gt_bpms
    means = (oof_est_bpms + oof_gt_bpms) / 2.0
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    
    ax2.scatter(means, diffs, alpha=0.6, color='#9467bd', edgecolors='k', linewidth=0.5, s=35, label='OOF Test Predictions (N=374)')
    ax2.axhline(mean_diff, color='#1f77b4', linestyle='-', linewidth=2.0, label=f'Mean Bias: {mean_diff:.4f} BPM')
    ax2.axhline(upper_loa, color='#d62728', linestyle='--', linewidth=1.5, label=f'+1.96 SD: {upper_loa:.4f} BPM')
    ax2.axhline(lower_loa, color='#d62728', linestyle='--', linewidth=1.5, label=f'-1.96 SD: {lower_loa:.4f} BPM')
    ax2.axhline(0, color='gray', linestyle=':', linewidth=1.0)
    
    ax2.set_title("(b) Bland-Altman Plot of Differences vs. Means", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Mean of GT and Estimated RR (BPM)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Difference (Est - GT) (BPM)", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, "kfold_6fold_cv_performance_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("=" * 90)
    print("     FINAL DSP OFFICIAL 6-FOLD CROSS-VALIDATION SUMMARY REPORT (FOR PAPER)")
    print("=" * 90)
    print(f"  Overall 6-Fold Out-of-Fold MAE   : {oof_mae:.4f} BPM")
    print(f"  Overall 6-Fold Out-of-Fold RMSE  : {oof_rmse:.4f} BPM")
    print(f"  Overall 6-Fold Out-of-Fold PCC   : {oof_pcc:.4f}")
    print(f"  Overall 6-Fold Out-of-Fold ICC   : {oof_icc:.4f}")
    print(f"  Overall Avg Selected Cells       : {oof_avg_cells:.2f} cells")
    print(f"  Success Rate (< 3.0 BPM)         : {oof_sr3:.2f}%")
    print(f"  Success Rate (< 5.0 BPM)         : {oof_sr5:.2f}%")
    print(f"  Average SNR                      : {oof_avg_snr:.2f} dB")
    print("-" * 90)
    
    # Posture-wise Breakdown Analysis for 6-Fold
    print("\n" + "=" * 90)
    print("       6-FOLD POSTURE-WISE PERFORMANCE BREAKDOWN (SUPINE / SIDE / PRONE)")
    print("=" * 90)
    posture_rows = []
    df_oof = pd.DataFrame(out_of_fold_test_predictions)
    for p_name in ['supine', 'side', 'prone']:
        p_sub = df_oof[df_oof['posture'] == p_name]
        if len(p_sub) > 0:
            p_n = len(p_sub)
            p_err = np.abs(p_sub['est_bpm'] - p_sub['gt_bpm'])
            p_mae = np.mean(p_err)
            p_rmse = np.sqrt(np.mean(p_err**2))
            p_est = p_sub['est_bpm'].values
            p_gt = p_sub['gt_bpm'].values
            p_pcc = np.corrcoef(p_est, p_gt)[0, 1] if p_n > 1 and np.std(p_est) > 0 and np.std(p_gt) > 0 else 0.0
            p_sr3 = np.mean(p_err <= 3.0) * 100.0
            p_sr5 = np.mean(p_err <= 5.0) * 100.0
            print(f"  [{p_name.capitalize():<6}] Count: {p_n:>3} ({p_n/len(df_oof)*100:>4.1f}%) | MAE: {p_mae:.4f} BPM | RMSE: {p_rmse:.4f} | PCC: {p_pcc:.4f} | SR(<3.0): {p_sr3:.2f}% | SR(<5.0): {p_sr5:.2f}%")
            posture_rows.append({
                'Posture': p_name.capitalize(),
                'Count': p_n,
                'Ratio': f"{p_n/len(df_oof)*100:.1f}%",
                'MAE_BPM': round(p_mae, 4),
                'RMSE': round(p_rmse, 4),
                'PCC': round(p_pcc, 4),
                'SR_3BPM': f"{round(p_sr3, 2)}%",
                'SR_5BPM': f"{round(p_sr5, 2)}%"
            })
    posture_csv = os.path.join(SAVE_DIR, "kfold_6fold_posture_wise_summary.csv")
    pd.DataFrame(posture_rows).to_csv(posture_csv, index=False, encoding='utf-8-sig')

    print("-" * 90)
    print(f" -> Saved 6-Fold Summaries to         : {fold_summary_csv}")
    print(f" -> Saved 6-Fold Out-of-Fold CSV to   : {oof_predictions_csv}")
    print(f" -> Saved 6-Fold Posture Breakdown to : {posture_csv}")
    print(f" -> Saved 6-Fold Paper Figures to     : {plot_path}")
    print("=" * 90)

if __name__ == "__main__":
    main()
