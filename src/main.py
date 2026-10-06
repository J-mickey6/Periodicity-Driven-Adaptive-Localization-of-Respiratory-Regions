import cv2
import numpy as np
import os
import glob
import re
import sys
import multiprocessing
import csv
import json
import time
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, resample, find_peaks
import h5py
from tqdm import tqdm
from scipy.signal import detrend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as patches

# ==========================================
# 1. Path & Configuration
# ==========================================
DATASET_DIR = r"C:\Users\USER\VScode\neonate\dataset\RE_baby"
SAVE_DIR = r"D:\neonate\Test8_final_media_ref_pp_color"
CACHE_DIR = r"D:\neonate\Test8_final_media_ref_pp_color\.cache"
VISUAL_DIR = r"D:\neonate\Test8_final_media_ref_pp_color\visualizations"

GRID_SIZE = (32, 18)
BREATHING_BAND = (0.2, 0.8)
SCALE_FACTOR = 0.2
MAX_ALLOWED_VARIATION = 5.0
TOP_K_CELLS = 8
NUM_WORKERS = 2  # Optimized for multi-core CPU

# ==========================================
# 2. Helper Functions
# ==========================================

def get_all_video_gt_pairs(dataset_dir):
    pairs = []
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith('.mp4'):
                v = os.path.join(root, f)
                folder = os.path.dirname(v)
                filename = os.path.splitext(f)[0]
                gt_path = os.path.join(folder, "out", f"{filename}.hdf5")
                if os.path.exists(gt_path):
                    pairs.append((os.path.normpath(v), os.path.normpath(gt_path)))
    pairs = sorted(pairs, key=lambda x: x[0])
    return pairs

def bandpass(sig, fs, low, high):
    nyq = 0.5 * fs
    b, a = butter(3, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def calculate_bpm_from_signal(sig, fs, band=BREATHING_BAND):
    sig = sig - np.mean(sig)
    N = len(sig)
    yf = np.abs(fft(sig))
    freqs = fftfreq(N, 1/fs)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if np.any(mask):
        peak_idx = np.argmax(yf[mask])
        peak_freq = freqs[mask][peak_idx]
        return peak_freq * 60.0
    else:
        return np.nan

def get_bpm_by_peaks(sig, fs):
    sig_detrend = detrend(sig)
    min_dist = int(fs / 1.5)
    peaks, _ = find_peaks(sig_detrend, distance=min_dist, prominence=np.std(sig_detrend) * 0.3) 
    if len(peaks) > 1:
        intervals = np.diff(peaks) / fs
        median_interval = np.median(intervals)
        return 60.0 / median_interval
    else:
        return np.nan

def calculate_icc_31(est_list, gt_list):
    N = len(est_list)
    if N <= 1:
        return 0.0
    ratings_matrix = np.column_stack((est_list, gt_list))
    K = 2
    grand_mean = np.mean(ratings_matrix)
    subject_means = np.mean(ratings_matrix, axis=1)
    rater_means = np.mean(ratings_matrix, axis=0)
    ss_total = np.sum((ratings_matrix - grand_mean) ** 2)
    ss_between = K * np.sum((subject_means - grand_mean) ** 2)
    ss_within = ss_total - ss_between
    ss_raters = N * np.sum((rater_means - grand_mean) ** 2)
    ss_error = ss_within - ss_raters
    ms_between = ss_between / (N - 1)
    ms_within = ss_within / (N * (K - 1))
    ms_raters = ss_raters / (K - 1)
    ms_error = ss_error / ((N - 1) * (K - 1))
    icc31 = (ms_between - ms_error) / (ms_between + (K - 1) * ms_error + 1e-8)
    return icc31

def save_overlay_visualization(first_frame, top_cells, filtered_signals, respiration_resampled, final_score_map, grid_size, scale_factor, y_start, grid_w, grid_h, bbox, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cropped_frame = first_frame[y_start:, :].copy()
    
    if bbox:
        bx1, by1, bx2, by2 = bbox
        by1_shifted = int(max(0, by1 - y_start))
        by2_shifted = int(max(0, by2 - y_start))
        cv2.rectangle(cropped_frame, (int(bx1), by1_shifted), (int(bx2), by2_shifted), (0, 255, 0), 2)
        cv2.putText(cropped_frame, "Body BBox", (int(bx1) + 5, by1_shifted + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    
    fig = Figure(figsize=(16, 7))
    canvas = FigureCanvas(fig)
    axes = fig.subplots(1, 2)
    
    heatmap_resized = cv2.resize(final_score_map, (cropped_frame.shape[1], cropped_frame.shape[0]))
    heatmap_norm = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(np.uint8(heatmap_norm), cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(cropped_rgb, 0.6, heatmap_color_rgb, 0.4, 0)
    axes[0].imshow(overlay)
    
    colors = ['red', 'green', 'blue', 'orange', 'cyan', 'magenta', 'yellow', 'lime']
    
    for i, (gy, gx, score) in enumerate(top_cells):
        x_start_resized = gx * grid_w
        x_end_resized = (gx + 1) * grid_w
        y_start_resized = gy * grid_h
        y_end_resized = (gy + 1) * grid_h
        
        x1 = int(x_start_resized / scale_factor)
        x2 = int(x_end_resized / scale_factor)
        y1 = int(y_start_resized / scale_factor)
        y2 = int(y_end_resized / scale_factor)
        
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colors[i % len(colors)], linewidth=2)
        axes[0].add_patch(rect)
        axes[0].text(x1, y1 - 10, f"Rank {i+1}", color=colors[i % len(colors)], fontsize=10, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
    axes[0].set_title(f"Top Cells (K={len(top_cells)}) on Fused Heatmap")
    axes[0].axis('off')
    
    # Plot GT
    gt_norm = (respiration_resampled - np.mean(respiration_resampled)) / (np.std(respiration_resampled) + 1e-8)
    axes[1].plot(gt_norm, label="Ground Truth (Resampled)", color='black', linewidth=2.0, linestyle='--')
    
    for i, sig in enumerate(filtered_signals):
        sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
        axes[1].plot(sig_norm, label=f"Rank {i+1} (Grid {top_cells[i][0]},{top_cells[i][1]})", color=colors[i % len(colors)], linewidth=1.5)
        
    axes[1].set_title("Normalized Respiration Signals Comparison")
    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("Normalized Amplitude")
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    fig.suptitle(f"Subject Visual Analysis (PP + Optical Flow + Color) - {os.path.basename(save_path)}", fontsize=14, weight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    fig.clf()
    plt.close('all')

class DummyLandmark:
    def __init__(self, x, y, visibility):
        self.x = x
        self.y = y
        self.visibility = visibility

def map_landmarks_to_original_normalized(pose_landmarks, rotation, width, height):
    pts_mapped = {}
    if not pose_landmarks:
        return pts_mapped
        
    if rotation == 'CCW':
        rot_w = height
        rot_h = width
        for idx, lm in enumerate(pose_landmarks.landmark):
            x_rot = lm.x * rot_w
            y_rot = lm.y * rot_h
            x_orig = width - 1 - y_rot
            y_orig = x_rot
            pts_mapped[idx] = DummyLandmark(x_orig / width, y_orig / height, lm.visibility)
    elif rotation == 'CW':
        rot_w = height
        rot_h = width
        for idx, lm in enumerate(pose_landmarks.landmark):
            x_rot = lm.x * rot_w
            y_rot = lm.y * rot_h
            x_orig = y_rot
            y_orig = height - 1 - x_rot
            pts_mapped[idx] = DummyLandmark(x_orig / width, y_orig / height, lm.visibility)
    else:
        for idx, lm in enumerate(pose_landmarks.landmark):
            pts_mapped[idx] = DummyLandmark(lm.x, lm.y, lm.visibility)
            
    return pts_mapped

def map_grid_to_infant_region(cx, cy, bbox):
    x_min, y_min, x_max, y_max = bbox
    eps = 1e-6
    x_rel = np.clip((cx - x_min) / (x_max - x_min + eps), 0.0, 1.0)
    y_rel = np.clip((cy - y_min) / (y_max - y_min + eps), 0.0, 1.0)

    if y_rel < 0.20:
        vertical_region = "HEAD"
    elif y_rel < 0.50:
        vertical_region = "UPPER_TORSO"
    elif y_rel < 0.80:
        vertical_region = "LOWER_TORSO"
    else:
        vertical_region = "LOWER_BODY"

    if x_rel < 0.25:
        horizontal_region = "ARM"
    elif x_rel > 0.75:
        horizontal_region = "ARM"
    else:
        horizontal_region = "CENTRAL"

    final_region = vertical_region
    if horizontal_region == "ARM":
        final_region = horizontal_region

    region_map = {
        "HEAD": 1, "UPPER_TORSO": 2, "LOWER_TORSO": 3,
        "LOWER_BODY": 4, "ARM": 5
    }
    return region_map.get(final_region, 2)

def is_grayscale_like(frame):
    """
    Check whether the frame is IR/grayscale-like.
    """
    b, g, r = cv2.split(frame)
    diff_bg = np.mean(np.abs(b.astype(np.float32) - g.astype(np.float32)))
    diff_gr = np.mean(np.abs(g.astype(np.float32) - r.astype(np.float32)))
    diff_br = np.mean(np.abs(b.astype(np.float32) - r.astype(np.float32)))
    channel_diff = (diff_bg + diff_gr + diff_br) / 3.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])
    return saturation < 15 and channel_diff < 5

def process_video_to_signals(video_path, grid_size=GRID_SIZE, scale_factor=SCALE_FACTOR):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    y_start = int(height * 0.1)
    
    color_signals_list = []
    flow_signals_list = []
    frames_cache = []
    
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read first frame of video.")
        
    frames_cache.append(first_frame)
    
    # Distinguish IR (Grayscale-like) vs RGB
    is_ir = is_grayscale_like(first_frame)
    
    cropped_first = first_frame[y_start:, :]
    sh = int(cropped_first.shape[0] * scale_factor)
    sw = int(cropped_first.shape[1] * scale_factor)
    grid_w = sw // grid_size[0]
    grid_w = max(1, grid_w)
    grid_h = sh // grid_size[1]
    grid_h = max(1, grid_h)
    H_target = grid_size[1] * grid_h
    W_target = grid_size[0] * grid_w
    
    def extract_channel_signal(cropped_f):
        frame_resized = cv2.resize(cropped_f, (sw, sh))
        if is_ir:
            # IR Video: Temporal Brightness (Intensity) Difference
            frame_ch = frame_resized.mean(axis=2)
        else:
            # Ycbcr, ycbcr
            # RGB Video: Temporal Color Difference (Chrominance in YCrCb: Cr - Cb)
            # Cr (Red-chroma) - Cb (Blue-chroma) isolates pure chromatic changes
            ycrcb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YCrCb)
            cr = ycrcb[:, :, 1].astype(np.float32)
            cb = ycrcb[:, :, 2].astype(np.float32)
            frame_ch = cr - cb
            
        frame_cropped = frame_ch[:H_target, :W_target]
        reshaped = frame_cropped.reshape(grid_size[1], grid_h, grid_size[0], grid_w)
        return reshaped.mean(axis=(1, 3))
    
    color_signals_list.append(extract_channel_signal(cropped_first))
    
    prev_gray = cv2.cvtColor(cropped_first, cv2.COLOR_BGR2GRAY)
    prev_resized = cv2.resize(prev_gray, (sw, sh))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frames_cache.append(frame)
        cropped_frame = frame[y_start:, :]
        
        color_signals_list.append(extract_channel_signal(cropped_frame))
        
        curr_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        curr_resized = cv2.resize(curr_gray, (sw, sh))
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_resized, curr_resized, None, 
            pyr_scale=0.5, levels=2, winsize=13, 
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0
        )
        flow_v = flow[..., 1]
        
        flow_cropped = flow_v[:H_target, :W_target]
        reshaped_flow = flow_cropped.reshape(grid_size[1], grid_h, grid_size[0], grid_w)
        flow_signals_list.append(reshaped_flow.mean(axis=(1, 3)))
        
        prev_resized = curr_resized
        
    cap.release()
    
    color_signals = np.transpose(np.array(color_signals_list), (1, 2, 0))
    flow_signals = np.transpose(np.array(flow_signals_list), (1, 2, 0))
    
    return color_signals, flow_signals, fps, width, height, y_start, grid_w, grid_h, frames_cache, is_ir

def save_csv_robust(csv_path, headers, rows, summary_stats_dict=None):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    for attempt in range(5):
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
                if summary_stats_dict:
                    writer.writerow({})
                    writer.writerow({'sample_id': 'Summary Statistics'})
                    for k, v in summary_stats_dict.items():
                        writer.writerow({'sample_id': k, 'video': v})
            return True
        except PermissionError:
            print(f"[Warning] Permission denied for {csv_path}. Retrying in 3 seconds... ({attempt+1}/5)")
            time.sleep(3)
            
    fallback_path = csv_path.replace(".csv", "_backup.csv")
    try:
        with open(fallback_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            if summary_stats_dict:
                writer.writerow({})
                writer.writerow({'sample_id': 'Summary Statistics'})
                for k, v in summary_stats_dict.items():
                    writer.writerow({'sample_id': k, 'video': v})
        print(f"[Success] Saved to backup file (due to lock): {fallback_path}")
        return True
    except Exception as e:
        print(f"[Error] Failed to write to backup file {fallback_path}: {e}")
        return False

# ==========================================
# 3. Worker Function for All Thresholds Sweep (0.5, 0.6, 0.7, 0.8, 0.9)
# ==========================================
def process_single_video_worker(args):
    video_path, gt_path, idx, total_count, scale_factor = args
    subject_folder = os.path.basename(os.path.dirname(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    posture = "supine"
    for p in ["supine", "side", "prone"]:
        if p in video_path.lower():
            posture = p
            break
            
    visual_filename = f"{subject_folder}_{video_name}_visualization.png"
    visual_path = os.path.join(VISUAL_DIR, visual_filename)
    
    cache_filename = f"{subject_folder}_{video_name}_color_scale_{scale_factor}_sweep_stats_3x3.json"
    cache_path = os.path.join(CACHE_DIR, cache_filename)
    
    if os.path.exists(cache_path) and os.path.exists(visual_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as cf:
                cached_data = json.load(cf)
            if cached_data.get('status') == 'success' and 'threshold_results' in cached_data:
                th_res = cached_data['threshold_results']
                if all(str(t) in th_res and 'k_results' in th_res[str(t)] and all((str(k) in th_res[str(t)]['k_results'].get('w_0.9', {}) or k in th_res[str(t)]['k_results'].get('w_0.9', {})) for k in range(1, TOP_K_CELLS + 1)) for t in [0.5, 0.6, 0.7, 0.8, 0.9]):
                    print(f"[{idx+1}/{total_count}] Loaded from cache: {subject_folder}/{video_name}.mp4")
                    return cached_data
        except Exception as ce:
            print(f"[Warning] Failed to read sweep cache {cache_path}: {ce}")

    threshold_results = {}
    
    try:
        # 1. Extract color (or intensity for IR) and optical flow signals
        color_signals, flow_signals, fps, width, height, y_start, grid_w, grid_h, frames_cache, is_ir = process_video_to_signals(video_path, scale_factor=scale_factor)
        total_frames = color_signals.shape[2]
        
        # Color difference & z-score
        color_signals_diff = np.diff(color_signals, axis=2)
        mean_c = np.mean(color_signals_diff, axis=2, keepdims=True)
        std_c = np.std(color_signals_diff, axis=2, keepdims=True)
        color_norm = (color_signals_diff - mean_c) / (std_c + 1e-8)
        
        # Flow z-score
        mean_f = np.mean(flow_signals, axis=2, keepdims=True)
        std_f = np.std(flow_signals, axis=2, keepdims=True)
        flow_norm = (flow_signals - mean_f) / (std_f + 1e-8)
        
        # 2. Fused FFT Periodicity Map
        periodicity_map = np.zeros((GRID_SIZE[1], GRID_SIZE[0]))
        freqs = fftfreq(total_frames - 1, 1/fps)
        mask = (freqs >= BREATHING_BAND[0]) & (freqs <= BREATHING_BAND[1])
        
        for gy in range(GRID_SIZE[1]):
            for gx in range(GRID_SIZE[0]):
                sig_c = color_norm[gy, gx]
                sig_c = sig_c - np.mean(sig_c)
                yf_c = np.abs(fft(sig_c))

                raw_diff_c = np.diff(color_signals[gy, gx])
                max_var_c = np.max(np.abs(raw_diff_c))
                penalty_c = np.exp(-0.2 * max(0.0, max_var_c - MAX_ALLOWED_VARIATION))
                
                sig_f = flow_norm[gy, gx]
                sig_f = sig_f - np.mean(sig_f)
                yf_f = np.abs(fft(sig_f))

                raw_diff_f = np.diff(flow_signals[gy, gx])
                max_var_f = np.max(np.abs(raw_diff_f))
                penalty_f = np.exp(-0.04 * max(0.0, max_var_f - MAX_ALLOWED_VARIATION * 5.0))
                
                yf_fused = yf_c * yf_f
                
                if np.any(mask):
                    dominant = np.max(yf_fused[mask])
                    total = np.sum(yf_fused)
                    periodicity_map[gy, gx] = dominant / (total + 1e-8)
        
        def get_correlation(sig1, sig2):
            s1, s2 = sig1 - np.mean(sig1), sig2 - np.mean(sig2)
            denom = np.sqrt(np.sum(s1**2) * np.sum(s2**2))
            return np.sum(s1 * s2) / denom if denom > 1e-8 else 0.0
            
        flat_idx = np.argsort(periodicity_map.flatten())[::-1]
        top_cells_first = None
        for i_idx in flat_idx:
            gy, gx = np.unravel_index(i_idx, periodicity_map.shape)
            score = periodicity_map[gy, gx]
            if score > 0:
                top_cells_first = (gy, gx, score)
                break
                
        # 3. Reference Bounding Box
        ref_bbox = (0, y_start, width, height)
        sec_bboxes = {sec: ref_bbox for sec in [0, 10, 20, 30, 40, 50, 60]}

        # 4. Load GT labels & compute gt_bpm
        with h5py.File(gt_path, 'r') as f:
            respiration = f['respiration'][:]
        respiration_resampled = resample(respiration, flow_norm.shape[2])
        
        N_gt = len(respiration)
        duration = total_frames / fps
        fs_gt = N_gt / duration
        gt_bpm = get_bpm_by_peaks(respiration, fs_gt)
        
        if np.isnan(gt_bpm):
            print(f"[{idx+1}/{total_count}] Failed: Invalid GT respiration signal (no peaks) for {subject_folder}/{video_name}.mp4")
            for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
                threshold_results[str(th)] = {
                    'status': 'false',
                    'top_cells': [],
                    'cell_regions_by_sec': {i: ['N/A'] * 7 for i in range(TOP_K_CELLS)},
                    'pairwise_corrs': {f"corr_{i+1}_{j+1}": np.nan for i in range(TOP_K_CELLS) for j in range(i + 1, TOP_K_CELLS)},
                    'k_results': {k: {'est_bpm': 0.0, 'accuracy': 0.0, 'snr': np.nan} for k in [1, 2, 3, 4, 5]},
                    'top_cells_corr': [0.0] * 5
                }
            return {
                'status': 'false',
                'sample_id': idx + 1,
                'video': video_name + '.mp4',
                'subject': subject_folder,
                'video_type': 'IR' if is_ir else 'RGB',
                'posture': posture,
                'gt_bpm': 0.0,
                'threshold_results': threshold_results
            }

        # 5. Run sweep for each threshold
        for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
            top_cells = []
            if top_cells_first is not None:
                top_cells.append(top_cells_first)
                curr_gy, curr_gx, _ = top_cells_first
                visited = {(curr_gy, curr_gx)}
                
                while len(top_cells) < TOP_K_CELLS:
                    ref_sig = flow_norm[curr_gy, curr_gx]
                    best_candidate = None
                    best_candidate_corr = -1.0
                    
                    # Grid expansion search window
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = curr_gy + dy, curr_gx + dx
                            if 0 <= ny < GRID_SIZE[1] and 0 <= nx < GRID_SIZE[0]:
                                if (ny, nx) not in visited:
                                    tgt_sig = flow_norm[ny, nx]
                                    corr = get_correlation(ref_sig, tgt_sig)
                                    if corr >= th and corr > best_candidate_corr:
                                        best_candidate_corr = corr
                                        best_candidate = (ny, nx, periodicity_map[ny, nx])
                                        
                    if best_candidate is not None:
                        top_cells.append(best_candidate)
                        visited.add((best_candidate[0], best_candidate[1]))
                        curr_gy, curr_gx = best_candidate[0], best_candidate[1]
                    else:
                        break
            
            # 5.1 Region Mapping for this top_cells
            cell_w_orig = width / float(GRID_SIZE[0])
            cell_h_orig = (height - y_start) / float(GRID_SIZE[1])
            region_names = {1: "HEAD", 2: "UPPER_TORSO", 3: "LOWER_TORSO", 4: "LOWER_BODY", 5: "ARM"}
            
            cell_regions_by_sec = {i: [] for i in range(TOP_K_CELLS)}
            for sec in [0, 10, 20, 30, 40, 50, 60]:
                bbox_t = sec_bboxes[sec]
                for i, (gy, gx, _) in enumerate(top_cells):
                    cx = (gx + 0.5) * cell_w_orig
                    cy = (gy + 0.5) * cell_h_orig + y_start
                    region_val = map_grid_to_infant_region(cx, cy, bbox_t)
                    region_name = region_names.get(region_val, "UPPER_TORSO")
                    cell_regions_by_sec[i].append(region_name)
                for i in range(len(top_cells), TOP_K_CELLS):
                    cell_regions_by_sec[i].append("N/A")
                    
            # 5.2 Pairwise Correlations
            pairwise_corrs = {}
            filtered_sigs_th = [bandpass(flow_norm[gy, gx], fps, BREATHING_BAND[0], BREATHING_BAND[1]) for gy, gx, _ in top_cells]
            for i in range(TOP_K_CELLS):
                for j in range(i + 1, TOP_K_CELLS):
                    key = f"corr_{i+1}_{j+1}"
                    if i < len(filtered_sigs_th) and j < len(filtered_sigs_th):
                        sig1 = filtered_sigs_th[i]
                        sig2 = filtered_sigs_th[j]
                        r_val = get_correlation(sig1, sig2)
                        pairwise_corrs[key] = round(abs(r_val), 4) if not np.isnan(r_val) else 0.0
                    else:
                        pairwise_corrs[key] = np.nan

            # 5.3 Calculate correlation for each top cell (Pearson abs_r against GT)
            top_cells_corr = []
            for gy, gx, _ in top_cells:
                cell_sig = flow_norm[gy, gx]
                cell_filt = bandpass(cell_sig, fps, BREATHING_BAND[0], BREATHING_BAND[1])
                r_val = get_correlation(cell_filt, respiration_resampled)
                top_cells_corr.append(abs(r_val) if not np.isnan(r_val) else 0.0)
                        
            # 5.4 Calculate est_bpm & accuracy for each K across different weights
            k_results = {}
            for w_idx in range(11):
                w_flow = w_idx * 0.1
                w_color = 1.0 - w_flow
                w_key = f"w_{w_flow:.1f}"
                k_results[w_key] = {}
                
                for k in range(1, TOP_K_CELLS + 1):
                    top_cells_k = top_cells[:k]
                    # Weighted Dual-channel Fused Ensemble Signal
                    ensemble_signals = [w_flow * flow_norm[gy, gx] + w_color * color_norm[gy, gx] for gy, gx, _ in top_cells_k] 
                    best_flow_sig = np.mean(ensemble_signals, axis=0)
                    # Bandpass filtering & linear detrending
                    best_filtered_sig = bandpass(best_flow_sig, fps, BREATHING_BAND[0], BREATHING_BAND[1])
                    best_filtered_sig = detrend(best_filtered_sig)
                    
                    est_bpm = calculate_bpm_from_signal(best_filtered_sig, fps)
                    
                    if gt_bpm > 0:
                        accuracy = (1.0 - abs(est_bpm - gt_bpm) / gt_bpm) * 100.0
                    else:
                        accuracy = 0.0
                        
                    # SNR
                    yf = np.abs(fft(best_filtered_sig - np.mean(best_filtered_sig)))
                    freqs_fft = fftfreq(len(best_filtered_sig), 1/fps)
                    mask_fft = (freqs_fft >= BREATHING_BAND[0]) & (freqs_fft <= BREATHING_BAND[1])
                    if np.any(mask_fft) and not np.isnan(est_bpm):
                        ref_hz = gt_bpm / 60.0
                        freqs_mask = freqs_fft[mask_fft]
                        yf_mask = yf[mask_fft]
                        signal_range = ((freqs_mask >= ref_hz - 0.05) & (freqs_mask <= ref_hz + 0.05)) | \
                                       ((freqs_mask >= 2 * ref_hz - 0.05) & (freqs_mask <= 2 * ref_hz + 0.05))
                        signal_energy = np.sum(yf_mask[signal_range] ** 2)
                        noise_energy = np.sum(yf_mask[~signal_range] ** 2)
                        if noise_energy > 1e-8 and signal_energy > 1e-8:
                            snr = 10 * np.log10(signal_energy / noise_energy)
                        else:
                            snr = np.nan
                    else:
                        snr = np.nan
                        
                    k_results[w_key][k] = {
                        'est_bpm': round(est_bpm, 2),
                        'accuracy': round(accuracy, 2),
                        'snr': round(snr, 2) if not np.isnan(snr) else np.nan
                    }
                    
            threshold_results[str(th)] = {
                'status': 'success',
                'top_cells': [(int(gy), int(gx)) for gy, gx, _ in top_cells],
                'cell_regions_by_sec': cell_regions_by_sec,
                'pairwise_corrs': pairwise_corrs,
                'k_results': k_results,
                'top_cells_corr': top_cells_corr
            }
            
        # Save visualization for optimal configuration (th=0.9, w_flow=0.9, w_color=0.1)
        try:
            th_09_cells = threshold_results.get('0.9', {}).get('top_cells', [])
            viz_cells = [(gy, gx, periodicity_map[gy, gx]) for gy, gx in th_09_cells[:4]]
            viz_sigs = []
            for gy, gx, _ in viz_cells:
                sig_fused = 0.9 * flow_norm[gy, gx] + 0.1 * color_norm[gy, gx]
                sig_filt = bandpass(sig_fused, fps, BREATHING_BAND[0], BREATHING_BAND[1])
                sig_filt = detrend(sig_filt)
                viz_sigs.append(sig_filt)
                
            save_overlay_visualization(
                first_frame=frames_cache[0],
                top_cells=viz_cells,
                filtered_signals=viz_sigs,
                respiration_resampled=respiration_resampled,
                final_score_map=periodicity_map,
                grid_size=GRID_SIZE,
                scale_factor=scale_factor,
                y_start=y_start,
                grid_w=grid_w,
                grid_h=grid_h,
                bbox=ref_bbox,
                save_path=visual_path
            )
            
            # Also save in posture subdirectory: e.g. D:\neonate\Test8_final_media_ref_pp_color\{posture}\visualizations\
            posture_visual_dir = os.path.join(SAVE_DIR, posture, "visualizations")
            os.makedirs(posture_visual_dir, exist_ok=True)
            import shutil
            shutil.copy2(visual_path, os.path.join(posture_visual_dir, visual_filename))
        except Exception as viz_e:
            print(f"[Warning] Failed to generate visualization for {subject_folder}/{video_name}: {viz_e}")

        print(f"[{idx+1}/{total_count}] Success ({'IR' if is_ir else 'RGB'}): {subject_folder}/{video_name}.mp4")
        
        result_dict = {
            'status': 'success',
            'sample_id': idx + 1,
            'video': video_name + '.mp4',
            'subject': subject_folder,
            'video_type': 'IR' if is_ir else 'RGB',
            'posture': posture,
            'gt_bpm': round(gt_bpm, 2),
            'threshold_results': threshold_results
        }
        
        # Save output dictionary to cache
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as cf:
                json.dump(result_dict, cf, ensure_ascii=False, indent=2)
        except Exception as ce:
            print(f"[Warning] Failed to write sweep cache {cache_path}: {ce}")
            
        return result_dict
        
    except Exception as e:
        print(f"[{idx+1}/{total_count}] Error processing {video_path}: {e}")
        import traceback
        traceback.print_exc()
        for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
            threshold_results[str(th)] = {
                'status': 'false',
                'top_cells': [],
                'cell_regions_by_sec': {i: ['N/A'] * 7 for i in range(TOP_K_CELLS)},
                'pairwise_corrs': {f"corr_{i+1}_{j+1}": np.nan for i in range(TOP_K_CELLS) for j in range(i + 1, TOP_K_CELLS)},
                'k_results': {k: {'est_bpm': 0.0, 'accuracy': 0.0, 'snr': np.nan} for k in [1, 2, 3, 4, 5]},
                'top_cells_corr': [0.0] * 5
            }
        return {
            'status': 'false',
            'sample_id': idx + 1,
            'video': video_name + '.mp4',
            'subject': subject_folder,
            'video_type': 'UNKNOWN',
            'posture': posture,
            'gt_bpm': 0.0,
            'threshold_results': threshold_results
        }

# ==========================================
# 4. Main Execution Pipeline
# ==========================================
def main():
    print("===== Scanning Dataset for Video-GT Pairs =====")
    pairs = get_all_video_gt_pairs(DATASET_DIR)
    total_count = len(pairs)
    print(f"Found {total_count} valid video-GT pairs in {DATASET_DIR}\n")
    
    if total_count == 0:
        print("[Error] No valid pairs found.")
        return
        
    worker_args = [(pairs[i][0], pairs[i][1], i, total_count, SCALE_FACTOR) for i in range(total_count)]
    
    results = []
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(process_single_video_worker, worker_args)
        
    results = sorted(results, key=lambda x: x['sample_id'])
    
    # We will accumulate sweep statistics for the summary table
    sweep_summary_rows = []
    
    # 4.1 Global Weight/Threshold/K Sweep in memory
    print("\n===== Running Global Grid Sweep over Weights, Thresholds, and K-Ensembles =====")
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for w_flow in weights:
        w_key = f"w_{w_flow:.1f}"
        for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for k in range(1, TOP_K_CELLS + 1):
                success_rows = []
                actual_cell_counts = []
                for r in results:
                    th_res = r['threshold_results'][str(th)]
                    if th_res['status'] == 'success':
                        k_res_dict = th_res.get('k_results', {}).get(w_key, {})
                        k_res = k_res_dict.get(str(k)) or k_res_dict.get(k)
                        if k_res is not None:
                            actual_k = min(k, len(th_res['top_cells']))
                            actual_cell_counts.append(actual_k)
                            success_rows.append({
                                'est_bpm': k_res['est_bpm'],
                                'gt_bpm': r['gt_bpm'],
                                'accuracy': k_res['accuracy'],
                                'snr': k_res['snr']
                            })
                
                success_count = len(success_rows)
                if success_count > 0:
                    est_bpms = np.array([row['est_bpm'] for row in success_rows])
                    gt_bpms = np.array([row['gt_bpm'] for row in success_rows])
                    errors = np.abs(est_bpms - gt_bpms)
                    mae = np.mean(errors)
                    rmse = np.sqrt(np.mean(errors**2))
                    pcc = np.corrcoef(est_bpms, gt_bpms)[0, 1] if len(success_rows) > 1 and np.std(est_bpms) > 0 and np.std(gt_bpms) > 0 else 0.0
                    icc_val = calculate_icc_31(est_bpms, gt_bpms)
                    success_3bpm = np.mean(errors <= 3.0) * 100.0
                    success_5bpm = np.mean(errors <= 5.0) * 100.0
                    valid_snrs = [row['snr'] for row in success_rows if not np.isnan(row['snr'])]
                    avg_snr = np.mean(valid_snrs) if len(valid_snrs) > 0 else 0.0
                    avg_selected_cells = np.mean(actual_cell_counts)
                else:
                    mae = rmse = pcc = icc_val = success_3bpm = success_5bpm = avg_snr = avg_selected_cells = 0.0
                    
                sweep_summary_rows.append({
                    'weight_flow': round(w_flow, 1),
                    'weight_color': round(1.0 - w_flow, 1),
                    'threshold': th,
                    'K': k,
                    'Avg_Selected_Cells': round(avg_selected_cells, 2),
                    'MAE': round(mae, 4),
                    'RMSE': round(rmse, 4),
                    'PCC': round(pcc, 4),
                    'ICC': round(icc_val, 4),
                    'Avg_SNR': round(avg_snr, 2),
                    'SR_3BPM': f"{round(success_3bpm, 2)}%",
                    'SR_5BPM': f"{round(success_5bpm, 2)}%"
                })
                
    # Identify the optimal combination (based on MAE)
    optimal_config = min(sweep_summary_rows, key=lambda x: x['MAE'])
    best_w_flow = optimal_config['weight_flow']
    best_w_key = f"w_{best_w_flow:.1f}"
    
    print(f"\n -> Found optimal configuration: w_flow={best_w_flow:.1f} (color={1.0-best_w_flow:.1f}) | th={optimal_config['threshold']} | K={optimal_config['K']} | MAE={optimal_config['MAE']:.4f}\n")

    # 4.2 Write K-results CSVs using the optimal weight ratio
    print(f"===== Saving Detailed CSVs to Disk using Optimal Weight {best_w_key} =====")
    for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
        th_dir = os.path.join(SAVE_DIR, f"threshold_{th}")
        os.makedirs(th_dir, exist_ok=True)
        
        for k in range(1, TOP_K_CELLS + 1):
            save_targets = [
                (th_dir, results, f"Global K = {k} (th={th})")
            ]
            for posture in ['prone', 'side', 'supine']:
                posture_results = [r for r in results if r['posture'] == posture]
                if len(posture_results) > 0:
                    save_targets.append((os.path.join(th_dir, posture), posture_results, f"{posture.upper()} K = {k} (th={th})"))
                    
            for target_dir, target_results, target_msg in save_targets:
                compiled_rows = []
                for r in target_results:
                    th_res = r['threshold_results'][str(th)]
                    if th_res['status'] == 'success':
                        k_res_dict = th_res.get('k_results', {}).get(best_w_key, {})
                        k_res = k_res_dict.get(str(k)) or k_res_dict.get(k)
                        est_b = k_res['est_bpm'] if k_res else 0.0
                        acc_b = k_res['accuracy'] if k_res else 0.0
                        snr_b = k_res['snr'] if k_res else np.nan
                        
                        row = {
                            'sample_id': r['sample_id'],
                            'video': r['video'],
                            'subject': r['subject'],
                            'video_type': r.get('video_type', 'UNKNOWN'),
                            'roi': f"({th_res['top_cells'][0][0]},{th_res['top_cells'][0][1]})" if len(th_res['top_cells']) > 0 else 'N/A',
                            'est_bpm': est_b,
                            'gt_bpm': r['gt_bpm'],
                            'accuracy': acc_b,
                            'snr': snr_b,
                            'status': 'success' if k_res else 'false'
                        }
                        for i in range(k):
                            if i < len(th_res['top_cells']):
                                row[f'top{i+1}_coord'] = f"({th_res['top_cells'][i][0]},{th_res['top_cells'][i][1]})"
                                for sec_idx, sec in enumerate([0, 10, 20, 30, 40, 50, 60]):
                                    if isinstance(th_res['cell_regions_by_sec'], dict):
                                        cell_reg_list = th_res['cell_regions_by_sec'].get(str(i)) or th_res['cell_regions_by_sec'].get(i)
                                    else:
                                        cell_reg_list = th_res['cell_regions_by_sec'][i]
                                    row[f'top{i+1}_region_{sec}s'] = cell_reg_list[sec_idx]
                            else:
                                row[f'top{i+1}_coord'] = 'N/A'
                                for sec in [0, 10, 20, 30, 40, 50, 60]:
                                    row[f'top{i+1}_region_{sec}s'] = 'N/A'
                        compiled_rows.append(row)
                    else:
                        row = {
                            'sample_id': r['sample_id'],
                            'video': r['video'],
                            'subject': r['subject'],
                            'video_type': r.get('video_type', 'UNKNOWN'),
                            'roi': 'N/A',
                            'est_bpm': 0.0,
                            'gt_bpm': 0.0,
                            'accuracy': 0.0,
                            'snr': np.nan,
                            'status': 'false'
                        }
                        for i in range(k):
                            row[f'top{i+1}_coord'] = 'N/A'
                            for sec in [0, 10, 20, 30, 40, 50, 60]:
                                row[f'top{i+1}_region_{sec}s'] = 'N/A'
                        compiled_rows.append(row)
                        
                success_rows = [row for row in compiled_rows if row['status'] == 'success']
                success_count = len(success_rows)
                if success_count > 0:
                    est_bpms = np.array([row['est_bpm'] for row in success_rows])
                    gt_bpms = np.array([row['gt_bpm'] for row in success_rows])
                    errors = np.abs(est_bpms - gt_bpms)
                    mae = np.mean(errors)
                    rmse = np.sqrt(np.mean(errors**2))
                    avg_est_bpm = np.mean(est_bpms)
                    avg_gt_bpm = np.mean(gt_bpms)
                    avg_accuracy = np.mean([row['accuracy'] for row in success_rows])
                    median_accuracy = np.median([row['accuracy'] for row in success_rows])
                    pcc = np.corrcoef(est_bpms, gt_bpms)[0, 1] if len(success_rows) > 1 and np.std(est_bpms) > 0 and np.std(gt_bpms) > 0 else 0.0
                    success_3bpm = np.mean(errors <= 3.0) * 100.0
                    success_5bpm = np.mean(errors <= 5.0) * 100.0
                    valid_snrs = [row['snr'] for row in success_rows if not np.isnan(row['snr'])]
                    avg_snr = np.mean(valid_snrs) if len(valid_snrs) > 0 else 0.0
                    icc_val = calculate_icc_31(est_bpms, gt_bpms)
                else:
                    mae = rmse = avg_est_bpm = avg_gt_bpm = avg_accuracy = median_accuracy = pcc = success_3bpm = success_5bpm = avg_snr = 0.0
                    icc_val = 0.0
                    
                csv_path = os.path.join(target_dir, f"respiration_results_top_{k}_3x3.csv")
                headers = ['sample_id', 'video', 'subject', 'video_type', 'roi', 'est_bpm', 'gt_bpm', 'accuracy', 'snr', 'status']
                for i in range(k):
                    headers.append(f'top{i+1}_coord')
                    for sec in [0, 10, 20, 30, 40, 50, 60]:
                        headers.append(f'top{i+1}_region_{sec}s')
                        
                stats_dict = {
                    'MAE': round(mae, 4),
                    'RMSE': round(rmse, 4),
                    'Average Est BPM': round(avg_est_bpm, 2),
                    'Average GT BPM': round(avg_gt_bpm, 2),
                    'Average Accuracy': f"{round(avg_accuracy, 2)}%",
                    'Median Accuracy': f"{round(median_accuracy, 2)}%",
                    'PCC': round(pcc, 4),
                    'ICC': round(icc_val, 4),
                    'Success Rate (<3.0 BPM)': f"{round(success_3bpm, 2)}%",
                    'Success Rate (<5.0 BPM)': f"{round(success_5bpm, 2)}%",
                    'Average SNR': f"{round(avg_snr, 2)} dB"
                }
                save_csv_robust(csv_path, headers, compiled_rows, stats_dict)

    # 4.3 Save Grand Summary CSV file of the sweep
    try:
        summary_csv_path = os.path.join(SAVE_DIR, "threshold_sweep_summary_3x3.csv")
        summary_headers = ['weight_flow', 'weight_color', 'threshold', 'K', 'Avg_Selected_Cells', 'MAE', 'RMSE', 'PCC', 'ICC', 'Avg_SNR', 'SR_3BPM', 'SR_5BPM']
        save_csv_robust(summary_csv_path, summary_headers, sweep_summary_rows)
        print(f"\n -> Saved grand threshold sweep summary to: {summary_csv_path}")
    except Exception as sum_err:
        print(f"[Warning] Failed to generate grand sweep summary CSV: {sum_err}")

    # Print the grand summary table to console (Top 15 configs by MAE)
    print("\n" + "="*145)
    print("TOP 15 GRID SWEEP CONFIGURATIONS (w_flow, w_color, threshold, K) SORTED BY MAE")
    print("="*145)
    print(f"{'Rank':<6} | {'W_Flow':<8} | {'W_Color':<8} | {'Threshold':<10} | {'K (Cells)':<10} | {'MAE (BPM)':<10} | {'RMSE':<10} | {'PCC':<10} | {'ICC':<10} | {'Avg SNR':<12} | {'SR (<5)':<10}")
    print("-"*145)
    sorted_sweep = sorted(sweep_summary_rows, key=lambda x: x['MAE'])
    for rank, row in enumerate(sorted_sweep[:15]):
        print(f"{rank+1:<6} | {row['weight_flow']:<8} | {row['weight_color']:<8} | {row['threshold']:<10} | {row['K']:<10} | {row['MAE']:<10.4f} | {row['RMSE']:<10.4f} | {row['PCC']:<10.4f} | {row['ICC']:<10.4f} | {row['Avg_SNR']:<8.2f} dB  | {row['SR_5BPM']:<10}")
    print("="*145)

    # 🌟 Generate posture_region_frequency.png, posture_region_frequency.csv, and body_table.png for threshold = 0.9
    try:
        print("\n===== Generating Posture-Region Frequency & Correlation Table Plots (th=0.9) =====")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        
        region_map_inv = {"HEAD": 1, "UPPER_TORSO": 2, "LOWER_TORSO": 3, "LOWER_BODY": 4, "ARM": 5}
        postures = ['prone', 'side', 'supine']
        
        # Aggregate Frequency
        freq_counts = {p: {r: 0 for r in range(1, 6)} for p in postures}
        corr_lists = {p: {r: [] for r in range(1, 6)} for p in postures}
        
        for r in results:
            th_res = r['threshold_results']['0.9']
            if th_res['status'] == 'success':
                posture = r['posture']
                if posture not in postures:
                    continue
                if len(th_res['top_cells']) > 0:
                    i = 0
                    if isinstance(th_res['cell_regions_by_sec'], dict):
                        cell_reg_list = th_res['cell_regions_by_sec'].get(str(i)) or th_res['cell_regions_by_sec'].get(i)
                    else:
                        cell_reg_list = th_res['cell_regions_by_sec'][i]
                    region_name = cell_reg_list[0]
                    region_id = region_map_inv.get(region_name, 2)
                    
                    freq_counts[posture][region_id] += 1
                    
                    corr_val = th_res.get('top_cells_corr', [0.0] * len(th_res['top_cells']))[i]
                    corr_lists[posture][region_id].append(corr_val)
                    
        # 1. Plot posture_region_frequency.png (Horizontal Stacked Bar Chart)
        fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=300)
        colors = {1: '#FF6B6B', 2: '#F4A261', 3: '#2A9D8F', 4: '#457B9D', 5: '#9B5DE5'}
        region_labels = {1: 'HEAD (1)', 2: 'UPPER_TORSO (2)', 3: 'LOWER_TORSO (3)', 4: 'LOWER_BODY (4)', 5: 'ARM (5)'}
        
        posture_order = postures[::-1] # ['supine', 'side', 'prone']
        lefts = np.zeros(len(postures))
        legend_handles = {}
        
        for r_id in range(1, 6):
            vals = []
            for p in posture_order:
                vals.append(freq_counts[p][r_id])
            vals = np.array(vals)
            bar = ax.barh([p.upper() for p in posture_order], vals, left=lefts, label=region_labels[r_id], color=colors[r_id], height=0.4, alpha=0.9, edgecolor='none', zorder=3)
            lefts += vals
            if r_id not in legend_handles:
                legend_handles[r_id] = bar[0]
                
        for idx, total_val in enumerate(lefts):
            ax.text(total_val + 1.5, idx, f"{int(total_val)}", ha='left', va='center', fontsize=8, color='#111111', weight='bold')
            
        ax.set_title("Distribution of Selected Grid Cell Position by Infant Sleeping Posture", fontsize=11, pad=12, weight='bold', color='#222222')
        ax.set_xlabel("Proportional of Selected Respiratory Position", fontsize=9, weight='bold', color='#222222')
        ax.set_ylabel("Posture", fontsize=9, weight='bold', color='#222222')
        
        handles = [legend_handles[r] for r in range(1, 6)]
        labels = [region_labels[r] for r in range(1, 6)]
        ax.legend(handles, labels, title="Respiratory Position", title_fontsize=9, fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=True, edgecolor='#e0e0e0')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.5, color='#cccccc')
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        freq_path = os.path.join(SAVE_DIR, "posture_region_frequency.png")
        horizontal_path = os.path.join(SAVE_DIR, "posture_region_stacked_horizontal.png")
        plt.savefig(freq_path, dpi=300, bbox_inches='tight')
        plt.savefig(horizontal_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" -> Saved posture_region_frequency.png & posture_region_stacked_horizontal.png to: {SAVE_DIR}")
        
        # 2. Save posture_region_frequency.csv (for Rank 1 cell / top_1.csv)
        try:
            freq_rank1 = {p: {r_id: 0 for r_id in range(1, 6)} for p in postures}
            for r in results:
                th_res = r['threshold_results']['0.9']
                if th_res['status'] == 'success':
                    posture = r['posture']
                    if posture not in postures:
                        continue
                    if len(th_res['top_cells']) > 0:
                        if isinstance(th_res['cell_regions_by_sec'], dict):
                            cell_reg_list = th_res['cell_regions_by_sec'].get('0') or th_res['cell_regions_by_sec'].get(0)
                        else:
                            cell_reg_list = th_res['cell_regions_by_sec'][0]
                        region_name = cell_reg_list[0]
                        region_id = region_map_inv.get(region_name, 2)
                        freq_rank1[posture][region_id] += 1
                        
            csv_freq_path = os.path.join(SAVE_DIR, "posture_region_frequency.csv")
            freq_headers = ['posture', 'HEAD', 'UPPER_TORSO', 'LOWER_TORSO', 'LOWER_BODY', 'ARM', 'TOTAL']
            freq_rows = []
            region_keys = {1: 'HEAD', 2: 'UPPER_TORSO', 3: 'LOWER_TORSO', 4: 'LOWER_BODY', 5: 'ARM'}
            for p in postures:
                row_data = {'posture': p}
                total = 0
                for r_id in range(1, 6):
                    cnt = freq_rank1[p][r_id]
                    row_data[region_keys[r_id]] = cnt
                    total += cnt
                row_data['TOTAL'] = total
                freq_rows.append(row_data)
                
            totals_row = {'posture': 'TOTAL'}
            grand_total = 0
            for r_id in range(1, 6):
                total_r = sum(freq_rank1[p][r_id] for p in postures)
                totals_row[region_keys[r_id]] = total_r
                grand_total += total_r
            totals_row['TOTAL'] = grand_total
            freq_rows.append(totals_row)
            
            save_csv_robust(csv_freq_path, freq_headers, freq_rows)
            print(f" -> Saved posture_region_frequency.csv to: {csv_freq_path}")
        except Exception as csv_err:
            print(f"[Warning] Failed to generate posture_region_frequency.csv: {csv_err}")
            
        # 3. Plot body_table.png (Absolute correlation table plot)
        try:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            ax.axis('off')
            
            table_data = []
            headers_table = ['Posture', 'HEAD', 'UPPER_TORSO', 'LOWER_TORSO', 'LOWER_BODY', 'ARM']
            
            for p in postures:
                row_data = [p.upper()]
                for r_id in range(1, 6):
                    corrs = corr_lists[p][r_id]
                    if len(corrs) > 0:
                        row_data.append(f"{np.mean(corrs):.4f}")
                    else:
                        row_data.append("N/A")
                table_data.append(row_data)
                
            # Add TOTAL row
            totals_row_data = ['ALL POSTURES']
            for r_id in range(1, 6):
                all_corrs = []
                for p in postures:
                    all_corrs.extend(corr_lists[p][r_id])
                if len(all_corrs) > 0:
                    totals_row_data.append(f"{np.mean(all_corrs):.4f}")
                else:
                    totals_row_data.append("N/A")
            table_data.append(totals_row_data)
            
            table = ax.table(cellText=table_data, colLabels=headers_table, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 2.0)
            
            # Styling table cells
            for (row_idx, col_idx), cell in table.get_celld().items():
                if row_idx == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#457B9D')
                elif col_idx == 0:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#F1FAEE')
                if row_idx == len(table_data):
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#E63946')
                    cell.set_text_props(color='white' if row_idx == len(table_data) else 'black')
                    
            ax.set_title("Mean Pearson Correlation Coefficient (abs_r) against GT", fontsize=14, pad=20, weight='bold')
            table_path = os.path.join(SAVE_DIR, "body_table.png")
            plt.savefig(table_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f" -> Saved body_table.png to: {table_path}")
        except Exception as tbl_err:
            print(f"[Warning] Failed to generate body_table.png: {tbl_err}")
            
    except Exception as plt_err:
         print(f"[Warning] Failed to generate frequency/correlation plots: {plt_err}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
