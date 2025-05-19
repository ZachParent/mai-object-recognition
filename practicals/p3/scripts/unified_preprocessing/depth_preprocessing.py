import os
import sys
import numpy as np
import cv2 # For saving images and image manipulation
from PIL import Image # For merging RGB and Alpha frames

# Ensure DataReader and its dependencies are in the Python path
sys.path.append('practicals/p3/scripts/unified_preprocessing')
sys.path.append('practicals/p3/scripts/unified_preprocessing/starter-kit')
sys.path.append('practicals/p3/scripts/unified_preprocessing/depth_reader')

from depth_reader.read import DataReader
from depth_render import Render
from util import intrinsic, extrinsic

# --- Configuration ---
OUTPUT_DIR = 'practicals/p3/scripts/unified_preprocessing/dataset_unified_test'

ORIG_IMG_WIDTH = 640
ORIG_IMG_HEIGHT = 480
TARGET_IMG_SIZE = 256
MARGIN = 10
CONTENT_SIZE = TARGET_IMG_SIZE - 2 * MARGIN # This is 236, so content area is 236x236

MAX_DEPTH_RENDER = 10.0 # Used for rendering and identifying background in depth map

# --- Helper Functions ---

# Removed generate_smpl_pose_visualization function

def quads2tris(F_quads):
    out_tris = []
    if F_quads is None or len(F_quads) == 0:
        return np.array([], dtype=np.int32).reshape(0,3)
    for f in F_quads:
        if f is None: continue # Should not happen with valid OBJ
        if len(f) == 3: out_tris.append(f)
        elif len(f) == 4:
            out_tris.append([f[0], f[1], f[2]]); out_tris.append([f[0], f[2], f[3]])
    if not out_tris: return np.array([], dtype=np.int32).reshape(0,3)
    return np.array(out_tris, np.int32)

def ensure_dir(directory):
    if not os.path.exists(directory): os.makedirs(directory)

def get_bbox_from_mask(mask):
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols): return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return cmin, rmin, cmax, rmax

def calculate_crop_and_scale_params_from_mask_bbox(
    mask_bbox_orig, orig_img_w, orig_img_h, target_content_w, target_content_h):
    if mask_bbox_orig is None: return None
    bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = mask_bbox_orig
    bbox_w_orig, bbox_h_orig = bbox_x_max - bbox_x_min + 1, bbox_y_max - bbox_y_min + 1
    if bbox_w_orig <= 0 or bbox_h_orig <= 0: return None
    scale_factor_w = target_content_w / bbox_w_orig if bbox_w_orig > 0 else float('inf')
    scale_factor_h = target_content_h / bbox_h_orig if bbox_h_orig > 0 else float('inf')
    scale_factor = min(scale_factor_w, scale_factor_h)
    if scale_factor <= 1e-6 or scale_factor == float('inf'): return None # Avoid division by zero or extremely small scale
    crop_window_w_orig, crop_window_h_orig = target_content_w / scale_factor, target_content_h / scale_factor
    center_x, center_y = bbox_x_min + bbox_w_orig / 2.0, bbox_y_min + bbox_h_orig / 2.0
    crop_u1, crop_v1 = center_x - crop_window_w_orig / 2.0, center_y - crop_window_h_orig / 2.0
    return {"crop_u1_orig": crop_u1, "crop_v1_orig": crop_v1,
            "crop_window_w_orig": crop_window_w_orig, "crop_window_h_orig": crop_window_h_orig,
            "scale_factor": scale_factor}

def transform_image_v2(image_full, crop_params, target_canvas_size, margin,
                       target_content_dim_w, target_content_dim_h, interpolation, background_value):
    cp = crop_params
    int_h, int_w = int(round(cp["crop_window_h_orig"])), int(round(cp["crop_window_w_orig"]))
    if int_h <= 0: int_h = 1
    if int_w <= 0: int_w = 1

    num_channels = image_full.shape[2] if len(image_full.shape) == 3 else 1
    is_color_or_bgra = num_channels >= 3

    canvas_dtype = image_full.dtype

    # Ensure background_value is a tuple matching num_channels if image_full is multi-channel
    if is_color_or_bgra and not isinstance(background_value, tuple):
        if num_channels == 3: # BGR
            background_value = (background_value, background_value, background_value)
        elif num_channels == 4: # BGRA default to transparent black
            background_value = (0,0,0,0) if background_value == 0 else (int(background_value), int(background_value), int(background_value), 255)

    if is_color_or_bgra:
        cropped_canvas = np.full((int_h, int_w, num_channels), background_value, dtype=canvas_dtype)
    else: # Grayscale
        cropped_canvas = np.full((int_h, int_w), background_value, dtype=canvas_dtype)

    img_h_orig, img_w_orig = image_full.shape[:2]
    src_x1f, src_y1f = max(0, cp["crop_u1_orig"]), max(0, cp["crop_v1_orig"])
    src_x2f, src_y2f = min(img_w_orig, cp["crop_u1_orig"] + cp["crop_window_w_orig"]), min(img_h_orig, cp["crop_v1_orig"] + cp["crop_window_h_orig"])
    dst_x1f, dst_y1f = max(0, -cp["crop_u1_orig"]), max(0, -cp["crop_v1_orig"])
    
    s_x1, s_y1, s_x2, s_y2 = map(int, map(round, [src_x1f, src_y1f, src_x2f, src_y2f]))
    d_x1, d_y1 = map(int, map(round, [dst_x1f, dst_y1f]))
    
    eff_w_src, eff_h_src = s_x2 - s_x1, s_y2 - s_y1

    if eff_w_src > 0 and eff_h_src > 0:
        d_x2_calc, d_y2_calc = d_x1 + eff_w_src, d_y1 + eff_h_src
        d_x2, d_y2 = min(d_x2_calc, int_w), min(d_y2_calc, int_h)
        eff_w_dst, eff_h_dst = d_x2 - d_x1, d_y2 - d_y1
        s_x2_adj, s_y2_adj = s_x1 + eff_w_dst, s_y1 + eff_h_dst
        if eff_w_dst > 0 and eff_h_dst > 0:
            if is_color_or_bgra:
                cropped_canvas[d_y1:d_y2, d_x1:d_x2, :] = image_full[s_y1:s_y2_adj, s_x1:s_x2_adj, :]
            else:
                cropped_canvas[d_y1:d_y2, d_x1:d_x2] = image_full[s_y1:s_y2_adj, s_x1:s_x2_adj]

    resize_w, resize_h = max(1, target_content_dim_w), max(1, target_content_dim_h)
    try:
        resized_content = cv2.resize(cropped_canvas, (resize_w, resize_h), interpolation=interpolation)
    except cv2.error:
        if is_color_or_bgra: resized_content = np.full((resize_h, resize_w, num_channels), background_value, dtype=canvas_dtype)
        else: resized_content = np.full((resize_h, resize_w), background_value, dtype=canvas_dtype)

    if is_color_or_bgra:
        final_canvas = np.full((target_canvas_size, target_canvas_size, num_channels), background_value, dtype=canvas_dtype)
    else:
        final_canvas = np.full((target_canvas_size, target_canvas_size), background_value, dtype=canvas_dtype)

    px, py = margin, margin
    ph, pw = min(resized_content.shape[0], target_content_dim_h), min(resized_content.shape[1], target_content_dim_w)

    if is_color_or_bgra:
        final_canvas[py:py+ph, px:px+pw, :] = resized_content[:ph, :pw, :]
    else:
        final_canvas[py:py+ph, px:px+pw] = resized_content[:ph, :pw]
    return final_canvas

def extract_and_merge_video_frames(sample_src_dir, sample_name, expected_num_frames):
    frames_output_dir = os.path.join(sample_src_dir, 'frames')
    rgb_mkv_path = os.path.join(sample_src_dir, sample_name + '.mkv')
    segm_mkv_path = os.path.join(sample_src_dir, sample_name + '_segm.mkv')

    if os.path.isdir(frames_output_dir):
        try:
            existing_frames = [f for f in os.listdir(frames_output_dir) if f.endswith('.png')]
            if len(existing_frames) == expected_num_frames:
                return True
        except OSError: pass

    if not (os.path.isfile(rgb_mkv_path) and os.path.isfile(segm_mkv_path)):
        print(f"    Warning: Video files for '{sample_name}' not found. Cannot extract frames.")
        return False

    print(f"    Extracting and merging frames for '{sample_name}'...")
    rgb_temp_dir = os.path.join(sample_src_dir, 'rgb_temp_extraction_')
    segm_temp_dir = os.path.join(sample_src_dir, 'segm_temp_extraction_')
    for d in [rgb_temp_dir, segm_temp_dir, frames_output_dir]: ensure_dir(d)

    try:
        cmd_rgb = f'ffmpeg -hide_banner -loglevel error -y -r 30 -i "{rgb_mkv_path}" -r 30 "{os.path.join(rgb_temp_dir, "%04d.png")}"'
        if os.system(cmd_rgb) != 0: print(f"      Error during RGB ffmpeg for {sample_name}."); return False
        cmd_segm = f'ffmpeg -hide_banner -loglevel error -y -r 30 -i "{segm_mkv_path}" -r 30 "{os.path.join(segm_temp_dir, "%04d.png")}"'
        if os.system(cmd_segm) != 0: print(f"      Error during Alpha ffmpeg for {sample_name}."); return False

        rgb_frame_files = sorted([f for f in os.listdir(rgb_temp_dir) if f.endswith('.png')])
        if not rgb_frame_files: print(f"      No RGB frames extracted for {sample_name}."); return False

        for fname in rgb_frame_files:
            frgb_path = os.path.join(rgb_temp_dir, fname)
            fsegm_path = os.path.join(segm_temp_dir, fname)
            fdst_path = os.path.join(frames_output_dir, fname)
            if not os.path.isfile(fsegm_path): continue
            try:
                rgb_img = Image.open(frgb_path)
                alpha_img = Image.open(fsegm_path).convert('L')
                if rgb_img.size != alpha_img.size: alpha_img = alpha_img.resize(rgb_img.size, Image.NEAREST)
                rgb_img.putalpha(alpha_img); rgb_img.save(fdst_path, "PNG")
            except Exception as e: print(f"      Error merging {fname} for {sample_name}: {e}")
        return True
    finally:
        for temp_dir in [rgb_temp_dir, segm_temp_dir]:
            if os.path.isdir(temp_dir):
                try:
                    for f_to_remove in os.listdir(temp_dir): os.remove(os.path.join(temp_dir, f_to_remove))
                    os.rmdir(temp_dir)
                except OSError: pass

def process_frame(reader, sample_name, frame_idx, info,
                  depth_dir, depth_vis_dir, rgb_dir): # Removed pose_dir
    V_human, F_human_smpl = reader.read_human(sample_name, frame_idx, absolute=True)
    F_human = np.array(F_human_smpl)
    V_combined_list, F_combined_list = [V_human], [F_human]
    current_v_offset = V_human.shape[0]
    for gt in list(info['outfit'].keys()):
        V_g = reader.read_garment_vertices(sample_name, gt, frame_idx, absolute=True)
        F_g_list = reader.read_garment_topology(sample_name, gt)
        if V_g.shape[0] > 0 and F_g_list:
            F_g_np = np.array(F_g_list);
            if F_g_np.size == 0: continue
            F_g_tris = quads2tris(F_g_np)
            if F_g_tris.size > 0:
                V_combined_list.append(V_g); F_combined_list.append(F_g_tris + current_v_offset)
                current_v_offset += V_g.shape[0]
    if not V_combined_list: return False
    V_all = np.concatenate(V_combined_list, axis=0)
    if not F_combined_list: return False
    F_all = np.concatenate(F_combined_list, axis=0).astype(np.int32)
    if F_all.size == 0: return False

    renderer = Render(max_depth=MAX_DEPTH_RENDER, depth_scale=1.0)
    renderer.set_mesh(V_all, F_all)
    K, E = intrinsic(), extrinsic(info['camLoc'])
    renderer.set_image(ORIG_IMG_WIDTH, ORIG_IMG_HEIGHT, K, E)
    depth_full_out = renderer.render()
    depth_full_np = depth_full_out.numpy().squeeze() if hasattr(depth_full_out, 'numpy') else np.squeeze(depth_full_out)
    
    mask_full = (depth_full_np < (MAX_DEPTH_RENDER - 1e-3)).astype(np.uint8)
    bbox = get_bbox_from_mask(mask_full);
    if bbox is None: return False
    crop_p = calculate_crop_and_scale_params_from_mask_bbox(bbox, ORIG_IMG_WIDTH, ORIG_IMG_HEIGHT, CONTENT_SIZE, CONTENT_SIZE)
    if crop_p is None: return False

    depth_256 = transform_image_v2(depth_full_np, crop_p, TARGET_IMG_SIZE, MARGIN, CONTENT_SIZE, CONTENT_SIZE, cv2.INTER_NEAREST, MAX_DEPTH_RENDER)
    np.save(os.path.join(depth_dir, f"frame{frame_idx:04d}_depth256.npy"), depth_256)
    depth_vis = depth_256.copy()
    is_bg = depth_vis >= (MAX_DEPTH_RENDER - 1e-3); fg = depth_vis[~is_bg]
    if fg.size > 0:
        min_v, max_v = fg.min(), fg.max()
        if max_v > min_v: # Avoid division by zero if all fg depths are the same
             depth_vis[~is_bg] = (fg - min_v) / (max_v - min_v) * 255.0
        else: # If all fg depths are the same, map to a mid-gray value
            depth_vis[~is_bg] = 128.0
    else: # No foreground pixels
        pass # Background is already set
    depth_vis[is_bg] = 0.0
    cv2.imwrite(os.path.join(depth_vis_dir, f"frame{frame_idx:04d}_depth256.png"), depth_vis.astype(np.uint8))

    # Removed SMPL pose image generation and saving
    # pose_full_bgra = generate_smpl_pose_visualization(reader, sample_name, frame_idx, ORIG_IMG_WIDTH, ORIG_IMG_HEIGHT)
    # pose_256_bgra = transform_image_v2(pose_full_bgra, crop_p, TARGET_IMG_SIZE, MARGIN, CONTENT_SIZE, CONTENT_SIZE, cv2.INTER_LINEAR, (0,0,0,0))
    # cv2.imwrite(os.path.join(pose_dir, f"frame{frame_idx:04d}_smpl_pose256.png"), pose_256_bgra)

    orig_rgb_fname = f"{(frame_idx + 1):04d}.png"
    orig_rgb_path = os.path.join(reader.SRC, sample_name, "frames", orig_rgb_fname)
    if os.path.exists(orig_rgb_path):
        orig_rgb_image_bgra = cv2.imread(orig_rgb_path, cv2.IMREAD_UNCHANGED)
        if orig_rgb_image_bgra is not None:
            if len(orig_rgb_image_bgra.shape) == 2 or orig_rgb_image_bgra.shape[2] == 1 : # Grayscale
                orig_rgb_image_bgra = cv2.cvtColor(orig_rgb_image_bgra, cv2.COLOR_GRAY2BGRA)
            elif orig_rgb_image_bgra.shape[2] == 3: # BGR
                orig_rgb_image_bgra = cv2.cvtColor(orig_rgb_image_bgra, cv2.COLOR_BGR2BGRA)
            # If already BGRA, it's fine.
            transparent_bg_value = (0, 0, 0, 0)
            rgb_256_bgra = transform_image_v2(orig_rgb_image_bgra, crop_p, TARGET_IMG_SIZE, MARGIN, CONTENT_SIZE, CONTENT_SIZE, cv2.INTER_LINEAR, transparent_bg_value)
            cv2.imwrite(os.path.join(rgb_dir, f"frame{frame_idx:04d}_rgb256.png"), rgb_256_bgra)
    return True

def main():
    ensure_dir(OUTPUT_DIR)
    try: reader = DataReader()
    except Exception as e: print(f"Fatal Error: Init DataReader: {e}"); return

    samples_base = reader.SRC
    if not os.path.isdir(samples_base): print(f"Fatal Error: Samples dir '{samples_base}' not found."); return
    
    try:
        sample_names = sorted([d for d in os.listdir(samples_base) if os.path.isdir(os.path.join(samples_base, d))])
    except OSError as e:
        print(f"Fatal Error: Cannot list directory '{samples_base}': {e}"); return

    if not sample_names: print(f"No samples in {samples_base}. Exiting."); return

    print(f"Found {len(sample_names)} samples. Output: {os.path.abspath(OUTPUT_DIR)}")
    total_frames_global, success_frames_global = 0, 0

    for s_idx, s_name in enumerate(sample_names):
        print(f"\nProcessing sample {s_idx+1}/{len(sample_names)}: {s_name}")
        s_out_base = os.path.join(OUTPUT_DIR, s_name)
        # Removed "pose" from subdirectories
        depth_d, vis_d, rgb_d = [os.path.join(s_out_base, sub) for sub in ["depth", "depth_vis", "rgb"]]
        # Removed pose_d from directory creation loop
        for d_path in [depth_d, vis_d, rgb_d]: ensure_dir(d_path)

        try: info = reader.read_info(s_name)
        except Exception as e: print(f"  Error reading info for '{s_name}': {e}. Skipping."); continue
        
        if 'poses' not in info or not isinstance(info['poses'], np.ndarray) or info['poses'].ndim != 2:
            print(f"  Error: 'poses' invalid for '{s_name}'. Skipping."); continue
        
        num_fs = info['poses'].shape[1]
        if num_fs == 0: print(f" -> Sample {s_name} has 0 frames. Skipping."); continue
        
        sample_src_path = os.path.join(reader.SRC, s_name)
        if not extract_and_merge_video_frames(sample_src_path, s_name, num_fs):
            print(f"  Failed to extract/verify frames for '{s_name}'. Skipping processing for this sample.")
            continue

        total_frames_global += num_fs
        print(f" ({num_fs} frames to process)")
        s_success, s_skip = 0, 0
        for f_idx in range(num_fs):
            if (f_idx + 1) % 25 == 0 or f_idx == num_fs -1: print(f"  Frame {f_idx + 1}/{num_fs}", end="\r", flush=True)
            try:
                # Removed pose_d from process_frame call
                if process_frame(reader, s_name, f_idx, info, depth_d, vis_d, rgb_d): s_success +=1
                else: s_skip +=1
            except Exception as e:
                s_skip +=1; print(f"\n    CRITICAL ERROR processing {s_name} frame {f_idx}: {e}")
                import traceback; traceback.print_exc()
        success_frames_global += s_success
        print(" " * 40, end="\r") # Clear progress line
        print(f"  Finished {s_name}. Processed: {s_success}/{num_fs}. Skipped: {s_skip}.")

    print(f"\n\nAll samples processed. Total frames processed globally: {success_frames_global}/{total_frames_global}.")

if __name__ == "__main__":
    main()