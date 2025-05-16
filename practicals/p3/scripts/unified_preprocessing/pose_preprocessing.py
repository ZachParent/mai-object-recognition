import os
import sys
import numpy as np
import cv2 # For saving images and image manipulation
# PIL.Image is no longer needed as extract_and_merge_video_frames is removed.

# Ensure DataReader and its dependencies are in the Python path
sys.path.append('practicals/p3/demo/cloth3d/unified_preprocessing')
sys.path.append('practicals/p3/demo/cloth3d/unified_preprocessing/starter-kit')
sys.path.append('practicals/p3/demo/cloth3d/unified_preprocessing/pose_reader')

from pose_reader.read import DataReader
from depth_render import Render
from util import intrinsic, extrinsic

# --- Configuration ---
OUTPUT_DIR = 'practicals/p3/demo/cloth3d/unified_preprocessing/preprocessed_dataset'

ORIG_IMG_WIDTH = 640
ORIG_IMG_HEIGHT = 480
TARGET_IMG_SIZE = 256
MARGIN = 10
CONTENT_SIZE = TARGET_IMG_SIZE - 2 * MARGIN # This is 236, so content area is 236x236

# MAX_DEPTH_RENDER = 10.0 # Removed: Not used without depth rendering

# --- Color Configuration for SMPL Pose Visualization ---
# Colors are in (B, G, R, A) format, with A=255 for opaque.

# DEFAULT_BONE_COLOR is used for bones whose child joint index is not in SMPL_CHILD_JOINT_TO_COLOR_MAP.
DEFAULT_BONE_COLOR = (128, 128, 128, 255)  # Gray

# JOINT_COLOR_BGRA is used for all joints (the small circles).
JOINT_COLOR_BGRA = (255, 255, 255, 255) # White, Opaque

# SMPL_CHILD_JOINT_TO_COLOR_MAP maps the child joint index of a bone to its color and a descriptive name.
# Standard SMPL model has 24 joints (0-23). Bones are defined by (parent, child) pairs.
# The child indices range from 1 to 23 for these bones.
SMPL_CHILD_JOINT_TO_COLOR_MAP = {
    # Child Joint Index: (Bone Name, (B, G, R, A))

    # Torso (Green shades)
    3:  ("Pelvis->Spine1", (0, 128, 0, 255)),    # Dark Green
    6:  ("Spine1->Spine2", (0, 180, 0, 255)),    # Medium Green
    9:  ("Spine2->Spine3", (0, 220, 0, 255)),    # Light Green

    # Head (Yellow)
    15: ("Neck->Head", (0, 255, 255, 255)),   # Yellow

    # Left Arm (Blue shades)
    16: ("LCollar->LShoulder", (255, 100, 0, 255)), # Cyan-Blue (Shoulder to Elbow)
    18: ("LShoulder->LElbow", (200, 50, 0, 255)),    # Medium Blue (Elbow to Wrist)
    20: ("LElbow->LWrist", (150, 0, 0, 255)),    # Dark Blue (Wrist to Hand)
    22: ("LWrist->LHand", (100, 0, 0, 255)),    # Darkest Blue (Hand - if exists)

    # Right Arm (Red shades)
    17: ("RCollar->RShoulder", (0, 100, 255, 255)), # Orange-Red
    19: ("RShoulder->RElbow", (0, 50, 200, 255)),   # Medium Red
    21: ("RElbow->RWrist", (0, 0, 150, 255)),   # Dark Red
    23: ("RWrist->RHand", (0, 0, 100, 255)),    # Darkest Red (Hand - if exists)

    # Left Leg (Magenta/Purple shades)
    4:  ("LHip->LKnee", (255, 0, 200, 255)),     # Magenta
    7:  ("LKnee->LAnkle", (200, 0, 150, 255)),    # Medium Purple
    10: ("LAnkle->LFoot", (150, 0, 100, 255)),   # Dark Purple

    # Right Leg (Teal/Cyan shades)
    5:  ("RHip->RKnee", (0, 200, 200, 255)),    # Teal
    8:  ("RKnee->RAnkle", (0, 150, 150, 255)),   # Medium Teal
    11: ("RAnkle->RFoot", (0, 100, 100, 255)),   # Dark Teal

    # Connectors (Neutral Grays - these connect torso to limbs/head)
    12: ("Spine3->Neck", (200, 200, 200, 255)),       # Light Gray (Torso to Neck)
    13: ("Spine3->LCollar", (220, 220, 220, 255)),    # Lighter Gray (Torso to LClavicle)
    14: ("Spine3->RCollar", (240, 240, 240, 255)),    # Lightest Gray (Torso to RClavicle)
    1:  ("Pelvis->LHip", (180, 180, 180, 255)),       # Gray (Pelvis to LHip)
    2:  ("Pelvis->RHip", (160, 160, 160, 255)),       # Darker Gray (Pelvis to RHip)
}

# --- Helper Functions ---

def generate_smpl_pose_visualization(reader, sample_name, frame_idx, img_width, img_height):
    # Pose image will be BGRA with transparent background
    pose_image = np.zeros((img_height, img_width, 4), dtype=np.uint8) # Use 4 channels (BGRA)
    info = reader.read_info(sample_name)
    gender = 'm' if info['gender'] else 'f'
    _, pose_params, shape_params, trans_params = reader.read_smpl_params(sample_name, frame_idx)
    _, J_3d_relative_to_root = reader.smpl[gender].set_params(pose=pose_params, beta=shape_params, trans=None)
    J_3d_relative_to_root -= J_3d_relative_to_root[0:1]
    zrot = info['zrot']
    c, s = np.cos(zrot), np.sin(zrot)
    zRotMat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    J_3d_rotated = zRotMat.dot(J_3d_relative_to_root.T).T
    J_3d_world = J_3d_rotated + trans_params.reshape(1,3)
    K_matrix, E_matrix = intrinsic(), extrinsic(info['camLoc'])
    P_matrix = K_matrix @ E_matrix
    J_3d_homog = np.hstack((J_3d_world, np.ones((J_3d_world.shape[0], 1))))
    J_2d_homog = (P_matrix @ J_3d_homog.T).T
    J_2d_homog[:, 2][J_2d_homog[:, 2] < 1e-5] = 1e-5 # Avoid division by zero or very small depth
    J_2d = J_2d_homog[:, :2] / J_2d_homog[:, 2, np.newaxis]
    parent_lookup = {c: p for p, c in zip(reader.smpl[gender].kintree_table[0,:], reader.smpl[gender].kintree_table[1,:])}
    if 0 not in parent_lookup: parent_lookup[0] = -1

    # Draw bones (lines)
    for i in range(J_2d.shape[0]): # i is the child joint index in a bone
        parent_idx = parent_lookup.get(i, -1)
        if parent_idx == -1 or parent_idx >= J_2d.shape[0]:
            continue # Skip root's "parent" or invalid parent index

        # Determine bone color based on the child joint index 'i'
        color_data = SMPL_CHILD_JOINT_TO_COLOR_MAP.get(i)
        if color_data is None:
            current_line_color_bgra = DEFAULT_BONE_COLOR
        else:
            current_line_color_bgra = color_data[1] # Get the (B,G,R,A) tuple

        pt_child = (int(round(J_2d[i, 0])), int(round(J_2d[i, 1])))
        pt_parent = (int(round(J_2d[parent_idx, 0])), int(round(J_2d[parent_idx, 1])))

        if (0 <= pt_child[0] < img_width and 0 <= pt_child[1] < img_height and
            0 <= pt_parent[0] < img_width and 0 <= pt_parent[1] < img_height):
            cv2.line(pose_image, pt_child, pt_parent, current_line_color_bgra, 1, lineType=cv2.LINE_AA)

    # Draw joints (circles)
    for i in range(J_2d.shape[0]):
        pt = (int(round(J_2d[i,0])), int(round(J_2d[i,1])))
        if (0 <= pt[0] < img_width and 0 <= pt[1] < img_height):
            # All joints are drawn with the globally defined JOINT_COLOR_BGRA
            cv2.circle(pose_image, pt, 2, JOINT_COLOR_BGRA, -1, lineType=cv2.LINE_AA)
            
    return pose_image

# def quads2tris(F_quads): # Removed: Only used for garment mesh processing for depth rendering
#     out_tris = []
#     if F_quads is None or len(F_quads) == 0:
#         return np.array([], dtype=np.int32).reshape(0,3)
#     for f in F_quads:
#         if f is None: continue # Should not happen with valid OBJ
#         if len(f) == 3: out_tris.append(f)
#         elif len(f) == 4:
#             out_tris.append([f[0], f[1], f[2]]); out_tris.append([f[0], f[2], f[3]])
#     if not out_tris: return np.array([], dtype=np.int32).reshape(0,3)
#     return np.array(out_tris, np.int32)

def ensure_dir(directory):
    if not os.path.exists(directory): os.makedirs(directory)

def get_bbox_from_mask(mask):
    # mask is expected to be a 2D numpy array (boolean or uint8)
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

# def extract_and_merge_video_frames(...): # Removed: Not generating RGB images

def process_frame(reader, sample_name, frame_idx, info, pose_dir):
    # --- Pose Generation ---
    pose_full_bgra = generate_smpl_pose_visualization(reader, sample_name, frame_idx, ORIG_IMG_WIDTH, ORIG_IMG_HEIGHT)

    # --- Bounding Box Calculation from Pose Alpha Channel ---
    # Use the alpha channel of the pose image as a mask
    pose_alpha_mask = pose_full_bgra[:, :, 3] > 0  # Pixels with any transparency are part of the pose
    bbox = get_bbox_from_mask(pose_alpha_mask)
    
    if bbox is None:
        # This means the pose image was entirely transparent, perhaps no joints visible.
        # print(f"Warning: No bounding box found for pose in {sample_name} frame {frame_idx}. Skipping frame.")
        return False
        
    crop_p = calculate_crop_and_scale_params_from_mask_bbox(bbox, ORIG_IMG_WIDTH, ORIG_IMG_HEIGHT, CONTENT_SIZE, CONTENT_SIZE)
    if crop_p is None:
        # print(f"Warning: Could not calculate crop parameters for pose in {sample_name} frame {frame_idx}. Skipping frame.")
        return False

    # --- Transform and Save Pose Image ---
    pose_256_bgra = transform_image_v2(pose_full_bgra, crop_p, TARGET_IMG_SIZE, MARGIN, CONTENT_SIZE, CONTENT_SIZE, cv2.INTER_LINEAR, (0,0,0,0)) # Transparent background
    cv2.imwrite(os.path.join(pose_dir, f"frame{frame_idx:04d}_smpl_pose256.png"), pose_256_bgra)

    # --- Depth and RGB sections removed ---
    
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
        # Only pose directory is needed now
        pose_d = os.path.join(s_out_base, "pose")
        ensure_dir(pose_d) # Ensure pose directory is created

        try: info = reader.read_info(s_name)
        except Exception as e: print(f"  Error reading info for '{s_name}': {e}. Skipping."); continue
        
        if 'poses' not in info or not isinstance(info['poses'], np.ndarray) or info['poses'].ndim != 2:
            print(f"  Error: 'poses' invalid for '{s_name}'. Skipping."); continue
        
        num_fs = info['poses'].shape[1]
        if num_fs == 0: print(f" -> Sample {s_name} has 0 frames. Skipping."); continue
        
        # Removed: extract_and_merge_video_frames call, as RGB frames are not processed.
        # sample_src_path = os.path.join(reader.SRC, s_name)
        # if not extract_and_merge_video_frames(sample_src_path, s_name, num_fs):
        #     print(f"  Failed to extract/verify frames for '{s_name}'. Skipping processing for this sample.")
        #     continue

        total_frames_global += num_fs
        print(f" ({num_fs} frames to process for poses)")
        s_success, s_skip = 0, 0
        for f_idx in range(num_fs):
            if (f_idx + 1) % 25 == 0 or f_idx == num_fs -1: print(f"  Frame {f_idx + 1}/{num_fs}", end="\r", flush=True)
            try:
                # Call process_frame with only the pose_dir
                if process_frame(reader, s_name, f_idx, info, pose_d): s_success +=1
                else: s_skip +=1
            except Exception as e:
                s_skip +=1; print(f"\n    CRITICAL ERROR processing {s_name} frame {f_idx}: {e}")
                import traceback; traceback.print_exc()
        success_frames_global += s_success
        print(" " * 40, end="\r") # Clear progress line
        print(f"  Finished {s_name}. Processed poses: {s_success}/{num_fs}. Skipped: {s_skip}.")

    print(f"\n\nAll samples processed. Total pose frames generated: {success_frames_global}/{total_frames_global}.")

if __name__ == "__main__":
    main()