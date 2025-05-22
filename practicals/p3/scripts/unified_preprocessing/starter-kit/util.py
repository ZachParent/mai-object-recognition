import numpy as np
import plotly.graph_objects as go # For 3D joint visualization
import cv2 # For pose visualization
import scipy.io as sio
from math import cos, sin

def loadInfo(filename):
	'''
	this function should be called instead of direct sio.loadmat
	as it cures the problem of not properly recovering python dictionaries
	from mat files. It calls the function check keys to cure all entries
	which are still mat-objects
	'''
	data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
	del data['__globals__']
	del data['__header__']
	del data['__version__']
	return _check_keys(data)

def _check_keys(dict):
	'''
	checks if entries in dictionary are mat-objects. If yes
	todict is called to change them to nested dictionaries
	'''
	for key in dict:
		if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
			dict[key] = _todict(dict[key])
	return dict

def _todict(matobj):
	'''
	A recursive function which constructs from matobjects nested dictionaries
	'''
	dict = {}
	for strg in matobj._fieldnames:
		elem = matobj.__dict__[strg]
		if isinstance(elem, sio.matlab.mio5_params.mat_struct):
			dict[strg] = _todict(elem)
		elif isinstance(elem, np.ndarray) and np.any([isinstance(item, sio.matlab.mio5_params.mat_struct) for item in elem]):
			dict[strg] = [None] * len(elem)
			for i,item in enumerate(elem):
				if isinstance(item, sio.matlab.mio5_params.mat_struct):
					dict[strg][i] = _todict(item)
				else:
					dict[strg][i] = item
		else:
			dict[strg] = elem
	return dict

# Computes matrix of rotation around z-axis for 'zrot' radians
def zRotMatrix(zrot):
	c, s = cos(zrot), sin(zrot)
	return np.array([[c, -s, 0],
					 [s,  c, 0],
					 [0,  0, 1]], np.float32)

# --- SMPL Pose Visualization Constants ---
SMPL_BONE_COLORS_RGB = [
    # Left Leg (Magenta) - (j1_idx, j2_idx, RGB_color)
    (0, 1, (255, 0, 255)), (1, 4, (255, 0, 255)), (4, 7, (255, 0, 255)), (7, 10, (255,0,255)),
    # Right Leg (Green)
    (0, 2, (0, 255, 0)), (2, 5, (0, 255, 0)), (5, 8, (0, 255, 0)), (8, 11, (0,255,0)),
    # Torso (Yellow)
    (0, 3, (255, 255, 0)), (3, 6, (255, 255, 0)), (6, 9, (255, 255, 0)),
    # Neck and Head connection
    (9, 12, (255, 255, 0)),   # Spine3 to Neck (Yellow)
    (12, 15, (173, 216, 230)), # Neck to Head (Light Blue)
    # Left Arm (Red)
    (9, 13, (255, 0, 0)), (13, 16, (255, 0, 0)), (16, 18, (255, 0, 0)), (18, 20, (255, 0, 0)), (20, 22, (255,0,0)),
    # Right Arm (Blue)
    (9, 14, (0, 0, 255)), (14, 17, (0, 0, 255)), (17, 19, (0, 0, 255)), (19, 21, (0, 0, 255)), (21, 23, (0,0,255))
]

SMPL_JOINT_COLORS_RGB_MAP = { # joint_idx: RGB_color
    0: (200, 200, 0),  # Pelvis (darker yellow)
    1: (255, 0, 255),  # L_Hip
    2: (0, 255, 0),    # R_Hip
    3: (255, 255, 0),  # Spine1
    4: (255, 0, 255),  # L_Knee
    5: (0, 255, 0),    # R_Knee
    6: (255, 255, 0),  # Spine2
    7: (255, 0, 255),  # L_Ankle
    8: (0, 255, 0),    # R_Ankle
    9: (255, 255, 0),  # Spine3
    10: (255, 0, 255), # L_Foot
    11: (0, 255, 0),   # R_Foot
    12: (255, 255, 0), # Neck
    13: (255, 0, 0),   # L_Collar
    14: (0, 0, 255),   # R_Collar
    15: (255, 255, 255),# Head (White)
    16: (255, 0, 0),   # L_Shoulder
    17: (0, 0, 255),   # R_Shoulder
    18: (255, 0, 0),   # L_Elbow
    19: (0, 0, 255),   # R_Elbow
    20: (255, 0, 0),   # L_Wrist
    21: (0, 0, 255),   # R_Wrist
    22: (255, 0, 0),   # L_Hand
    23: (0, 0, 255)    # R_Hand
}

""" CAMERA """
def intrinsic():
	RES_X = 640
	RES_Y = 480
	f_mm             = 50 # blender default
	sensor_w_mm      = 36 # blender default
	sensor_h_mm = sensor_w_mm * RES_Y / RES_X

	fx_px = f_mm * RES_X / sensor_w_mm;
	fy_px = f_mm * RES_Y / sensor_h_mm;

	u = RES_X / 2;
	v = RES_Y / 2;

	return np.array([[fx_px, 0,     u],
					 [0,     fy_px, v],
					 [0,     0,     1]], np.float32)

def extrinsic(camLoc):
	R_w2bc = np.array([[0, 1, 0],
					   [0, 0, 1],
					   [1, 0, 0]], np.float32)

	T_w2bc = -1 * R_w2bc.dot(camLoc)

	R_bc2cv = np.array([[1,  0,  0],
						[0, -1,  0],
						[0,  0, -1]], np.float32)

	R_w2cv = R_bc2cv.dot(R_w2bc)
	T_w2cv = R_bc2cv.dot(T_w2bc)

	return np.concatenate((R_w2cv, T_w2cv[:,None]), axis=1)

def proj(camLoc):
	return intrinsic().dot(extrinsic(camLoc))

"""
Mesh to UV map
Computes correspondences between 3D mesh and UV map
NOTE: 3D mesh vertices can have multiple correspondences with UV vertices
"""
def mesh2UV(F, Ft):
	m2uv = {v: set() for f in F for v in f}
	for f, ft in zip(F, Ft):
		for v, vt in zip(f, ft):
			m2uv[v].add(vt)
	# m2uv = {k:list(v) for k,v in m2uv.items()}
	return m2uv

# Maps UV coordinates to texture space (pixel)
IMG_SIZE = 2048 # all image textures have this squared size
def uv_to_pixel(vt):
	px = vt * IMG_SIZE # scale to image plane
	px %= IMG_SIZE # wrap to [0, IMG_SIZE]
	# Note that Blender graphic engines invert vertical axis
	return int(px[0]), int(IMG_SIZE - px[1]) # texel X, texel Y

# --- Helper for converting quad faces to triangles ---
def quads2tris(F_quads):
    """
    Converts an array of faces (some quads, some tris) to all triangles.
    Args:
        F_quads (np.ndarray): Array of faces, where each face is a list/array of vertex indices.
    Returns:
        np.ndarray: Array of triangulated faces.
    """
    F_tris = []
    if F_quads is None or len(F_quads) == 0:
        return np.array([], dtype=np.int32).reshape(0,3)
        
    for f in F_quads:
        if len(f) == 3:
            F_tris.append(f)
        elif len(f) == 4:
            F_tris.append([f[0], f[1], f[2]])
            F_tris.append([f[0], f[2], f[3]])
        else:
            # This case should ideally not happen with typical OBJ files or SMPL faces
            # print(f"Warning: Encountered a face with {len(f)} vertices. Skipping.")
            pass # Or handle as an error
    if not F_tris: # If all faces were non-tri/quad or input was empty
        return np.array([], dtype=np.int32).reshape(0,3)
    return np.array(F_tris, dtype=np.int32)


# --- Helper to get SMPL joints in world coordinates ---
def get_world_smpl_joints(reader_instance, sample_name: str, frame_idx: int) -> np.ndarray:
    """
    Computes and returns SMPL 3D joint locations in world coordinates for a given sample and frame.
    This includes applying the SMPL parameters (pose, shape, trans) and the sample's z-rotation.
    Args:
        reader_instance: An instance of DataReader.
        sample_name (str): Name of the sample.
        frame_idx (int): Frame number.
    Returns:
        np.ndarray: 3D joint locations (24 x 3) in world coordinates.
    """
    # print(f"[DEBUG get_world_smpl_joints] Called for sample: {sample_name}, frame: {frame_idx}")
    info = reader_instance.read_info(sample_name)
    gender, pose_params, shape_params, trans_params = reader_instance.read_smpl_params(sample_name, frame_idx)
    
    smpl_model = reader_instance.smpl[gender]
    
    # Get posed and translated joints from SMPL model
    # V_smpl_raw, J_smpl_raw = smpl_model.set_params(pose=pose_params, beta=shape_params, trans=trans_params)
    # The J returned by set_params already includes the translation if trans is provided.
    # It is also affected by pose and shape.
    # smpl_model.update() is called internally by set_params. The self.J attribute will be the posed joints.
    _, J_posed_translated = smpl_model.set_params(pose=pose_params, beta=shape_params, trans=trans_params)

    # Apply the world z-rotation
    zrot_m = zRotMatrix(info['zrot'])
    J_world = (zrot_m @ J_posed_translated.T).T
    # print(f"[DEBUG get_world_smpl_joints] J_world (after zRot) (root, joint 15): {J_world[0]}, {J_world[15]}")
    return J_world

# --- SMPL Pose Visualization Functions ---
def _create_smpl_pose_image_draw(J_3d_world: np.ndarray, P_matrix: np.ndarray,
                                 img_width: int, img_height: int,
                                 limb_thickness: int, joint_radius: int) -> np.ndarray:
    """
    Helper function to draw the SMPL pose on an image.
    Args:
        J_3d_world (np.ndarray): 3D joint locations (N_joints x 3) in world coordinates.
        P_matrix (np.ndarray): Camera projection matrix (3 x 4), from world to image.
        img_width (int): Width of the output image.
        img_height (int): Height of the output image.
        limb_thickness (int): Thickness of the drawn limbs.
        joint_radius (int): Radius of the drawn joints.
    Returns:
        np.ndarray: The pose image (H x W x 3) in BGR format (OpenCV default).
    """
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8) # Black background

    # Project 3D joints to 2D
    J_3d_hom = np.concatenate([J_3d_world, np.ones((J_3d_world.shape[0], 1))], axis=1).T # 4 x N_joints
    J_2d_hom = P_matrix @ J_3d_hom # 3 x N_joints
    
    J_2d = np.zeros((J_3d_world.shape[0], 2), dtype=np.int32)
    
    # Valid points are those in front of the camera (z > small_epsilon)
    valid_mask = J_2d_hom[2, :] > 1e-5 # Check depth is positive
    
    # Avoid division by zero for points with non-positive depth
    # Create a safe version of J_2d_hom[2,:] for division, replacing non-positive with 1 (or a large number)
    # The result for these points will be filtered out by valid_mask anyway
    depth_for_division = np.where(J_2d_hom[2, :] > 1e-5, J_2d_hom[2, :], 1.0)

    J_2d[:, 0] = J_2d_hom[0, :] / depth_for_division # u = x/z
    J_2d[:, 1] = J_2d_hom[1, :] / depth_for_division # v = y/z
    
    # Draw bones
    for j1_idx, j2_idx, color_rgb in SMPL_BONE_COLORS_RGB:
        if valid_mask[j1_idx] and valid_mask[j2_idx]: # Only draw if both joints are valid
            p1 = (J_2d[j1_idx, 0], J_2d[j1_idx, 1])
            p2 = (J_2d[j2_idx, 0], J_2d[j2_idx, 1])
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0]) # OpenCV uses BGR
            cv2.line(img, p1, p2, color_bgr, limb_thickness)

    # Draw joints
    for joint_idx, color_rgb in SMPL_JOINT_COLORS_RGB_MAP.items():
        if joint_idx < J_2d.shape[0] and valid_mask[joint_idx]: # Only draw if joint is valid and index exists
            center = (J_2d[joint_idx, 0], J_2d[joint_idx, 1])
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0]) # OpenCV uses BGR
            cv2.circle(img, center, joint_radius, color_bgr, -1) # -1 for filled circle

    return img

def generate_smpl_pose_visualization(reader_instance, sample_name: str, frame_idx: int,
                                     img_width: int = 640, img_height: int = 480,
                                     limb_thickness: int = 3, joint_radius: int = 5) -> np.ndarray:
    """
    Generates a color-coded SMPL pose image for a given sample and frame.
    Args:
        reader_instance: An instance of DataReader.
        sample_name (str): Name of the sample.
        frame_idx (int): Frame number.
        img_width (int): Width of the output image.
        img_height (int): Height of the output image.
        limb_thickness (int): Thickness of limbs.
        joint_radius (int): Radius of joints.
    Returns:
        np.ndarray: The pose image (H x W x 3) in BGR format.
    """
    # Get joints in world coordinates
    J_world = get_world_smpl_joints(reader_instance, sample_name, frame_idx)

    # Get camera projection matrix P = K @ E
    info = reader_instance.read_info(sample_name)
    P_matrix = proj(info['camLoc']) # proj function from util.py

    pose_image = _create_smpl_pose_image_draw(J_world, P_matrix, img_width, img_height,
                                              limb_thickness, joint_radius)
    return pose_image

def visualize_3d_smpl_joints_plotly(J_3d_coords: np.ndarray, title: str) -> 'go.Figure':
    """
    Visualizes 3D SMPL joints and bones using Plotly.
    Args:
        J_3d_coords (np.ndarray): 3D joint coordinates (N_joints x 3).
        title (str): Title for the plot.
    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    traces = []

    # Plot joints as scatter points
    joint_x = J_3d_coords[:, 0]
    joint_y = J_3d_coords[:, 1]
    joint_z = J_3d_coords[:, 2]
    joint_colors_list = [f'rgb({SMPL_JOINT_COLORS_RGB_MAP.get(i, (128,128,128))[0]},{SMPL_JOINT_COLORS_RGB_MAP.get(i, (128,128,128))[1]},{SMPL_JOINT_COLORS_RGB_MAP.get(i, (128,128,128))[2]})' for i in range(J_3d_coords.shape[0])]

    traces.append(go.Scatter3d(x=joint_x, y=joint_y, z=joint_z, mode='markers',
                               marker=dict(size=4, color=joint_colors_list, opacity=0.8),
                               name='Joints'))

    # Plot bones as lines
    for j1_idx, j2_idx, color_rgb in SMPL_BONE_COLORS_RGB:
        if j1_idx < J_3d_coords.shape[0] and j2_idx < J_3d_coords.shape[0]: # Check indices
            p1 = J_3d_coords[j1_idx]
            p2 = J_3d_coords[j2_idx]
            traces.append(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                                    mode='lines',
                                    line=dict(color=f'rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})', width=5),
                                    name=f'Bone {j1_idx}-{j2_idx}'))

    fig = go.Figure(data=traces)
    fig.update_layout(title=title,
                      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                                 aspectmode='data'), # 'data' or 'cube' or 'auto'
                      showlegend=False) # Legend can be crowded
    return fig