import os
import sys
import numpy as np
from PIL import Image
# from cv2 import VideoCapture # Not used in this file


from .smpl_np import SMPLModel # Assuming smpl_np.py is in a 'smpl' subdirectory

sys.path.append('practicals/p3/scripts/unified_preprocessing/starter-kit')
from util import loadInfo, zRotMatrix, proj, mesh2UV, uv_to_pixel # Assuming util.py is in the same directory or findable
from IO import readOBJ, readPC2Frame # Assuming IO.py is in the same directory or findable

class DataReader:

    def __init__(self):
        # Data paths
        # Correctly determine the path to 'Samples/' relative to this file's location.
        # __file__ is the path to read.py
        # os.path.dirname(__file__) is the directory containing read.py (e.g., DataReader/)
        # os.path.abspath(...) makes it an absolute path.
        # '/../Samples/' goes up one level from DataReader/ and then into Samples/
        self.SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), 'whereveryouplacetheextracteddataset'))
        
        # SMPL model
        # Path to the 'smpl' directory, assuming it's a subdirectory of where read.py is.
        smpl_dir_path = os.path.join(os.path.dirname(__file__), 'whereveryouplacethesmplmodels')
        
        # Check if SMPL model files exist
        model_f_path = os.path.join(smpl_dir_path, 'model_f.pkl')
        model_m_path = os.path.join(smpl_dir_path, 'model_m.pkl')

        if not os.path.exists(model_f_path):
            raise FileNotFoundError(f"SMPL female model not found at: {model_f_path}. Please ensure it's in DataReader/smpl/")
        if not os.path.exists(model_m_path):
            raise FileNotFoundError(f"SMPL male model not found at: {model_m_path}. Please ensure it's in DataReader/smpl/")
            
        self.smpl = {
            'f': SMPLModel(model_f_path),
            'm': SMPLModel(model_m_path)
        }
        
    """ 
    Read sample info 
    Input:
    - sample: name of the sample e.g.:'01_01_s0'
    """
    def read_info(self, sample):
        # The info file is now expected to be 'info.mat' as per CLOTH3D dataset
        info_path = os.path.join(self.SRC, sample, 'info.mat')
        if not os.path.exists(info_path):
            # Fallback for original script expecting 'info' without extension
            info_path_no_ext = os.path.join(self.SRC, sample, 'info')
            if os.path.exists(info_path_no_ext):
                info_path = info_path_no_ext
            else:
                raise FileNotFoundError(f"Sample info file not found at {info_path} or {info_path_no_ext} for sample '{sample}'")
        return loadInfo(info_path)
        
    """ Human data """
    """
    Read SMPL parameters for the specified sample and frame
    Inputs:
    - sample: name of the sample
    - frame: frame number
    """
    def read_smpl_params(self, sample, frame):
        # Read sample data
        info = self.read_info(sample)
        # SMPL parameters
        gender = 'm' if info['gender'] else 'f'
        
        if frame >= info['poses'].shape[1]:
            raise IndexError(f"Frame index {frame} is out of bounds for sample {sample} which has {info['poses'].shape[1]} frames.")

        pose = info['poses'][:, frame].reshape(self.smpl[gender].pose_shape)
        shape = info['shape'] # shape is static per sample
        trans = info['trans'][:, frame].reshape(self.smpl[gender].trans_shape)
        return gender, pose, shape, trans
    
    """
    Computes human mesh for the specified sample and frame
    Inputs:
    - sample: name of the sample
    - frame: frame number
    - absolute: True for absolute vertex locations (world coordinates after zRot and trans),
                False for locations relative to the model's origin after zRot but before global trans.
    Outputs:
    - V: human mesh vertices
    - F: mesh faces
    """
    def read_human(self, sample, frame, absolute=True):
        info = self.read_info(sample)
        gender, pose_params, shape_params, trans_params = self.read_smpl_params(sample, frame)

        # Step 1: Get POSED vertices and joints RELATIVE TO THE MODEL'S CANONICAL ORIGIN.
        # SMPLModel's set_params (your code (3).py version) when trans=None will compute
        # verts and J in a local coordinate system (posed, shaped, but origin-centered).
        verts_local_posed, J_local_posed = self.smpl[gender].set_params(
            pose=pose_params,
            beta=shape_params,
            trans=None # Get local posed geometry first
        )

        # Step 2: Center the locally posed vertices around their *posed root joint*
        # J_local_posed[0:1] is the location of the posed root joint, relative to the model's canonical origin.
        # Subtracting this makes the model's reference point (its posed root) be at (0,0,0) for rotation.
        verts_to_rotate = verts_local_posed - J_local_posed[0:1]

        # Step 3: Apply Z-axis rotation (around the now (0,0,0) effective root point).
        zRot_matrix = zRotMatrix(info['zrot'])
        verts_rotated = zRot_matrix.dot(verts_to_rotate.T).T

        # Step 4: Apply global translation if requested.
        if absolute:
            final_verts = verts_rotated + trans_params.reshape([1, 3])
        else:
            final_verts = verts_rotated # Remains centered (on its root), rotated, at origin

        return final_verts, self.smpl[gender].faces
    
    """ Garment data """
    """
    Reads garment vertices location for the specified sample, garment and frame
    Inputs:
    - sample: name of the sample
    - garment: type of garment (e.g.: 'Tshirt', 'Jumpsuit', ...)
    - frame: frame number
    - absolute: True for absolute vertex locations (world coords after zRot and trans),
                False for locations relative to the model's origin after zRot but before global trans.
    Outputs:
    - V: 3D vertex locations for the specified sample, garment and frame
    """
    def read_garment_vertices(self, sample, garment, frame, absolute=True):
        # Read garment vertices (these are relative to SMPL root joint as per .pc16 spec)
        pc16_path = os.path.join(self.SRC, sample, garment + '.pc16')
        if not os.path.exists(pc16_path):
            raise FileNotFoundError(f"Garment animation file not found: {pc16_path}")
            
        V_relative_to_root = readPC2Frame(pc16_path, frame, True) # This is V_garment - J_smpl_root_posed_local
        
        # The .pc16 vertices are ALREADY relative to the SMPL root joint for that frame.
        # This means they are effectively 'pre-centered' for the z-rotation if we consider
        # the SMPL root as the center.
        # The 'trans' in info.mat is the world location of this SMPL root joint.

        # Step 1: Apply Z-axis rotation
        # Garment vertices are already relative to the (untranslated) root, so rotate directly.
        info = self.read_info(sample) # Need zrot and trans
        zRot_matrix = zRotMatrix(info['zrot'])
        verts_rotated = zRot_matrix.dot(V_relative_to_root.T).T
        
        # Step 2: Apply global translation of the SMPL root if requested.
        if absolute:
            trans_params = info['trans'][:, frame].reshape([1, 3])
            final_verts = verts_rotated + trans_params
        else:
            final_verts = verts_rotated # Remains centered on its root, rotated, at origin

        return final_verts


    """
    Reads garment faces for the specified sample and garment
    Inputs:
    - sample: name of the sample
    - garment: type of garment (e.g.: 'Tshirt', 'Jumpsuit', ...)
    Outputs:
    - F: mesh faces (list of lists)
    """
    def read_garment_topology(self, sample, garment):
        obj_path = os.path.join(self.SRC, sample, garment + '.obj')
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Garment OBJ file not found: {obj_path}")
        _, F, _, _ = readOBJ(obj_path) # readOBJ returns V, F, Vt, Ft
        return F # F is a list of lists

    """	
    Reads garment UV map for the specified sample and garment
    Inputs:
    - sample: name of the sample
    - garment: type of garment (e.g.: 'Tshirt', 'Jumpsuit', ...)
    Outputs:
    - Vt: UV map vertices (NumPy array)
    - Ft: UV map faces (list of lists)
    """
    def read_garment_UVMap(self, sample, garment):
        obj_path = os.path.join(self.SRC, sample, garment + '.obj')
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Garment OBJ file not found: {obj_path}")
        _, _, Vt, Ft = readOBJ(obj_path)	
        return Vt, Ft	

    """
    Reads vertex colors of the specified sample and garment
    Inputs:
    - sample: name of the sample
    - garment: type of garment (e.g.: 'Tshirt', 'Jumpsuit', ...)
    - F: mesh faces (list of lists, from read_garment_topology)
    - Vt: UV map vertices (NumPy array, from read_garment_UVMap)
    - Ft: UV map faces (list of lists, from read_garment_UVMap)
    Output
    - C: RGB colors (NumPy array, N_verts x 3 or 1x3 for plain color)
    """
    def read_garment_vertex_colors(self, sample, garment, F, Vt, Ft):
        info = self.read_info(sample)
        texture_info = info['outfit'][garment]['texture']
        
        if texture_info['type'] == 'color':
            # Ensure data is a NumPy array and scaled correctly
            color_data = np.array(texture_info['data'])
            if color_data.max() <= 1.0: # Assuming colors are 0-1 range if not already 0-255
                return (255 * color_data).astype(np.uint8)
            return color_data.astype(np.uint8)

        # Image texture
        img_texture_filename = info['outfit'][garment]['texture'].get('filename', garment + '.png') # CLOTH3D often names it based on garment type
        img_path = os.path.join(self.SRC, sample, img_texture_filename)
        if not os.path.exists(img_path):
             raise FileNotFoundError(f"Garment texture image not found: {img_path}")

        try:
            img = Image.open(img_path).convert('RGB') # Ensure 3 channels
        except Exception as e:
            raise IOError(f"Could not open or convert texture image {img_path}: {e}")

        if Vt is None or len(Vt) == 0 or Ft is None or len(Ft) == 0:
            # Fallback: if no UV map, perhaps return a default color or average image color
            print(f"Warning: No UV map data for {sample}/{garment}. Returning average image color.")
            avg_color = np.array(img).reshape(-1,3).mean(axis=0)
            num_verts_in_F = 0
            if F and F[0] is not None : num_verts_in_F = max(v for face in F for v in face) + 1 if F else 0
            return np.tile(avg_color.astype(np.uint8), (num_verts_in_F if num_verts_in_F > 0 else 1, 1))


        # Get color of each UV vertex
        # uv_to_pixel expects Vt elements, img.size (width, height)
        img_w, img_h = img.size
        try:
            # Ensure Vt is valid for pixel lookup
            valid_vt = np.clip(Vt, 0.0, 1.0) # Clip UVs to be safe, though ideally they are 0-1
            pixel_coords = [uv_to_pixel(vt_coord, img_w, img_h) for vt_coord in valid_vt]
            colors_at_uv_verts = np.array([img.getpixel(px_coord) for px_coord in pixel_coords], dtype=np.uint8)
        except Exception as e:
            print(f"Error getting pixel colors from UVs for {sample}/{garment}: {e}")
            # Fallback or re-raise
            avg_color = np.array(img).reshape(-1,3).mean(axis=0)
            num_verts_in_F = 0
            if F and F[0] is not None : num_verts_in_F = max(v for face in F for v in face) + 1 if F else 0
            return np.tile(avg_color.astype(np.uint8), (num_verts_in_F if num_verts_in_F > 0 else 1, 1))


        # Compute correspondence between V (mesh) and Vt (UV map)
        # F and Ft are list of lists. mesh2UV expects this.
        m2uv = mesh2UV(F, Ft) # Returns dict {mesh_v_idx: {uv_v_idx1, uv_v_idx2, ...}}
        
        # Determine the maximum vertex index in F to size the output color array
        max_mesh_vertex_idx = 0
        if F and F[0] is not None: # Ensure F is not empty or list of Nones
            for face in F:
                if face: # Ensure face is not empty or None
                    max_mesh_vertex_idx = max(max_mesh_vertex_idx, max(face))
        
        num_mesh_vertices = max_mesh_vertex_idx + 1
        
        # Initialize vertex colors (e.g., to black or a default color)
        vertex_colors_rgb = np.zeros((num_mesh_vertices, 3), dtype=np.uint8)
        
        for mesh_v_idx in range(num_mesh_vertices):
            uv_indices_for_mesh_v = m2uv.get(mesh_v_idx)
            if uv_indices_for_mesh_v:
                # Average the colors of the corresponding UV vertices
                colors_to_average = colors_at_uv_verts[list(uv_indices_for_mesh_v)]
                vertex_colors_rgb[mesh_v_idx] = np.mean(colors_to_average, axis=0).astype(np.uint8)
            # else:
                # Mesh vertex has no UV mapping, color remains as initialized (e.g., black)
                # print(f"Warning: Mesh vertex {mesh_v_idx} has no UV mapping for {sample}/{garment}")


        return vertex_colors_rgb
        
    """ Scene data """
    """
    Read camera location and compute projection matrix
    Input:
    - sample: name of the sample
    Output:
    - P: camera projection matrix (3 x 4)
    """
    def read_camera(self, sample):
        info = self.read_info(sample)
        camLoc = info['camLoc']
        return proj(camLoc) # proj is from util.py which uses intrinsic() and extrinsic()
        
# TESTING
if __name__ == '__main__':
    print("DataReader Test Script")
    # Ensure 'Samples' directory is correctly located relative to this script for testing.
    # Typically, if this script is in project_root/DataReader/read.py,
    # and Samples is in project_root/Samples/, then self.SRC should be correct.
    
    # Find a sample in the Samples directory
    try:
        # Create a dummy DataReader instance to access self.SRC
        # This assumes that the __init__ can find the SMPL models.
        # For testing, you might need to adjust paths or provide dummy SMPL pkl files.
        _test_reader_instance_for_path = DataReader()
        samples_dir = _test_reader_instance_for_path.SRC
        available_samples = [d for d in os.listdir(samples_dir) if os.path.isdir(os.path.join(samples_dir, d))]
        if not available_samples:
            print(f"No samples found in {samples_dir} for testing.")
            sys.exit(1)
        sample = available_samples[0] # Take the first available sample
        print(f"Using sample: {sample} for testing.")
    except Exception as e:
        print(f"Error during test setup (finding sample or DataReader init): {e}")
        print("Make sure SMPL model files are in DataReader/smpl/ and 'Samples' directory is accessible.")
        sys.exit(1)

    frame = 0
    
    reader = DataReader() # Initialize for real test
    
    print(f"\n--- Testing read_info for sample: {sample} ---")
    try:
        info = reader.read_info(sample)
        print(f"Successfully read info. Gender: {'male' if info['gender'] else 'female'}, Num poses: {info['poses'].shape[1]}")
        garment_types = list(info['outfit'].keys())
        if not garment_types:
            print("No garments found for this sample in info file.")
            sys.exit(1)
        garment = garment_types[0] # Take the first garment
        print(f"Using garment: {garment} for further tests.")
    except Exception as e:
        print(f"Error reading info: {e}")
        sys.exit(1)

    print(f"\n--- Testing read_smpl_params for frame: {frame} ---")
    gender, pose, shape, trans = reader.read_smpl_params(sample, frame)
    print(f"Gender: {gender}, Pose shape: {pose.shape}, Shape shape: {shape.shape}, Trans shape: {trans.shape}")

    print(f"\n--- Testing read_human (absolute=True) for frame: {frame} ---")
    V_human_abs, F_human = reader.read_human(sample, frame, absolute=True)
    print(f"Human verts shape: {V_human_abs.shape}, Human faces type: {type(F_human)}, Num faces: {len(F_human) if F_human is not None else 0}")

    print(f"\n--- Testing read_human (absolute=False) for frame: {frame} ---")
    V_human_rel, _ = reader.read_human(sample, frame, absolute=False)
    print(f"Human verts (relative) shape: {V_human_rel.shape}")

    print(f"\n--- Testing read_garment_vertices (absolute=True) for garment: {garment}, frame: {frame} ---")
    try:
        V_garment_abs = reader.read_garment_vertices(sample, garment, frame, absolute=True)
        print(f"Garment verts (absolute) shape: {V_garment_abs.shape}")
    except FileNotFoundError as e:
        print(f"Skipping garment vertices test: {e}")
    
    print(f"\n--- Testing read_garment_topology for garment: {garment} ---")
    try:
        F_garment = reader.read_garment_topology(sample, garment)
        print(f"Garment faces type: {type(F_garment)}, Num faces: {len(F_garment) if F_garment is not None else 0}")
        if F_garment and F_garment[0] is not None : print(f"First garment face: {F_garment[0]}")
    except FileNotFoundError as e:
        print(f"Skipping garment topology test: {e}")
        F_garment = None # Set to None if not found, for vertex color test

    print(f"\n--- Testing read_garment_UVMap for garment: {garment} ---")
    try:
        Vt_garment, Ft_garment = reader.read_garment_UVMap(sample, garment)
        print(f"Garment UV verts shape: {Vt_garment.shape if Vt_garment is not None else 'None'}, UV faces type: {type(Ft_garment)}, Num UV faces: {len(Ft_garment) if Ft_garment is not None else 0}")
    except FileNotFoundError as e:
        print(f"Skipping garment UV map test: {e}")
        Vt_garment, Ft_garment = None, None # Set to None for vertex color test

    if F_garment is not None and Vt_garment is not None and Ft_garment is not None:
        print(f"\n--- Testing read_garment_vertex_colors for garment: {garment} ---")
        try:
            C_garment = reader.read_garment_vertex_colors(sample, garment, F_garment, Vt_garment, Ft_garment)
            print(f"Garment vertex colors shape: {C_garment.shape if C_garment is not None else 'None'}")
            if C_garment is not None and C_garment.ndim == 2 and C_garment.shape[0] > 0:
                 print(f"First vertex color: {C_garment[0]}")
            elif C_garment is not None and C_garment.ndim == 1:
                 print(f"Plain color: {C_garment}")
        except FileNotFoundError as e: # Texture image might be missing
            print(f"Skipping garment vertex colors test due to missing file: {e}")
        except Exception as e:
            print(f"Error in read_garment_vertex_colors: {e}")
    else:
        print(f"\n--- Skipping read_garment_vertex_colors due to missing topology or UV map for garment: {garment} ---")


    print(f"\n--- Testing read_camera for sample: {sample} ---")
    P_cam = reader.read_camera(sample)
    print(f"Camera projection matrix shape: {P_cam.shape}")

    print("\n--- DataReader Test Script Finished ---")