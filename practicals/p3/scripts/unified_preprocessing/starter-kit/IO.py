import os
import numpy as np
from struct import pack, unpack

"""
Reads OBJ files
Only handles vertices, faces and UV maps
Input:
- file: path to .obj file
Outputs:
- V: 3D vertices (NumPy array)
- F: 3D faces (list of lists, each sublist contains 3 integer indices)
- Vt: UV vertices (NumPy array)
- Ft: UV faces (list of lists, each sublist contains 3 integer indices)
Correspondence between mesh and UV map is implicit in F to Ft correspondences
If no UV map data in .obj file, it shall return Vt=None and Ft=[]
"""
def readOBJ(file):
	V_list, Vt_list, F_list, Ft_list = [], [], [], [] # Use temporary lists
	with open(file, 'r') as f:
		T = f.readlines()
	for t_line in T: # Renamed t to t_line to avoid conflict if t is used later
		t_line = t_line.strip() # Remove leading/trailing whitespace
		if not t_line: continue # Skip empty lines

		# 3D vertex
		if t_line.startswith('v '):
			v = [float(n) for n in t_line.replace('v ','').split(' ')]
			V_list.append(v)
		# UV vertex
		elif t_line.startswith('vt '):
			v = [float(n) for n in t_line.replace('vt ','').split(' ')]
			Vt_list.append(v)
		# Face
		elif t_line.startswith('f '):
			parts = t_line.replace('f ','').split(' ')
			idx_vert = []
			idx_uv = []
            # Check if UVs are present in this face definition by looking at the first vertex spec
			has_uv = '/' in parts[0] and len(parts[0].split('/')) > 1 and parts[0].split('/')[1] != ''


			for part_group in parts:
				indices = part_group.split('/')
				idx_vert.append(int(indices[0]) - 1)
				if has_uv:
					if len(indices) > 1 and indices[1]: # Ensure UV index exists and is not empty
						idx_uv.append(int(indices[1]) - 1)
					# else: # This case implies missing UV for a vertex in a face that otherwise has UVs
						# This could be an issue, or means mix-and-match, usually not good for consistent UV mapping
						# For now, if Ft is to be consistent, we might need a placeholder or error
						# print(f"Warning: Inconsistent UV data in face: {t_line} in file {file}")
						# pass # Or append a placeholder like -1, or ensure idx_uv length matches idx_vert
			
			if len(idx_vert) == 3: # It's already a triangle
				F_list.append(idx_vert)
				if has_uv and len(idx_uv) == 3:
					Ft_list.append(idx_uv)
				elif has_uv and len(idx_uv) != 3: # Mismatch
					print(f"Warning: UV indices count ({len(idx_uv)}) mismatch for triangle face {idx_vert} in {file}. Expected 3.")
			elif len(idx_vert) == 4: # It's a quad, triangulate it
				F_list.append([idx_vert[0], idx_vert[1], idx_vert[2]]) # Triangle 1: (v0, v1, v2)
				F_list.append([idx_vert[0], idx_vert[2], idx_vert[3]]) # Triangle 2: (v0, v2, v3)
				if has_uv and len(idx_uv) == 4:
					Ft_list.append([idx_uv[0], idx_uv[1], idx_uv[2]])
					Ft_list.append([idx_uv[0], idx_uv[2], idx_uv[3]])
				elif has_uv and len(idx_uv) != 4:
					print(f"Warning: UV indices count ({len(idx_uv)}) mismatch for quad face {idx_vert} in {file}. Expected 4.")
			# else: # Polygon with more than 4 vertices - not handled by this simple triangulation
				# print(f"Warning: Face with {len(idx_vert)} vertices found in {file}. Only handling triangles and quads. Skipping this face.")
	
	V_np = np.array(V_list, np.float32) if V_list else np.empty((0,3), dtype=np.float32)
	Vt_np = np.array(Vt_list, np.float32) if Vt_list else np.empty((0,2), dtype=np.float32) 
	
    # F_list and Ft_list are now lists of (triangulated) faces.
	# The calling function (preprocess_cloth3d.py) will convert F_list to a NumPy array.
	if not Ft_list and has_uv: # If we detected UVs but couldn't build Ft_list consistently
		print(f"Warning: UVs were detected in {file} but Ft_list is empty or inconsistent. Setting Vt, Ft to None/empty.")
		Vt_np, Ft_list = None, []

	if not Vt_list: # If no UV vertices were found at all
	    Vt_np = None 
	    Ft_list = []


	return V_np, F_list, Vt_np, Ft_list


# ... (Keep all other functions in IO.py as they were: writeOBJ, readPC2, readPC2Frame, writePC2, readFaceBIN, writeFaceBIN) ...
"""
Writes OBJ files
Only handles vertices, faces and UV maps
Inputs:
- file: path to .obj file (overwrites if exists)
- V: 3D vertices
- F: 3D faces
- Vt: UV vertices
- Ft: UV faces
Correspondence between mesh and UV map is implicit in F to Ft correspondences
If no UV map data as input, it will write only 3D data in .obj file
"""
def writeOBJ(file, V, F, Vt=None, Ft=None):
	if not Vt is None:
		assert len(F) == len(Ft), 'Inconsistent data, mesh and UV map do not have the same number of faces'
		
	with open(file, 'w') as file:
		# Vertices
		for v in V:
			line = 'v ' + ' '.join([str(_) for _ in v]) + '\n'
			file.write(line)
		# UV verts
		if not Vt is None:
			for v in Vt:
				line = 'vt ' + ' '.join([str(_) for _ in v]) + '\n'
				file.write(line)
		# 3D Faces / UV faces
		if Ft:
			F = [[str(i+1)+'/'+str(j+1) for i,j in zip(f,ft)] for f,ft in zip(F,Ft)]
		else:
			F = [[str(i + 1) for i in f] for f in F]		
		for f in F:
			line = 'f ' + ' '.join(f) + '\n'
			file.write(line)

"""
Reads PC2 files, and proposed format PC16 files
Inputs:
- file: path to .pc2/.pc16 file
- float16: False for PC2 files, True for PC16
Output:
- data: dictionary with .pc2/.pc16 file data
NOTE: 16-bit floats lose precision with high values (positive or negative),
	  we do not recommend using this format for data outside range [-2, 2]
"""
def readPC2(file, float16=False):
	assert file.endswith('.pc2') and not float16 or file.endswith('.pc16') and float16, 'File format not consistent with specified input format'
	data = {}
	bytes = 2 if float16 else 4
	dtype = np.float16 if float16 else np.float32
	with open(file, 'rb') as f:
		# Header
		data['sign'] = f.read(12)
		# data['version'] = int.from_bytes(f.read(4), 'little')
		data['version'] = unpack('<i', f.read(4))[0]
		# Num points
		# data['nPoints'] = int.from_bytes(f.read(4), 'little')
		data['nPoints'] = unpack('<i', f.read(4))[0]
		# Start frame
		data['startFrame'] = unpack('f', f.read(4))
		# Sample rate
		data['sampleRate'] = unpack('f', f.read(4))
		# Number of samples
		# data['nSamples'] = int.from_bytes(f.read(4), 'little')
		data['nSamples'] = unpack('<i', f.read(4))[0]
		# Animation data
		size = data['nPoints']*data['nSamples']*3*bytes
		data['V'] = np.frombuffer(f.read(size), dtype=dtype).astype(np.float32)
		data['V'] = data['V'].reshape(data['nSamples'], data['nPoints'], 3)
		
	return data
	
"""
Reads an specific frame of PC2/PC16 files
Inputs:
- file: path to .pc2/.pc16 file
- frame: number of the frame to read
- float16: False for PC2 files, True for PC16
Output:
- T: mesh vertex data at specified frame
"""
def readPC2Frame(file, frame, float16=False):
	assert file.endswith('.pc2') and not float16 or file.endswith('.pc16') and float16, 'File format not consistent with specified input format'
	assert frame >= 0 and isinstance(frame,int), 'Frame must be a positive integer'
	bytes = 2 if float16 else 4
	dtype = np.float16 if float16 else np.float32
	with open(file,'rb') as f:
		# Num points
		f.seek(16)
		# nPoints = int.from_bytes(f.read(4), 'little')
		nPoints = unpack('<i', f.read(4))[0]
		# Number of samples
		f.seek(28)
		# nSamples = int.from_bytes(f.read(4), 'little')
		nSamples = unpack('<i', f.read(4))[0]
		if frame > nSamples:
			print("Frame index outside size")
			print("\tN. frame: " + str(frame))
			print("\tN. samples: " + str(nSamples))
			return
		# Read frame
		size = nPoints * 3 * bytes
		f.seek(size * frame, 1) # offset from current '1'
		T = np.frombuffer(f.read(size), dtype=dtype).astype(np.float32)
	return T.reshape(nPoints, 3)

"""
Writes PC2 and PC16 files
Inputs:
- file: path to file (overwrites if exists)
- V: 3D animation data as a three dimensional array (N. Frames x N. Vertices x 3)
- float16: False for writing as PC2 file, True for PC16
This function assumes 'startFrame' to be 0 and 'sampleRate' to be 1
NOTE: 16-bit floats lose precision with high values (positive or negative),
	  we do not recommend using this format for data outside range [-2, 2]
"""
def writePC2(file, V, float16=False):
	assert file.endswith('.pc2') and not float16 or file.endswith('.pc16') and float16, 'File format not consistent with specified input format'
	if float16: V = V.astype(np.float16)
	else: V = V.astype(np.float32)
	with open(file, 'wb') as f:
		# Create the header
		headerFormat='<12siiffi'
		headerStr = pack(headerFormat, b'POINTCACHE2\0',
						1, V.shape[1], 0, 1, V.shape[0])
		f.write(headerStr)
		# Write vertices
		f.write(V.tobytes())

"""
Reads proposed compressed file format for mesh topology.
Inputs:
- fname: name of the file to read
Outputs:
- F: faces of the mesh, as triangles
"""
def readFaceBIN(fname):
	if '.' in os.path.basename(fname) and not fname.endswith('.bin'): 
		print("File name extension should be '.bin'")
		return
	elif not '.' in os.path.basename(fname): fname += '.bin'
	with open(fname, 'rb') as f:
		F = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int32)
		return F.reshape((-1,3))
			
"""
Compress mesh topology into uint16 (Note that this imposes a maximum of 65,536 vertices).
Writes this data into the specified file.
Inputs:
- fname: name of the file to be created (provide NO extension)
- F: faces. MUST be an Nx3 array
"""
def writeFaceBIN(fname, F):
	assert type(F) is np.ndarray, "Make sure faces is an Nx3 NumPy array"
	assert len(F.shape) == 2 and F.shape[1] == 3, "Faces have the wron shape (should be Nx3)"
	if '.' in os.path.basename(fname) and not fname.endswith('.bin'): 
		print("File name extension should be '.bin'")
		return
	elif not '.' in os.path.basename(fname): fname += '.bin'
	F = F.astype(np.uint16)
	with open(fname, 'wb') as f:
		f.write(F.tobytes())