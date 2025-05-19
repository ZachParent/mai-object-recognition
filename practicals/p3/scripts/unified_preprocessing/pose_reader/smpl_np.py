import sys
import numpy as np
import pickle

class SMPLModel():
  def __init__(self, model_path):
    """
    SMPL model.

    Parameter:
    ---------
    model_path: Path to the SMPL model parameters, pre-processed by
    `preprocess.py`.

    """
    with open(model_path, 'rb') as f:
      if sys.version_info[0] == 2: 
	      params = pickle.load(f) # Python 2.x
      elif sys.version_info[0] == 3: 
	      params = pickle.load(f, encoding='latin1') # Python 3.x
      self.J_regressor = params['J_regressor']
      self.weights = params['weights']
      self.posedirs = params['posedirs']
      self.v_template = params['v_template']
      self.shapedirs = params['shapedirs']
      self.faces = params['f']
      self.kintree_table = params['kintree_table']

    id_to_col = {
      self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
    }
    self.parent = {
      i: id_to_col[self.kintree_table[0, i]]
      for i in range(1, self.kintree_table.shape[1])
    }

    self.pose_shape = [24, 3]
    self.beta_shape = [10]
    self.trans_shape = [3]

    self.pose = np.zeros(self.pose_shape)
    self.beta = np.zeros(self.beta_shape)
    self.trans = np.zeros(self.trans_shape)

    self.verts = None
    self.J = None
    self.R = None

    # Add a debug flag
    self.debug_smpl = False # Set to True to enable debug prints, False to disable

    self.update()

  def set_params(self, pose=None, beta=None, trans=None):
    """
    Set pose, shape, and/or translation parameters of SMPL model. Verices of the
    model will be updated and returned.

    Parameters:
    ---------
    pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
    relative to parent joint. For root joint it's global orientation.
    Represented in a axis-angle format.

    beta: Parameter for model shape. A vector of shape [10]. Coefficients for
    PCA component. Only 10 components were released by MPI.

    trans: Global translation of shape [3].

    Return:
    ------
    Updated vertices.

    """
    if self.debug_smpl: print(f"[DEBUG SMPL.set_params] Called.")
    if pose is not None:
      if self.debug_smpl: print(f"[DEBUG SMPL.set_params] Input pose (first 6 of flattened): {pose.ravel()[:6]}")
      self.pose = pose
    if beta is not None:
      if self.debug_smpl: print(f"[DEBUG SMPL.set_params] Input beta (first 3): {beta[:3]}")
      self.beta = beta
    if trans is not None:
      if self.debug_smpl: print(f"[DEBUG SMPL.set_params] Input trans: {trans}")
      self.trans = trans
    self.update()
    return self.verts, self.J

  def update(self):
    """
    Called automatically when parameters are updated.

    """
    if self.debug_smpl: print(f"[DEBUG SMPL.update] Starting update.")
    if self.debug_smpl: print(f"[DEBUG SMPL.update] self.pose used (first 6 of flattened): {self.pose.ravel()[:6]}")
    if self.debug_smpl: print(f"[DEBUG SMPL.update] self.beta used (first 3): {self.beta[:3]}")
    if self.debug_smpl: print(f"[DEBUG SMPL.update] self.trans used: {self.trans}")
    
    # how beta affect body shape
    v_shaped = self.shapedirs.dot(self.beta) + self.v_template
    if self.debug_smpl: print(f"[DEBUG SMPL.update] v_shaped (mean of coords): {np.mean(v_shaped, axis=0)}")

    # joints location
    self.J = self.J_regressor.dot(v_shaped) # This J is based on v_shaped (T-pose shape)
    if self.debug_smpl: print(f"[DEBUG SMPL.update] self.J (from v_shaped) (root, joint 15): {self.J[0]}, {self.J[15]}")
    
    pose_cube = self.pose.reshape((-1, 1, 3))
    # rotation matrix for each joint
    self.R = self.rodrigues(pose_cube) # R is key!
    if self.debug_smpl:
        is_R_identity = True
        identity_count = 0
        for i in range(self.R.shape[0]):
            if np.allclose(self.R[i], np.eye(3)):
                identity_count +=1
            else:
                is_R_identity = False # Only false if at least one is NOT identity
        print(f"[DEBUG SMPL.update] self.R (rotation matrices) - Count of identity matrices: {identity_count}/{self.R.shape[0]}")
        if not is_R_identity and self.R.shape[0] > 0: # Print first non-identity if one exists
             print(f"[DEBUG SMPL.update] self.R[0] (root rotation):\n{self.R[0]}")
             if self.R.shape[0] > 1: print(f"[DEBUG SMPL.update] self.R[1] (e.g., L_Hip relative rot):\n{self.R[1]}")

    I_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      (self.R.shape[0]-1, 3, 3) # R[1:] means R.shape[0]-1
    )
    lrotmin = (self.R[1:] - I_cube).ravel() # This is based on R. If R is identity, lrotmin is zero.
    if self.debug_smpl: print(f"[DEBUG SMPL.update] lrotmin (sum of abs, first 6): {np.sum(np.abs(lrotmin))}, {lrotmin[:6]}")

    # how pose affect body shape in zero pose
    # If lrotmin is zero, v_posed will be same as v_shaped
    posedirs_dot_lrotmin = self.posedirs.dot(lrotmin)
    v_posed = v_shaped + posedirs_dot_lrotmin
    if self.debug_smpl:
        print(f"[DEBUG SMPL.update] Pose effect on v_template (posedirs.dot(lrotmin)) (mean abs): {np.mean(np.abs(posedirs_dot_lrotmin))}")
        print(f"[DEBUG SMPL.update] v_posed (mean of coords): {np.mean(v_posed, axis=0)}")


    # world transformation of each joint
    G = np.empty((self.kintree_table.shape[1], 4, 4))
    # G[0] is the global transformation of the root joint
    G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
    for i in range(1, self.kintree_table.shape[1]):
      # G[i] is G_local_i * G_parent_i
      G[i] = G[self.parent[i]].dot(
        self.with_zeros(
          np.hstack( # This is the local transformation matrix for joint i
            [self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))] # R_local_i, T_local_i
          )
        )
      )
    
    # "undo" the T-pose joint locations from G, effectively making G a transform from T-pose space to posed space
    # G_k = G_k_world * (T_k_restpose)^-1
    # where T_k_restpose is the transform that places the origin at joint k in rest pose.
    # Homogeneous joint coords in rest pose: J_hom = [J_rest, 1]
    # G = G - pack(G . J_hom) effectively subtracts the world-transformed rest pose joint locations.
    # This makes G suitable for transforming T-pose vertices.
    G = G - self.pack(
      np.matmul(
        G, # Current G matrices (world transforms of joints)
        np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1]) # T-pose joint locations (self.J is from v_shaped)
        )
      )
    if self.debug_smpl: print(f"[DEBUG SMPL.update] G[0] (root transform after adjustment):\n{G[0]}")

    # transformation of each vertex using skinning weights
    T = np.tensordot(self.weights, G, axes=[[1], [0]]) # Skinning transforms (one 4x4 per vertex)
    rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1]))) # v_posed in homogeneous coords
    
    # Apply skinning transforms
    v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
    self.verts = v + self.trans.reshape([1, 3]) # Add global translation

    # For the J returned by set_params, it should be the posed joint locations.
    # These can be re-calculated from the final posed vertices using the J_regressor for accuracy.
    # Subtract self.trans because J_regressor expects vertices relative to the model's origin before global translation.
    posed_verts_relative_to_origin = self.verts - self.trans.reshape([1,3])
    self.J = self.J_regressor.dot(posed_verts_relative_to_origin)
    self.J += self.trans.reshape([1,3]) # Add back global translation to get world-space posed joints.
    
    if self.debug_smpl: print(f"[DEBUG SMPL.update] Final self.J (output from update) (root, joint 15): {self.J[0]}, {self.J[15]}")
    if self.debug_smpl: print(f"[DEBUG SMPL.update] Finished update.")


  def rodrigues(self, r):
    """
    Rodrigues' rotation formula that turns axis-angle vector into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation vector of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].

    """
    if self.debug_smpl and r.shape[0] > 0: print(f"[DEBUG SMPL.rodrigues] input r (shape, r[0]): {r.shape}, {r[0]}")
    
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    
    # Threshold for considering theta as zero to avoid division by zero and numerical instability
    # For theta very close to zero, the rotation matrix should be identity.
    zero_theta_threshold = 1e-9 # A small epsilon
    is_zero_theta = theta < zero_theta_threshold
    
    # Avoid division by zero for r_hat calculation.
    # For zero_theta cases, r_hat can be anything as it will be multiplied by (1-cos(theta)) which is ~0,
    # and sin(theta) which is ~0. Setting it to zero vector is safe.
    # For non-zero_theta, proceed as usual.
    theta_safe = np.maximum(theta, zero_theta_threshold) # Use this for division
    r_hat = r / theta_safe
    
    # For thetas that were actually zero, ensure r_hat components are zero to prevent NaNs if r was also zero.
    # This step might be redundant if r/theta_safe already handles it, but it's safer.
    r_hat[is_zero_theta.squeeze(), :, :] = 0.0

    if self.debug_smpl and theta.shape[0] > 0:
        print(f"[DEBUG SMPL.rodrigues] theta (min, max, mean): {theta.min()}, {theta.max()}, {theta.mean()}")
        print(f"[DEBUG SMPL.rodrigues] Count of near-zero thetas: {np.sum(is_zero_theta)}")

    cos_theta = np.cos(theta) # Use the original theta for cos and sin
    sin_theta = np.sin(theta)

    z_stick = np.zeros(theta.shape[0])
    # Skew-symmetric matrix K from r_hat
    m = np.dstack([
      z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
      r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
      -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
    ).reshape([-1, 3, 3])
    
    i_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      [theta.shape[0], 3, 3]
    )
    
    # Rodrigues' formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
    # K^2 = r_hat * r_hat^T - I (or r_hat (outer) r_hat - I_cube for batch)
    # Using the form: R = cos(theta)*I + (1-cos(theta))*(r_hat * r_hat^T) + sin(theta)*K
    # where (r_hat * r_hat^T) is the outer product.
    # A = np.transpose(r_hat, axes=[0, 2, 1]) # This was for A.B to get r_hat.r_hat (scalar) which is 1.
    # B = r_hat
    # dot = np.matmul(A, B) # This is r_hat_i * r_hat_i = 1. Not what's needed for outer product.

    # Outer product r_hat (outer) r_hat for each batch item:
    # r_hat is (batch, 1, 3). We need (batch, 3, 1) @ (batch, 1, 3) -> (batch, 3, 3)
    r_hat_col = r_hat.transpose(0,2,1) # (batch, 3, 1)
    r_hat_outer_r_hat = np.matmul(r_hat_col, r_hat) # (batch, 3, 3)

    R_calc = cos_theta * i_cube + (1 - cos_theta) * r_hat_outer_r_hat + sin_theta * m
    
    # For thetas that were effectively zero, explicitly set R to identity
    R_calc[is_zero_theta.squeeze(), :, :] = np.eye(3)
    
    if self.debug_smpl and R_calc.shape[0] > 0 : print(f"[DEBUG SMPL.rodrigues] Output R_calc (R_calc[0]):\n{R_calc[0]}")
    return R_calc

  def with_zeros(self, x):
    """
    Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

    Parameter:
    ---------
    x: Matrix to be appended.

    Return:
    ------
    Matrix after appending of shape [4,4]

    """
    return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

  def pack(self, x):
    """
    Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
    manner.

    Parameter:
    ----------
    x: Matrices to be appended of shape [batch_size, 4, 1]

    Return:
    ------
    Matrix of shape [batch_size, 4, 4] after appending.

    """
    return np.dstack((np.zeros((x.shape[0], 4, 3)), x))