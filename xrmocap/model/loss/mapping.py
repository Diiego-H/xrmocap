LOSS_MAPPING = {
    'keypoints3d_limb_len': ['betas'],
    'keypoints3d_mse': ['body_pose'],
    'keypoints2d_mse': ['body_pose'],
    'shape_prior': ['betas'],
    'shape_bound': ['betas'],
    'joint_prior': ['body_pose'],
    'smooth_joint': ['body_pose'],
    'pose_reg': ['body_pose'],
    'smooth_global': ['global_orient'],
    'smooth_transl': ['transl'],
    'smooth_hands': ['left_hand_pose', 'right_hand_pose'],
}
