# yapf: disable
import logging
import numpy as np
from typing import List, Tuple, Union, overload
from xrprimer.data_structure import Keypoints
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.transform.convention.keypoints_convention import (
    convert_keypoints,
)
from xrprimer.utils.ffmpeg_utils import video_to_array

from xrmocap.data_structure.body_model import SMPLData, SMPLXData
from xrmocap.human_perception.builder import (
    MMdetDetector, MMposeTopDownEstimator, build_detector,
)
from xrmocap.io.image import load_multiview_images
from xrmocap.model.registrant.builder import SMPLify, build_registrant
from xrmocap.model.registrant.handler.builder import build_handler

from .base_estimator import BaseEstimator

from mmdet.utils import register_all_modules as register_det_modules
from mmpose.utils import register_all_modules as register_pose_modules

# yapf: enable


class MultiViewSinglePersonSMPLEstimator(BaseEstimator):
    """Api for estimating smpl in a multi-view single-person scene."""

    def __init__(self,
                 work_dir: str,
                 bbox_detector: Union[dict, MMdetDetector],
                 kps2d_estimator: Union[dict, MMposeTopDownEstimator],
                 smplify: Union[dict, SMPLify],
                 load_batch_size: int = 500,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Initialization of the class.

        Args:
            work_dir (str):
                Path to the folder for running the api. No file in work_dir
                will be modified
                or added by MultiViewSinglePersonSMPLEstimator.
            bbox_detector (Union[dict, MMdetDetector]):
                A human bbox_detector or its config.
            kps2d_estimator (Union[dict, MMposeTopDownEstimator]):
                A top-down kps2d estimator or its config.
            smplify (Union[dict, SMPLify]):
                A SMPLify instance or its config.
            load_batch_size (int, optional):
                How many frames are loaded at the same time. Defaults to 500.
            verbose (bool, optional):
                Whether to print(logger.info) information during estimating.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(work_dir, verbose, logger)
        self.load_batch_size = load_batch_size

        register_det_modules()
        if isinstance(bbox_detector, dict):
            bbox_detector['logger'] = logger
            self.bbox_detector = build_detector(bbox_detector)
        else:
            self.bbox_detector = bbox_detector

        register_pose_modules()
        if isinstance(kps2d_estimator, dict):
            kps2d_estimator['logger'] = logger
            self.kps2d_estimator = build_detector(kps2d_estimator)
        else:
            self.kps2d_estimator = kps2d_estimator

        if isinstance(smplify, dict):
            smplify['logger'] = logger
            if smplify['type'].lower() == 'smplify':
                self.smpl_data_type = 'smpl'
            elif smplify['type'].lower() == 'smplifyx':
                self.smpl_data_type = 'smplx'
            else:
                self.logger.error('SMPL data type not found.')
                raise TypeError

            self.smplify = build_registrant(smplify)
        else:
            self.smplify = smplify

    def run(
        self,
        cam_param: List[FisheyeCameraParameter],
        img_arr: Union[List, np.ndarray],
        init_smpl_data: Union[None, SMPLData] = None,
    ) -> Tuple[List[Keypoints], Keypoints, SMPLData]:
        """Run multi-view single-person SMPL estimator.

        Args:
            cam_param (List[FisheyeCameraParameter]):
                A list of FisheyeCameraParameter instances.
            img_arr (Union[List, np.ndarray]):
                A list (or array) of multi-view images, in shape [n_view, n_frame, h, w, c].

        Returns:
            Tuple[List[Keypoints], Keypoints, SMPLData]:
                A list of kps2d, an instance of Keypoints 3d,
                an instance of SMPLData.
        """
        keypoints2d_list = self.estimate_keypoints2d(img_arr=img_arr)
        keypoints3d = self.estimate_keypoints3d(
            cam_param=cam_param, keypoints2d_list=keypoints2d_list)
        smpl_data = self.estimate_smpl(
            keypoints3d=keypoints3d, init_smpl_data=init_smpl_data)
        return keypoints2d_list, keypoints3d, smpl_data

    def estimate_keypoints2d(self, img_arr: Union[List, np.ndarray]) -> List[Keypoints]:
        """Estimate keypoints2d in a top-down way.

        Args:
            img_arr (Union[List, np.ndarray]):
                A list (or array) of multi-view images, in shape [n_view, n_frame, h, w, c].

        Returns:
            List[Keypoints]:
                A list of keypoints2d instances.
        """
        self.logger.info('Estimating keypoints2d.')
        ret_list = []
        for view_index in range(len(img_arr)):
            view_img_arr = img_arr[view_index]
            register_det_modules()
            bbox_list = self.bbox_detector.infer_array(
                image_array=view_img_arr,
                disable_tqdm=(not self.verbose),
                multi_person=False)
            register_pose_modules()
            kps2d_list = self.kps2d_estimator.infer_array(
                image_array=view_img_arr,
                bbox_list=bbox_list,
                disable_tqdm=(not self.verbose),
            )
            if len(kps2d_list) == 1 and \
                    len(kps2d_list[0]) == 1 and \
                    kps2d_list[0][0] is None:
                kps2d_list = [[]]
            keypoints2d = self.kps2d_estimator.get_keypoints_from_result(
                kps2d_list)
            ret_list.append(keypoints2d)
        return ret_list

    def estimate_keypoints3d(self, cam_param: List[FisheyeCameraParameter],
                             keypoints2d_list: List[Keypoints]) -> Keypoints:
        """Estimate keypoints3d by triangulation and optimizers if exists.

        Args:
            cam_param (List[FisheyeCameraParameter]):
                A list of FisheyeCameraParameter instances.
            keypoints2d_list (List[Keypoints]):
                A list of Keypoints2d, in same mask and convention,
                and the time axis are aligned.

        Returns:
            Keypoints: A keypoints3d Keypoints instance.
        """
        # TODO: BUNDLE ADJUSTMENT FROM MEYSAM
        self.logger.info('Estimating keypoints3d.')
        # prepare input np.ndarray
        kps_arr_list = []
        mask_list = []
        default_keypoints2d = None
        for keypoints2d in keypoints2d_list:
            if keypoints2d is not None:
                default_keypoints2d = keypoints2d.clone()
                default_keypoints2d.set_keypoints(
                    np.zeros_like(default_keypoints2d.get_keypoints()))
                default_keypoints2d.set_mask(
                    np.zeros_like(default_keypoints2d.get_mask()))
                break
        if default_keypoints2d is None:
            self.logger.error('No one has been detected in any view.')
            raise AttributeError
        for keypoints2d in keypoints2d_list:
            if keypoints2d is None:
                keypoints2d = default_keypoints2d
            if keypoints2d.dtype != 'numpy':
                keypoints2d = keypoints2d.to_numpy()
            kps_arr_list.append(keypoints2d.get_keypoints()[:, 0, ...])
            mask_list.append(keypoints2d.get_mask()[:, 0, ...])
        mview_kps2d_arr = np.asarray(kps_arr_list)
        mview_mask = np.asarray(mask_list)
        mview_mask = np.expand_dims(mview_mask, -1)
        # select camera
        cam_indexes = self.select_camera(cam_param, mview_kps2d_arr, mview_mask)
        self.triangulator.set_cameras(cam_param)
        selected_triangulator = self.triangulator[cam_indexes]
        mview_kps2d_arr = mview_kps2d_arr[np.asarray(cam_indexes), ...]
        triangulate_mask = mview_mask[np.asarray(cam_indexes), ...]
        # cascade point selectors
        self.logger.info('Selecting points.')
        if self.final_selectors is not None:
            for selector in self.final_selectors:
                triangulate_mask = selector.get_selection_mask(
                    points=mview_kps2d_arr, init_points_mask=triangulate_mask)
        kps3d_arr = selected_triangulator.triangulate(
            points=mview_kps2d_arr, points_mask=triangulate_mask)
        kps3d_arr = np.concatenate(
            (kps3d_arr, np.ones_like(kps3d_arr[..., 0:1])), axis=-1)
        kps3d_arr = np.expand_dims(kps3d_arr, axis=1)
        kps3d_mask = np.sum(mview_mask, axis=(0, 1), keepdims=False)
        kps3d_mask = np.sign(np.abs(kps3d_mask))
        if kps3d_mask.shape[-1] == 1:
            kps3d_mask = kps3d_mask[..., 0]
        keypoints3d = Keypoints(
            dtype='numpy',
            kps=kps3d_arr,
            mask=kps3d_mask,
            convention=default_keypoints2d.get_convention())
        optim_kwargs = dict(
            mview_kps2d=np.expand_dims(mview_kps2d_arr, axis=2),
            mview_kps2d_mask=np.expand_dims(triangulate_mask, axis=2))
        if self.kps3d_optimizers is not None:
            for optimizer in self.kps3d_optimizers:
                if hasattr(optimizer, 'triangulator'):
                    optimizer.triangulator = selected_triangulator
                keypoints3d = optimizer.optimize_keypoints3d(
                    keypoints3d, **optim_kwargs)
        return keypoints3d

    def estimate_smpl(self,
                      keypoints3d: Keypoints,
                      init_smpl_data: Union[None, SMPLData] = None,
                      return_joints: bool = False,
                      return_verts: bool = False) -> SMPLData:
        """Estimate smpl parameters according to keypoints3d.

        Args:
            keypoints3d (Keypoints):
                A keypoints3d Keypoints instance, with only one person
                inside. This method will take the person at
                keypoints3d.get_keypoints()[:, 0, ...] to run smplify.
            init_smpl_dict (dict, optional):
                A dict of init parameters. init_dict.keys() is a
                sub-set of self.__class__.OPTIM_PARAM.
                Defaults to an empty dict.
            return_joints (bool, optional):
                Whether to return joints. Defaults to False.
            return_verts (bool, optional):
                Whether to return vertices. Defaults to False.

        Returns:
            SMPLData:
                Smpl data of the person.
        """
        self.logger.info('Estimating SMPL.')
        working_convention = self.smplify.body_model.keypoint_convention
        keypoints3d = convert_keypoints(
            keypoints=keypoints3d, dst=working_convention)
        keypoints3d = keypoints3d.to_tensor(device=self.smplify.device)
        kps3d_tensor = keypoints3d.get_keypoints()[:, 0, :, :3].float()
        kps3d_conf = keypoints3d.get_mask()[:, 0, ...]

        # load init smpl data
        if init_smpl_data is not None:
            init_smpl_dict = init_smpl_data.to_tensor_dict(
                device=self.smplify.device)
        else:
            init_smpl_dict = {}

        # build and run
        kp3d_mse_input = build_handler(
            dict(
                type='Keypoint3dMSEInput',
                keypoints3d=kps3d_tensor,
                keypoints3d_conf=kps3d_conf,
                keypoints3d_convention=working_convention,
                handler_key='keypoints3d_mse'))
        kp3d_llen_input = build_handler(
            dict(
                type='Keypoint3dLimbLenInput',
                keypoints3d=kps3d_tensor,
                keypoints3d_conf=kps3d_conf,
                keypoints3d_convention=working_convention,
                handler_key='keypoints3d_limb_len'))

        registrant_output = self.smplify(
            input_list=[kp3d_mse_input, kp3d_llen_input],
            init_param_dict=init_smpl_dict,
            return_joints=return_joints,
            return_verts=return_verts)

        if self.smpl_data_type == 'smplx':
            smpl_data = SMPLXData()
        elif self.smpl_data_type == 'smpl':
            smpl_data = SMPLData()

        smpl_data.from_param_dict(registrant_output)

        if return_joints:
            smpl_data['joints'] = registrant_output['joints']
        if return_verts:
            smpl_data['vertices'] = registrant_output['vertices']

        return smpl_data
