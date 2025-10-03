import cv2
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Union
from xrprimer.data_structure import Keypoints
from xrprimer.transform.convention.keypoints_convention import get_keypoint_num
from xrprimer.utils.ffmpeg_utils import video_to_array
from xrprimer.utils.log_utils import get_logger

from mmpose.apis import inference_topdown, init_model


class MMposeTopDownEstimator:

    def __init__(self,
                 mmpose_kwargs: dict,
                 batch_size: int = 1,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Init a detector from mmpose.

        Args:
            mmpose_kwargs (dict):
                A dict contains args of mmpose.apis.init_detector.
                Necessary keys: config, checkpoint
                Optional keys: device
            bbox_thr (float, optional):
                Threshold of a bbox. Those have lower scores will be ignored.
                Defaults to 0.0.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        # build the pose model from a config file and a checkpoint file
        self.pose_model = init_model(**mmpose_kwargs)
        # mmpose inference api takes one image per callj
        self.batch_size = batch_size
        self.logger = get_logger(logger)

    def get_keypoints_convention_name(self) -> str:
        """Get data_source from dataset type in config file of the pose model.

        Returns:
            str:
                Name of the keypoints convention. Must be
                a key of KEYPOINTS_FACTORY.
        """
        return __translate_data_source__(
            self.pose_model.cfg.test_dataloader.dataset.type)

    def infer_array(self,
                    image_array: Union[np.ndarray, list],
                    bbox_list: Union[tuple, list],
                    disable_tqdm: bool = False) -> Tuple[list, list]:
        """Infer frames already in memory(ndarray type).

        Args:
            image_array (Union[np.ndarray, list]):
                BGR image ndarray in shape [n_frame, height, width, 3],
                or a list of image ndarrays in shape [height, width, 3] while
                len(list) == n_frame.
            bbox_list (Union[tuple, list]):
                A list of human bboxes.
                Shape of the nested lists is (n_frame, n_human, 4).
                Each bbox is a bbox_xyxy.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.

        Returns:
            Tuple[list, list]:
                keypoints_list (list):
                    A list of human keypoints.
                    Shape of the nested lists is
                    (n_frame, n_human, n_keypoints, 3).
                    Each keypoint is an array of (x, y, confidence).
        """
        ret_kps_list = []
        n_frame = len(image_array)
        for start_index in tqdm(range(0, n_frame, self.batch_size), disable=disable_tqdm, leave=False):
            end_index = min(n_frame, start_index + self.batch_size)

            # Extended to take multiple frames (account for no person detected)
            img_batch = image_array[start_index:end_index]
            bboxes_batch = []
            for i, bboxes in enumerate(bbox_list[start_index:end_index]):
                if len(bboxes) != 1:
                    print(f"{len(bboxes)} people detected in frame {i+start_index}!")
                    if len(bboxes) == 0:
                        h,w = img_batch[0].shape[:2]
                        bboxes_batch.append(np.array([0,0,w,h], dtype=np.float32))
                    else:
                        raise ValueError(f"This is a single person setting, but {len(bboxes)} people were detected!")
                
                else:
                    bboxes_batch.append(bboxes[0])

            pose_results = inference_topdown(
                model=self.pose_model,
                imgs=img_batch,
                bboxes=bboxes_batch,
                bbox_format='xyxy')

            # Concatenate results (add n_human axis)
            for frame_result in pose_results:
                pred = frame_result.pred_instances

                # NOTE: Visibility is interpreted as confidence
                # ret_kps_list.append(np.expand_dims(np.hstack((np.squeeze(pred.keypoints), pred.keypoints_visible.T)), axis=0))
                # NOTE: Scores are interpreted from raw logits
                ret_kps_list.append(np.expand_dims(np.hstack((np.squeeze(pred.keypoints), pred.keypoint_scores.T)), axis=0))

        return ret_kps_list

    def infer_frames(
            self,
            frame_path_list: list,
            bbox_list: Union[tuple, list],
            disable_tqdm: bool = False,
            return_heatmap: bool = False,
            load_batch_size: Union[None, int] = None) -> Tuple[list, list]:
        """Infer frames from file.

        Args:
            frame_path_list (list):
                A list of frames' absolute paths.
            bbox_list (Union[tuple, list]):
                A list of human bboxes.
                Shape of the nested lists is (n_frame, n_human, 5).
                Each bbox is a bbox_xyxy with a bbox_score at last.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.
            return_heatmap (bool, optional):
                Whether to return heatmap.
                Defaults to False.
            load_batch_size (Union[None, int], optional):
                How many frames are loaded at the same time.
                Defaults to None, load all frames in frame_path_list.

        Returns:
            Tuple[list, list]:
                keypoints_list (list):
                    A list of human keypoints.
                    Shape of the nested lists is
                    (n_frame, n_human, n_keypoints, 3).
                    Each keypoint is an array of (x, y, confidence).
                heatmap_list (list):
                    A list of keypoint heatmaps. len(heatmap_list) == n_frame
                    and the shape of heatmap_list[f] is
                    (n_human, n_keypoints, width, height).
        """
        ret_kps_list = []
        ret_heatmap_list = []
        ret_boox_list = []
        if load_batch_size is None:
            load_batch_size = len(frame_path_list)
        for start_idx in range(0, len(frame_path_list), load_batch_size):
            end_idx = min(len(frame_path_list), start_idx + load_batch_size)
            if load_batch_size < len(frame_path_list):
                self.logger.info(
                    'Processing mmpose on frames' +
                    f'({start_idx}-{end_idx})/{len(frame_path_list)}')
            image_array_list = []
            for frame_abs_path in frame_path_list[start_idx:end_idx]:
                img_np = cv2.imread(frame_abs_path)
                image_array_list.append(img_np)
            batch_pose_list, batch_heatmap_list, batch_boox_list = \
                self.infer_array(
                    image_array=image_array_list,
                    bbox_list=bbox_list[start_idx:end_idx],
                    disable_tqdm=disable_tqdm,
                    return_heatmap=return_heatmap)
            ret_kps_list += batch_pose_list
            ret_heatmap_list += batch_heatmap_list
            ret_boox_list += batch_boox_list
        return ret_kps_list, ret_heatmap_list, ret_boox_list

    def infer_video(self,
                    video_path: str,
                    bbox_list: Union[tuple, list],
                    disable_tqdm: bool = False,
                    return_heatmap: bool = False) -> Tuple[list, list]:
        """Infer frames from a video file.

        Args:
            video_path (str):
                Path to the video to be detected.
            bbox_list (Union[tuple, list]):
                A list of human bboxes.
                Shape of the nested lists is (n_frame, n_human, 5).
                Each bbox is a bbox_xyxy with a bbox_score at last.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.
            return_heatmap (bool, optional):
                Whether to return heatmap.
                Defaults to False.

        Returns:
            Tuple[list, list]:
                keypoints_list (list):
                    A list of human keypoints.
                    Shape of the nested lists is
                    (n_frame, n_human, n_keypoints, 3).
                    Each keypoint is an array of (x, y, confidence).
                heatmap_list (list):
                    A list of keypoint heatmaps. len(heatmap_list) == n_frame
                    and the shape of heatmap_list[f] is
                    (n_human, n_keypoints, width, height).
        """
        image_array = video_to_array(input_path=video_path, logger=self.logger)
        ret_kps_list, ret_heatmap_list, ret_boox_list = self.infer_array(
            image_array=image_array,
            bbox_list=bbox_list,
            disable_tqdm=disable_tqdm,
            return_heatmap=return_heatmap)
        return ret_kps_list, ret_heatmap_list, ret_boox_list

    def get_keypoints_from_result(
            self, kps2d_list: List[list]) -> Union[Keypoints, None]:
        """Convert returned keypoints2d into an instance of class Keypoints.

        Args:
            kps2d_list (List[list]):
                A list of human keypoints, returned by
                infer methods.
                Shape of the nested lists is
                (n_frame, n_human, n_keypoints, 3).

        Returns:
            Union[Keypoints, None]:
                An instance of Keypoints with mask and
                convention, data type is numpy.
                If no one has been detected in any frame,
                a None will be returned.
        """
        # shape: (n_frame, n_human, n_keypoints, 3)
        n_frame = len(kps2d_list)
        human_count_list = [len(human_list) for human_list in kps2d_list]
        if len(human_count_list) > 0:
            n_human = max(human_count_list)
        else:
            n_human = 0
        n_keypoints = get_keypoint_num(self.get_keypoints_convention_name())
        if n_human > 0:
            kps2d_arr = np.zeros(shape=(n_frame, n_human, n_keypoints, 3))
            mask_arr = np.ones_like(kps2d_arr[..., 0], dtype=np.uint8)
            for f_idx in range(n_frame):
                if len(kps2d_list[f_idx]) <= 0:
                    mask_arr[f_idx, ...] = 0
                    continue
                for h_idx in range(n_human):
                    if h_idx < len(kps2d_list[f_idx]):
                        mask_arr[f_idx, h_idx, ...] = np.sign(kps2d_list[f_idx][h_idx, :, -1])
                        kps2d_arr[f_idx, h_idx, :, :] = kps2d_list[f_idx][h_idx]
                    else:
                        mask_arr[f_idx, h_idx, ...] = 0
            keypoints2d = Keypoints(
                kps=kps2d_arr,
                mask=mask_arr,
                convention=self.get_keypoints_convention_name(),
                logger=self.logger)
        else:
            keypoints2d = None
        return keypoints2d


def __translate_data_source__(mmpose_dataset_name):
    if mmpose_dataset_name == 'TopDownSenseWholeBodyDataset':
        return 'sense_whole_body'
    elif mmpose_dataset_name == 'TopDownCocoWholeBodyDataset' or mmpose_dataset_name == 'CocoWholeBodyDataset':
        return 'coco_wholebody'
    else:
        raise NotImplementedError
