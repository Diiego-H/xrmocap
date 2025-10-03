import cv2
import logging
import numpy as np
from tqdm import tqdm
from typing import Union
from xrprimer.utils.ffmpeg_utils import video_to_array
from xrprimer.utils.log_utils import get_logger

from xrmocap.transform.bbox import qsort_bbox_list

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception


class MMdetDetector:
    """Detector wrapped from mmdetection."""

    def __init__(self,
                 mmdet_kwargs: dict,
                 batch_size: int = 1,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Init a detector from mmdetection.

        Args:
            mmdet_kwargs (dict):
                A dict contains args of mmdet.apis.init_detector.
                Necessary keys: config, checkpoint
                Optional keys: device, cfg_options
            batch_size (int, optional):
                Batch size for inference. Defaults to 1.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.logger = get_logger(logger)
        if not has_mmdet:
            self.logger.error(import_exception)
            raise ModuleNotFoundError('Please install mmdet to run detection.')
        # build the detector from a config file and a checkpoint file
        self.det_model = init_detector(**mmdet_kwargs)
        self.batch_size = batch_size

    def infer_array(self,
                    image_array: np.ndarray,
                    disable_tqdm: bool = False,
                    multi_person: bool = False) -> list:
        """Infer frames already in memory(ndarray type).

        Args:
            image_array (np.ndarray):
                BGR image ndarray in shape [n_frame, height, width, 3].
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.
            multi_person (bool, optional):
                Whether to allow multi-person detection. If False,
                only the biggest bbox will be returned.
                Defaults to False.

        Returns:
            list:
                List of bboxes. Shape of the nested lists is
                (n_frame, n_human, 5)
                and each bbox is (x, y, x, y, score).
        """
        ret_list = []
        bbox_results = []
        n_frame = len(image_array)
        for start_index in tqdm(range(0, n_frame, self.batch_size), disable=disable_tqdm, leave=False):
            end_index = min(n_frame, start_index + self.batch_size)
            img_batch = image_array[start_index:end_index]
            mmdet_results = inference_detector(self.det_model, img_batch)
            for frame_result in mmdet_results:
                prediction = frame_result.pred_instances
                bbox_results.append({
                    "bboxes": prediction.bboxes.cpu().numpy(),
                    "scores": prediction.scores.cpu().numpy(),
                    "labels": prediction.labels.cpu().numpy()
                })
        ret_list = process_mmdet_results(
            bbox_results, multi_person=multi_person)
        return ret_list

    def infer_frames(self,
                     frame_path_list: list,
                     disable_tqdm: bool = False,
                     multi_person: bool = False,
                     load_batch_size: Union[None, int] = None) -> list:
        """Infer frames from file.

        Args:
            frame_path_list (list):
                A list of frames' absolute paths.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.
            multi_person (bool, optional):
                Whether to allow multi-person detection, which is
                slower than single-person.
                Defaults to False.
            load_batch_size (Union[None, int], optional):
                How many frames are loaded at the same time.
                Defaults to None, load all frames in frame_path_list.

        Returns:
            list:
                List of bboxes. Shape of the nested lists is
                (n_frame, n_human, 5)
                and each bbox is (x, y, x, y, score).
        """
        ret_list = []
        if load_batch_size is None:
            load_batch_size = len(frame_path_list)
        for start_idx in range(0, len(frame_path_list), load_batch_size):
            end_idx = min(len(frame_path_list), start_idx + load_batch_size)
            if load_batch_size < len(frame_path_list):
                self.logger.info(
                    'Processing mmdet on frames' +
                    f'({start_idx}-{end_idx})/{len(frame_path_list)}')
            image_array_list = []
            for frame_abs_path in frame_path_list[start_idx:end_idx]:
                img_np = cv2.imread(frame_abs_path)
                image_array_list.append(img_np)
            load_batch_result = self.infer_array(
                image_array=image_array_list,
                disable_tqdm=disable_tqdm,
                multi_person=multi_person)
            ret_list += load_batch_result
        return ret_list

    def infer_video(self,
                    video_path: str,
                    disable_tqdm: bool = False,
                    multi_person: bool = False) -> list:
        """Infer frames from a video file.

        Args:
            video_path (str):
                Path to the video to be detected.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.
            multi_person (bool, optional):
                Whether to allow multi-person detection, which is
                slower than single-person.
                Defaults to False.

        Returns:
            list:
                List of bboxes. Shape of the nested lists is
                (n_frame, n_human, 5)
                and each bbox is (x, y, x, y, score).
        """
        image_array = video_to_array(input_path=video_path, logger=self.logger)
        ret_list = self.infer_array(
            image_array=image_array,
            disable_tqdm=disable_tqdm,
            multi_person=multi_person)
        return ret_list


def process_mmdet_results(mmdet_results: list,
                          cat_id: int = 0,
                          multi_person: bool = True) -> list:
    """Process mmdet results, sort bboxes by area in descending order.

    Args:
        mmdet_results (list):
            Result of mmdet.apis.inference_detector
            when the input is a batch.
            List of dictionaries with keys
            "bboxes", "scores" and "labels".
        cat_id (int, optional):
            Category ID. This function will only select
            the selected category, and drop the others.
            Defaults to 0, ID of human category.
        multi_person (bool, optional):
            Whether to allow multi-person detection, which is
            slower than single-person. If false, the function
            only assure that the first person of each frame
            has the biggest bbox.
            Defaults to True.

    Returns:
        list:
            A list of detected bounding boxes.
            Shape of the nested lists is
            (n_frame, n_human, 5)
            and each bbox is (x, y, x, y, score).
    """
    ret_list = []
    only_max_arg = not multi_person
    for _, frame_results in enumerate(mmdet_results):
        # Filter bboxes for selected category
        cat_mask = frame_results["labels"] == cat_id
        bboxes = frame_results["bboxes"][cat_mask]

        if len(bboxes) > 0 and only_max_arg:
            # Append bbox with maximum score
            scores = frame_results["scores"][cat_mask]
            ret_list.append([bboxes[np.argmax(scores)]])
        else:
            ret_list.append(bboxes)
    return ret_list
