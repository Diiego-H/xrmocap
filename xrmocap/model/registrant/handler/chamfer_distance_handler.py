import logging
import torch
from typing import TypeVar, Union
import pickle

from xrmocap.model.loss.builder import build_loss
from .base_handler import BaseHandler, BaseInput

_ChamferDistanceLoss = TypeVar('_ChamferDistanceLoss')

with open("/code/src/my_list.pkl", "rb") as f:
    smpl_idxs = pickle.load(f)

class ChamferDistanceInput(BaseInput):

    def __init__(
        self,
        vertices: torch.Tensor,
        handler_key='chamfer_distance',
    ) -> None:
        """Construct an input instance for ChamferDistanceInput.

        Args:
            vertices (torch.Tensor):
                3D vertices in shape (n_vertices, 3).
            handler_key (str, optional):
                Key of this input-handler pair. This input will
                be assigned to a handler who has the same key.
                Defaults to 'chamfer_distance'.
        """
        self.vertices = vertices
        super().__init__(handler_key=handler_key)

    def get_batch_size(self) -> int:
        """Get batch size of the input.

        Returns:
            int: batch_size
        """
        return int(self.vertices.shape[0])


class ChamferDistanceHandler(BaseHandler):

    def __init__(self,
                 loss: Union[_ChamferDistanceLoss, dict],
                 handler_key='chamfer_distance',
                 device: Union[torch.device, str] = 'cuda',
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Construct a ChamferDistanceHandler instance compute smpl(x/xd)
        parameters and BaseInput, return a loss Tensor.

        Args:
            loss (Union[ChamferDistanceLoss, dict]):
                An instance of ChamferDistanceLoss, or a config dict of
                ChamferDistanceLoss.
            handler_key (str, optional):
                Key of this input-handler pair. This input will
                be assigned to a handler who has the same key.
                Defaults to 'chamfer_distance'.
            device (Union[torch.device, str], optional):
                Device in pytorch. Defaults to 'cuda'.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.

        Raises:
            TypeError: loss is neither a torch.nn.Module nor a dict.
        """
        super().__init__(handler_key=handler_key, device=device, logger=logger)
        if isinstance(loss, dict):
            self.loss = build_loss(loss)
        elif isinstance(loss, torch.nn.Module):
            self.loss = loss
        else:
            self.logger.error('Type of loss is not correct.\n' +
                              f'Type: {type(loss)}.')
            raise TypeError
        self.loss = self.loss.to(self.device)

    def requires_verts(self) -> bool:
        """Whether this handler requires body_model vertices.

        Returns:
            bool: Whether this handler requires body_model vertices.
        """
        return True

    def get_loss_weight(self) -> float:
        """Get the weight value of this loss handler.

        Returns:
            float: Weight value.
        """
        loss_weight = self.loss.loss_weight
        return float(loss_weight)

    def __call__(self,
                 related_input: ChamferDistanceInput,
                 model_vertices: torch.Tensor,
                 loss_weight_override: float = None,
                 **kwargs: dict) -> torch.Tensor:
        """Taking Keypoint3dLimbLenInput and smpl(x/xd) parameters, compute
        loss and return a Tensor.

        Args:

            related_input (Keypoint3dLimbLenInput):
                An instance of Keypoint3dLimbLenInput, having the same
                key as self.
            model_vertices (torch.Tensor):
                Vertices from body_model.
            loss_weight_override (float, optional):
                Override the global weight of this loss.
                Defaults to None.
            kwargs (dict):
                Redundant smpl(x/d) keyword arguments to be
                ignored.

        Returns:
            torch.Tensor:
                A Tensor of loss result.
        """

        # 20th frame is used
        source_vertices = model_vertices[20, ...]
        target_vertices = related_input.vertices[20, ...]

        # CHECK: ONLY USE VERTICES FROM TORSO
        # source_vertices = source_vertices[smpl_idxs]

        chamfer_distance = self.loss(
            pred=source_vertices,
            target=target_vertices,
            loss_weight_override=loss_weight_override)
        return chamfer_distance
