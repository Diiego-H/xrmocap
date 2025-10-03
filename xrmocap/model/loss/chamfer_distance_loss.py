# yapf:disable
import torch
import torch.nn as nn

# yapf:enable


class ChamferDistanceLoss(nn.Module):
    """ChamferDistanceLoss for two mesh surfaces."""

    def __init__(self, loss_weight: float=1.0):
        super(ChamferDistanceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, loss_weight_override=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction. Shape should be (V1, 3).
                V1: number of vertices.
            target (torch.Tensor): The learning target of the prediction.
                Shape should be (V2, 3).
            loss_weight_override (float, optional): The overall weight of loss
                used to override the original weight of loss.
        Returns:
            torch.Tensor: The calculated loss
        """
        loss_weight = loss_weight_override\
            if loss_weight_override is not None \
            else self.loss_weight

        V1, V2 = len(pred), len(target)

        # Squared distances
        diffs = pred[:, None, :] - target[None, :, :]   # (V1, V2, 3)
        diffs = torch.sqrt(torch.sum(diffs**2, dim=-1)) # (V1, V2)

        # Find nearest vertex across the other mesh
        pred2target, _ = torch.min(diffs, dim=1)       # (V1,)
        target2pred, _ = torch.min(diffs, dim=0)       # (V2,)

        # Chamfer distance
        # loss = pred2target.mean() + target2pred.mean()
        # loss = 0.5 * pred2target.mean() + 0.5 * target2pred.mean()
        # TRYING COVERAGE FROM SMPL-X TO DEPTH
        loss = pred2target.mean()

        print(loss)

        loss = loss_weight * loss
        return loss
