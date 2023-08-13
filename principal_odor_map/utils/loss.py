import torch
from typing import Optional, Callable, List
from deepchem.models.losses import Loss


class CustomMultiLabelLoss(Loss):
    """
    Custom Multi-Label Loss function for multi-label classification.

    The objective function is a summed cross-entropy loss over all tasks,
    with each tasks's contribution to the loss being weighted by a factor
    of log(1+ class_imbalance_ratio), such that rarer tasks were given
    a higher weighting.

    This loss function is based on:
    `A Principal Odor Map Unifies Diverse Tasks in Human Olfactory Perception
    preprint <https://www.biorxiv.org/content/10.1101/2022.09.01.504602v4>`_.

    The labels should have shape (batch_size) or (batch_size, tasks), and be
    integer class labels.  The outputs have shape (batch_size, classes) or
    (batch_size, tasks, classes) and be logits that are converted to
    probabilities using a softmax function.
    """

    def __init__(self,
                 class_imbalance_ratio: Optional[List] = None,
                 loss_aggr_type: str = 'sum',
                 device: Optional[str] = None):
        """
        Parameters
        ---------
        class_imbalance_ratio: Optional[List]
            list of class imbalance ratios
        loss_aggr_type: str
            loss aggregation type; 'sum' or 'mean'
        device: Optional[str]
            The device on which to run computations. If None, a device is
            chosen automatically.
        """
        super(CustomMultiLabelLoss, self).__init__()
        if class_imbalance_ratio is None:
            print(Warning("No class imbalance ratio provided!"))
            self.class_imbalance_ratio: Optional[torch.Tensor] = None
        else:
            self.class_imbalance_ratio = torch.Tensor(class_imbalance_ratio)

        if loss_aggr_type not in ['sum', 'mean']:
            raise ValueError(f"Invalid loss aggregate type: {loss_aggr_type}")
        self.loss_aggr_type: str = loss_aggr_type

        if device is not None:
            if self.class_imbalance_ratio is not None:
                self.class_imbalance_ratio = self.class_imbalance_ratio.to(
                    device)

    def _create_pytorch_loss(
            self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns loss function for pytorch backend
        """
        ce_loss_fn: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(
            reduction='none')

        def loss(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            """
            The objective function is a summed cross-entropy loss over all
            tasks, with each tasks's contribution to the loss being weighted
            by a factor of log(1+ class_imbalance_ratio), such that rarer
            tasks were given a higher weighting.

            Parameters
            ---------
            output: torch.Tensor
                Output logits from model's forward pass per batch
            labels: torch.Tensor
                Target labels per batch

            Returns
            -------
            loss: torch.Tensor
                total or mean loss depending on loss aggregation type
            """
            # Convert (batch_size, tasks, classes)
            # to (batch_size, classes, tasks)
            # CrossEntropyLoss only supports (batch_size, classes, tasks)
            # This is for API consistency
            if len(output.shape) == 3:
                output = output.permute(0, 2, 1)

            if len(labels.shape) == len(output.shape):
                labels = labels.squeeze(-1)

            # handle multilabel
            # output shape => (batch_size, classes=1, tasks)
            # binary_output shape => (batch_size, classes=2, tasks)
            # where now we have (1 - probabilities) for ce loss calculation
            probabilities: torch.Tensor = output[:, 0, :]
            complement_probabilities: torch.Tensor = 1 - probabilities
            binary_output: torch.Tensor = torch.stack(
                [complement_probabilities, probabilities], dim=1)

            ce_loss: torch.Tensor = ce_loss_fn(binary_output, labels.long())

            if self.class_imbalance_ratio is None:
                if self.loss_aggr_type == 'sum':
                    loss: torch.Tensor = ce_loss.sum(dim=1)
                else:
                    loss = ce_loss.mean(dim=1)
            else:
                balancing_factors: torch.Tensor = torch.log(
                    1 + self.class_imbalance_ratio)

                # loss being weighted by a factor of
                # log(1+ class_imbalance_ratio)
                balanced_losses: torch.Tensor = torch.mul(
                    ce_loss, balancing_factors)

                if self.loss_aggr_type == 'sum':
                    # sum balanced loss across all tasks;
                    # shape => (batch_size)
                    loss = balanced_losses.sum(dim=1)
                else:
                    # mean balanced loss across all tasks;
                    # shape => (batch_size)
                    loss = balanced_losses.mean(dim=1)

            # duplicate loss across all tasks in a batch;
            # shape => (batch_size, n_tasks)
            # This is for API consistency
            return loss.unsqueeze(-1).repeat(1, output.shape[-1])

        return loss
