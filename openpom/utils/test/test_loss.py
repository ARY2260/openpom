import torch
from openpom.utils.loss import CustomMultiLabelLoss


def test_custom_multilabel_loss_sum():
    """
    Test CustomMultiLabelLoss with class imbalance ratio and sum aggregation
    """
    class_imbalance_ratio = [1.0, 0.5, 0.25]
    loss = CustomMultiLabelLoss(class_imbalance_ratio, loss_aggr_type='sum')
    loss_fn = loss._create_pytorch_loss()
    sample_output = torch.Tensor([[[0.75], [0.25],
                                   [0.90]]])  # shape: (1, 3, 1)
    sample_target = torch.Tensor([[1.0, 0.0, 0.0]])

    computed_loss = loss_fn(sample_output, sample_target)
    assert computed_loss.shape == (1, 3)
    assert torch.allclose(computed_loss,
                          torch.Tensor([[0.7822, 0.7822, 0.7822]]),
                          atol=0.001)


def test_custom_multilabel_loss_mean():
    """
    Test CustomMultiLabelLoss with class imbalance ratio and mean aggregation
    """
    class_imbalance_ratio = [1.0, 0.5, 0.25]
    loss = CustomMultiLabelLoss(class_imbalance_ratio, loss_aggr_type='mean')
    loss_fn = loss._create_pytorch_loss()
    sample_output = torch.Tensor([[[0.75], [0.25],
                                   [0.90]]])  # shape: (1, 3, 1)
    sample_target = torch.Tensor([[[1.0], [0.0], [0.0]]])

    computed_loss = loss_fn(sample_output, sample_target)
    assert computed_loss.shape == (1, 3)
    assert torch.allclose(computed_loss,
                          torch.Tensor([[0.2607, 0.2607, 0.2607]]),
                          atol=0.001)


def test_custom_multilabel_no_class_imbalance():
    """
    Test CustomMultiLabelLoss with no class imbalance ratio
    """
    loss = CustomMultiLabelLoss(loss_aggr_type='mean')
    loss_fn = loss._create_pytorch_loss()
    sample_output = torch.Tensor([[[0.75], [0.25],
                                   [0.90]]])  # shape: (1, 3, 1)
    sample_target = torch.Tensor([[[1.0], [0.0], [0.0]]])

    computed_loss = loss_fn(sample_output, sample_target)
    assert computed_loss.shape == (1, 3)
    assert torch.allclose(computed_loss,
                          torch.Tensor([[0.7064, 0.7064, 0.7064]]),
                          atol=0.001)
