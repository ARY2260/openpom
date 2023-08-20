from openpom.utils.optimizer import get_optimizer


def test_get_optimizer():
    """
    Test get_optimizer function
    """
    optim = get_optimizer(optimizer_name='adam')
    assert optim.__class__.__name__ == 'Adam'

    optim = get_optimizer(optimizer_name='adagrad')
    assert optim.__class__.__name__ == 'AdaGrad'

    optim = get_optimizer(optimizer_name='adamw')
    assert optim.__class__.__name__ == 'AdamW'

    optim = get_optimizer(optimizer_name='sparseadam')
    assert optim.__class__.__name__ == 'SparseAdam'

    optim = get_optimizer(optimizer_name='rmsprop')
    assert optim.__class__.__name__ == 'RMSProp'

    optim = get_optimizer(optimizer_name='sgd')
    assert optim.__class__.__name__ == 'GradientDescent'

    optim = get_optimizer(optimizer_name='kfac')
    assert optim.__class__.__name__ == 'KFAC'

    optim = get_optimizer(optimizer_name='APPLE')  # test invalid name
    assert optim.__class__.__name__ == 'Adam'
