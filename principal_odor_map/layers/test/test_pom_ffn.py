import torch
from principal_odor_map.layers.pom_ffn import CustomPositionwiseFeedForward


def test_custom_position_wise_feed_forward():
    """Test invoking CustomPositionwiseFeedForward."""
    torch.manual_seed(0)
    input_ar = torch.tensor([[1., 2.], [5., 6.]])
    ffn = CustomPositionwiseFeedForward(d_input=2,
                                        d_hidden_list=[16, 3],
                                        d_output=2,
                                        activation='leakyrelu',
                                        dropout_p=0.1,
                                        dropout_at_input_no_act=True,
                                        batch_norm=True)
    assert len(ffn.batchnorms) == 2
    embbedding_result, output_result = ffn(input_ar)
    assert embbedding_result.shape == (2, 3)
    assert output_result.shape == (2, 2)


def test_custom_position_wise_feed_forward_no_batchnorm():
    """Test invoking CustomPositionwiseFeedForward without batch norm"""
    torch.manual_seed(0)
    input_ar = torch.tensor([[1., 2.], [5., 6.]])
    ffn = CustomPositionwiseFeedForward(d_input=2,
                                        d_hidden_list=[16, 3],
                                        d_output=2,
                                        activation='leakyrelu',
                                        dropout_p=0.1,
                                        dropout_at_input_no_act=True,
                                        batch_norm=False)
    assert not hasattr(ffn, 'batchnorms')
    embbedding_result, output_result = ffn(input_ar)
    assert embbedding_result.shape == (2, 3)
    assert output_result.shape == (2, 2)
