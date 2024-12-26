import torch

from basic_pinn.pinn import make_forward_fn_1d, LinearNN


def test_make_forward_fn_1d():
    model = LinearNN(num_layers=2)
    fns = make_forward_fn_1d(model, derivative_order=2)

    batch_size = 10
    x = torch.randn(batch_size)
    # params = dict(model.named_parameters())
    params = dict(model.named_parameters())

    fn_x = fns[0](x, params)
    assert fn_x.shape[0] == batch_size

    dfn_x = fns[1](x, params)
    assert dfn_x.shape[0] == batch_size

    ddfn_x = fns[2](x, params)
    assert ddfn_x.shape[0] == batch_size
