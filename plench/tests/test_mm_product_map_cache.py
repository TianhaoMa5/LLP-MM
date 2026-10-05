"""GPU product maps must share storage without changing the polynomial loss."""
import pytest
import torch

from mo_matching.llp.multiclass import (
    _multiply_polynomials,
    _polynomial_product_map_tensor_cpu,
    _polynomial_product_map_tensor_device,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA cache test')
def test_gpu_product_map_is_shared_and_keeps_values_and_gradients():
    device = torch.device('cuda:0')
    first = _polynomial_product_map_tensor_device(3, 2, 2, device)
    second = _polynomial_product_map_tensor_device(3, 2, 2, device)
    assert first.data_ptr() == second.data_ptr()
    torch.manual_seed(19)
    left = torch.randn(2, 6, device=device, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 6, device=device, dtype=torch.float64, requires_grad=True)
    expected_left = left.detach().clone().requires_grad_()
    expected_right = right.detach().clone().requires_grad_()
    destination = _polynomial_product_map_tensor_cpu(3, 2, 2).to(device)
    products = (expected_left.unsqueeze(2) * expected_right.unsqueeze(1)).reshape(2, -1)
    expected = torch.zeros(2, 15, device=device, dtype=torch.float64).scatter_add(
        1, destination.unsqueeze(0).expand(2, -1), products
    )
    actual = _multiply_polynomials(left, right, classes=3, left_order=2, right_order=2)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(left.grad, expected_left.grad)
    torch.testing.assert_close(right.grad, expected_right.grad)
