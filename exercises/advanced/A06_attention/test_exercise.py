import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lab.exercise_loader import load_target  # noqa: E402

ex = load_target(__file__)


def test_softmax_is_stable():
    out = ex.softmax(np.array([[1000.0, 1001.0, 1002.0]]))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    np.testing.assert_allclose(out, ex.softmax(np.array([[0.0, 1.0, 2.0]])))


def test_causal_mask():
    assert ex.causal_mask(3).tolist() == [[True, False, False], [True, True, False], [True, True, True]]


def test_attention_matches_pytorch():
    rng = np.random.default_rng(0)
    Q, K, V = rng.normal(size=(5, 8)), rng.normal(size=(7, 8)), rng.normal(size=(7, 4))
    out, weights = ex.scaled_dot_product_attention(Q, K, V)
    reference = torch.nn.functional.scaled_dot_product_attention(*(torch.tensor(a).unsqueeze(0) for a in (Q, K, V)))
    np.testing.assert_allclose(out, reference.squeeze(0).numpy(), atol=1e-6)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert weights.shape == (5, 7)


def test_causal_attention_has_exact_zeros():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 4))
    out, weights = ex.scaled_dot_product_attention(X, X, X, mask=ex.causal_mask(6))
    assert np.all(weights[np.triu_indices(6, k=1)] == 0.0)
    reference = torch.nn.functional.scaled_dot_product_attention(*(torch.tensor(X).unsqueeze(0) for _ in range(3)), is_causal=True)
    np.testing.assert_allclose(out, reference.squeeze(0).numpy(), atol=1e-6)


def test_multi_head_matches_pytorch():
    torch.manual_seed(0)
    d_model, n_heads, n = 16, 4, 5
    mha = torch.nn.MultiheadAttention(d_model, n_heads, bias=False, batch_first=True).double()
    X = np.random.default_rng(2).normal(size=(n, d_model))
    W_in = mha.in_proj_weight.detach().numpy()
    W_q, W_k, W_v = (W_in[i * d_model:(i + 1) * d_model].T for i in range(3))
    W_o = mha.out_proj.weight.detach().numpy().T
    ours = ex.multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads)
    with torch.no_grad():
        reference, _ = mha(*(torch.tensor(X).unsqueeze(0) for _ in range(3)))
    np.testing.assert_allclose(ours, reference.squeeze(0).numpy(), atol=1e-6)
    with pytest.raises(ValueError):
        ex.multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads=3)
