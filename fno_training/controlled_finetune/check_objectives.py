"""Optional user-run mathematical checks; no model or optimizer is constructed."""


def main():
    import numpy as np
    import torch
    if __package__:
        from .paired_objectives import loss_terms, summarize_validation
    else:
        from paired_objectives import loss_terms, summarize_validation

    target = torch.arange(30, dtype=torch.float64).reshape(2, 3, 5) / 17 + .2
    pred = (target + torch.linspace(-.2, .3, 30).reshape(2, 3, 5)).requires_grad_()
    w = torch.tensor([.1, .2, .3, .2, .1], dtype=torch.float64)
    delta, coefficient = [.1, .2, .3], .5
    terms = loss_terms(pred, target, w, coefficient, delta)
    D = ((pred - target)**2 * w).sum(-1) / w.sum()
    A2 = (target**2 * w).sum(-1) / w.sum()
    torch.testing.assert_close(terms["mse"], ((pred - target)**2 * w).sum() / (6 * w.sum()))
    torch.testing.assert_close(terms["mixed"], ((1-coefficient)*D + coefficient*D/(A2 + torch.tensor(delta, dtype=torch.float64)**2)).mean(), rtol=1e-12, atol=1e-12)
    gradient = torch.autograd.grad(terms["mixed"], pred)[0]
    factor = (1-coefficient) + coefficient / (A2 + torch.tensor(delta, dtype=torch.float64)**2)
    torch.testing.assert_close(gradient, 2*(pred-target)*w*factor[..., None]/(6*w.sum()))
    torch.testing.assert_close(loss_terms(pred, target, w, 0., delta)["mixed"], terms["mse"])
    scales = np.array([2., .5, 3.])
    physical_pred = pred.detach().numpy() * scales[None, :, None]
    physical_target = target.numpy() * scales[None, :, None]
    metrics = summarize_validation(physical_pred, physical_target, w.numpy(), scales, coefficient, delta)
    alternate_floor = summarize_validation(physical_pred, physical_target, w.numpy(), scales, coefficient, [1., 2., 3.])
    assert metrics["fields"] == alternate_floor["fields"], "delta changed true errors"
    np.testing.assert_allclose(metrics["mse"], float(terms["mse"].detach()), rtol=1e-12)
    np.testing.assert_allclose(metrics["mixed_loss"], float(terms["mixed"].detach()), rtol=1e-12)
    zero = summarize_validation(np.ones((2, 3, 5)), np.zeros((2, 3, 5)), w.numpy(), scales, .5, delta)
    assert zero["macro_mean_relative_l2"] is None
    assert zero["joint_relative_le_1pct"]["pass_count"] == 0
    assert zero["joint_relative_le_1pct"]["undefined_count"] == 2
    print("Objective formula, analytic gradient, physical losses and zero-norm conventions passed. No optimizer/model was run.")


if __name__ == "__main__":
    main()
