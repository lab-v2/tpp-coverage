"""CPU checks of the trainer's loss masking, gradient accumulation, optimizers and schedules."""

from __future__ import annotations

import importlib
import sys

import pytest
import torch
import torch.nn as nn

from conftest import REPO, SRC, run

sys.path.insert(0, str(SRC / "training_models" / "training"))
T = importlib.import_module("training_models.training.DistributedTrainer")


def _ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def test_padded_loss_equals_the_loss_over_real_tokens():
    torch.manual_seed(0)
    vocab, seq = 11, 9
    lengths = [9, 5, 2]  # real tokens per row; the rest is padding (id 0)
    ids = torch.zeros(len(lengths), seq, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for r, n in enumerate(lengths):
        ids[r, :n] = torch.randint(1, vocab, (n,))
        mask[r, :n] = 1
    logits = torch.randn(len(lengths), seq - 1, vocab)

    inputs, labels, in_mask = T.shift_for_next_token(ids, mask)
    assert inputs.shape == labels.shape == in_mask.shape == (len(lengths), seq - 1)
    masked = _ce(logits, labels)

    # Independent reference: score each row's real next-token targets only, then average.
    per_token = []
    for r, n in enumerate(lengths):
        for t in range(n - 1):
            per_token.append(nn.functional.cross_entropy(logits[r, t][None], ids[r, t + 1][None]))
    reference = torch.stack(per_token).mean()
    torch.testing.assert_close(masked, reference)

    # The unmasked labels score the padding too and give a different number.
    unmasked = _ce(logits, ids[:, 1:])
    assert not torch.isclose(unmasked, reference)


def test_unpadded_batch_is_unchanged_by_the_mask():
    ids = torch.randint(0, 50, (4, 17))
    _, labels, _ = T.shift_for_next_token(ids, torch.ones_like(ids))
    assert torch.equal(labels, ids[:, 1:])


@pytest.mark.parametrize("n_examples,batch,accum", [(10, 4, 2), (12, 4, 4), (7, 6, 3), (8, 8, 1)])
def test_accumulated_gradient_equals_the_full_batch_gradient(n_examples, batch, accum):
    torch.manual_seed(1)
    x = torch.randn(n_examples, 5)
    y = torch.randint(0, 3, (n_examples,))
    micro = batch // accum
    micro_batches = [(x[i:i + micro], y[i:i + micro]) for i in range(0, n_examples, micro)]

    model = nn.Linear(5, 3)
    expected = []
    for lo in range(0, n_examples, batch):
        model.zero_grad()
        nn.CrossEntropyLoss()(model(x[lo:lo + batch]), y[lo:lo + batch]).backward()
        expected.append(model.weight.grad.clone())

    got, seen = [], 0
    model.zero_grad()
    for i, (xb, yb) in enumerate(micro_batches):
        group_size, n_group = T.accumulation_group(i, accum, len(micro_batches), batch, n_examples)
        (nn.CrossEntropyLoss()(model(xb), yb) * (len(xb) / n_group)).backward()
        seen += 1
        if seen == group_size:
            got.append(model.weight.grad.clone())
            model.zero_grad()
            seen = 0
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        torch.testing.assert_close(g, e)


@pytest.mark.parametrize("name,cls", [("adamw", torch.optim.AdamW), ("adam", torch.optim.AdamW),
                                      ("sgd", torch.optim.SGD), ("lion", T.Lion),
                                      ("adafactor", torch.optim.Adafactor)])
def test_make_optimizer_steps(name, cls):
    torch.manual_seed(2)
    model = nn.Linear(4, 1)
    opt = T.make_optimizer(name, model.parameters(), lr=1e-2, weight_decay=0.1)
    assert isinstance(opt, cls)
    if name == "adam":
        assert opt.param_groups[0]["weight_decay"] == 0.0
    before = model.weight.detach().clone()
    model(torch.randn(8, 4)).pow(2).mean().backward()
    opt.step()
    assert not torch.equal(before, model.weight)


def test_lion_update_is_the_sign_of_the_interpolated_momentum():
    p = nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    opt = T.Lion([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)
    p.grad = torch.tensor([0.3, -0.4, 0.0])
    opt.step()
    # First step: momentum is zero, so the update is -lr * sign(grad).
    torch.testing.assert_close(p.detach(), torch.tensor([0.9, -1.9, 0.5]))


def test_unknown_optimizer_is_rejected():
    with pytest.raises(ValueError, match="unknown optimizer"):
        T.make_optimizer("rmsprop", nn.Linear(2, 2).parameters(), 1e-3, 0.0)


def test_schedule_factor():
    assert T.schedule_factor("constant", 999, 1000) == 1.0
    assert T.schedule_factor("wsd", 0, 1000) == 1.0
    assert T.schedule_factor("wsd", 799, 1000) == 1.0
    assert T.schedule_factor("wsd", 900, 1000) == pytest.approx(0.55)
    assert T.schedule_factor("wsd", 1000, 1000) == pytest.approx(0.1)  # the cosine floor
    with pytest.raises(ValueError, match="unknown schedule"):
        T.schedule_factor("linear", 0, 10)


def test_trainer_cli_has_the_new_flags():
    out = run(["python", REPO / "src/training_models/training/dist_trainer_main.py", "--help"]).stdout
    for flag in ("--shape", "--optimizer", "--schedule", "--grad-accum-steps", "--seed",
                 "--score-padding", "fineweb"):
        assert flag in out, flag


def test_score_padding_reproduces_the_released_labels():
    ids = torch.tensor([[5, 6, 7, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    _, legacy, _ = T.shift_for_next_token(ids, mask, mask_padding=False)
    _, fixed, _ = T.shift_for_next_token(ids, mask)
    assert legacy.tolist() == [[6, 7, 0, 0]]
    assert fixed.tolist() == [[6, 7, -100, -100]]
    assert T.DistributedTrainer().mask_padding is True
