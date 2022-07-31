import torch.nn.utils as nn
import torch
import torch.nn.functional as F


def go_through_batches(
    a2c,
    exps,
    state_embedding_model,
    state_embedding_model_opt,
    inverse_model,
    inverse_model_opt,
    forward_model,
    forward_model_opt,
):
    """
    based on:
    https://github.com/lcswillems/torch-ac/blob
    /master/torch_ac/algos/a2c.py
    """
    forward_loss = 0
    inverse_loss = 0

    inds = a2c._get_starting_indexes()

    for i in range(a2c.recurrence):
        sb = exps[inds + i]
        sb = add_noisy_tv(sb)
        input_images = sb.obs.image.reshape(sb.obs.image.size()[0], 3, 7, 7)
        emb_t = state_embedding_model(input_images[:-1])
        emb_tp1 = state_embedding_model(input_images[1:])
        pred_actions = inverse_model(emb_t, emb_tp1)
        pred_emb_tp1, uncertainty = forward_model(emb_t, sb.action[1:].to(torch.int64))
        batch_forward_loss = F.mse_loss(
            pred_emb_tp1, emb_tp1.squeeze(2).squeeze(2), reduction="none"
        )
        # batch_forward_loss = torch.exp(-uncertainty) * batch_forward_loss
        batch_forward_loss = torch.mean(batch_forward_loss, dim=1)
        batch_forward_loss = torch.cat((batch_forward_loss, torch.FloatTensor([0])))
        sb.reward = batch_forward_loss
        batch_forward_loss = 0.5 * (
            torch.sum(batch_forward_loss) + torch.sum(uncertainty)
        )
        batch_inverse_loss = F.cross_entropy(
            pred_actions.squeeze(1).squeeze(1), exps.action[1:].to(torch.long)
        )
        print(exps[inds + i].reward)
        exps[inds + i] = sb
        print(exps[inds + i].reward)
        forward_loss += batch_forward_loss
        inverse_loss += batch_inverse_loss

    loss = torch.sum(forward_loss) + inverse_loss
    loss.backward()
    state_embedding_model_opt.zero_grad()
    forward_model_opt.zero_grad()
    inverse_model_opt.zero_grad()
    nn.clip_grad_norm_(state_embedding_model.parameters(), 40)
    nn.clip_grad_norm_(forward_model.parameters(), 40)
    nn.clip_grad_norm_(inverse_model.parameters(), 40)
    state_embedding_model_opt.step()
    forward_model_opt.step()
    inverse_model_opt.step()

    return (
        exps,
        state_embedding_model,
        state_embedding_model_opt,
        inverse_model,
        inverse_model_opt,
        forward_model,
        forward_model_opt,
    )


def add_noisy_tv(exps):
    for time_step in range(len(exps.action[:-1])):
        if exps.action[time_step] == 0:
            exps.obs.image[time_step + 1] = torch.randint(0, 8, (7, 7, 3))
    return exps
