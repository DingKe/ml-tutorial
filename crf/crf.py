'''
Pytorch CRF
'''
import torch
from torch.autograd import Function, Variable
from torch.autograd.function import once_differentiable

import numpy as np


def reduce_logsumexp(input, dim=None):
    if dim is None:
        max_val = torch.max(input)
        ret = max_val + torch.log(torch.sum(torch.exp(input - max_val)))
        return ret
    else:
        max_val, _ = torch.max(input, dim=dim, keepdim=True)
        ret = max_val.squeeze(dim=dim) + torch.log(torch.sum(torch.exp(input - max_val), dim=dim))
        return ret


def logsumexp(a, b):
    max_val = torch.max(a, b)

    dtype = a.type()
    tmp = (b - a) * (a > b).type(dtype) + (a - b) * (a <= b).type(dtype)

    return max_val + torch.log1p(torch.exp(tmp))


def one_hot(size, index):
    '''
    voc = size[-1]
    ret = index.expand(*size) == torch.arange(0, voc).type(torch.LongTensor).unsqueeze(0).expand(*size)
    return ret
    '''
    mask = torch.LongTensor(*size).fill_(0)
    ret = mask.scatter_(1, index, 1)
    return ret


def _crf_forward(emit_score, P, S=None, E=None):
    batch_size, seq_len, voc = emit_score.size()
    log_alpha = emit_score.new(batch_size, seq_len, voc)

    # temp storage
    energy = P.new(batch_size, voc, voc)

    if S is not None:
       log_alpha[:, 0, :] = S.unsqueeze(0).expand(batch_size, voc)
    else:
       log_alpha[:, 0, :] = 0
    log_alpha[:, 0, :] += emit_score[:, 0, :]

    for t in range(1, seq_len):
        energy[:] = P.unsqueeze(0).expand(batch_size, voc, voc)  # transmit score
        energy += emit_score[:, t, :].unsqueeze(1).expand(batch_size, voc, voc)  # emit score

        log_alpha_tm1 = log_alpha[:, t - 1, :].unsqueeze(1).expand(batch_size, voc, voc)
        log_alpha[:, t, :] = reduce_logsumexp(log_alpha_tm1 + energy, dim=-1)

    return log_alpha


def _crf_backward(emit_score, P, S=None, E=None):
    batch_size, seq_len, voc = emit_score.size()
    log_beta = emit_score.new(batch_size, seq_len, voc)

    # temp storage
    energy = P.new(batch_size, voc, voc)

    if E is not None:
       log_beta[:, -1, :] = E.unsqueeze(0).expand(batch_size, voc)
    else:
       log_beta[:, -1, :] = 0

    for t in range(seq_len - 2, -1, -1):
        energy[:] = P.transpose(0, 1).unsqueeze(0).expand(batch_size, voc, voc)  # transmit score
        energy += emit_score[:, t, :].unsqueeze(1).expand(batch_size, voc, voc)  # emit score

        log_beta_tp1 = log_beta[:, t + 1, :].unsqueeze(-1).expand(batch_size, voc, voc)
        log_beta[:, t, :] = reduce_logsumexp(log_beta_tp1 + energy, dim=1)

    if S is not None:
        log_beta[:, 0, :] += S.unsqueeze(0).expand(batch_size, voc)


    return log_beta


class CRFF(Function):
    '''
    '''
    @staticmethod
    def forward(ctx, input, y, P, S=None, E=None, size_average=True):
        emit_score, labels = input, y 

        batch_size, seq_len, voc = input.size()

        ctx.size_average = size_average
        ctx.log_alpha = _crf_forward(emit_score, P, S, E)

        # norm
        if E is not None:
            ctx.log_Z = reduce_logsumexp(ctx.log_alpha[:, -1, :] + E.unsqueeze(0).expand(batch_size, voc), dim=-1)
        else:
            ctx.log_Z = reduce_logsumexp(ctx.log_alpha[:, -1, :], dim=-1)

        # score for y
        labels_l = labels[:, :-1]
        expanded_size = labels_l.size() + (voc,)
        labels_l = labels_l.unsqueeze(-1).expand(expanded_size)

        labels_r = labels[:, 1:]
        labels_r = labels_r.unsqueeze(-1)

        P_row = P.unsqueeze(0).expand(batch_size, voc, voc).gather(1, labels_l)
        y_transmit_score = P_row.gather(2, labels_r).squeeze(-1)
        y_emit_score = emit_score.gather(2, labels.unsqueeze(2)).squeeze(-1)

        log_M = torch.sum(y_emit_score, dim=1) + torch.sum(y_transmit_score, dim=1)

        if S is not None:
            log_M += S.gather(0, labels[:, 0])

        if E is not None:
            log_M += E.gather(0, labels[:, -1])

        # negative likelihood
        nll = ctx.log_Z - log_M
        nll = nll.sum(0).view(1)

        if size_average:
            nll.div_(batch_size)

        ctx.save_for_backward(input, y, P, S, E)

        return nll

    @staticmethod
    def backward(ctx, output_grad):
        input, y, P, S, E = ctx.saved_variables

        emit_score, labels = input, y
        batch_size, seq_len, voc = input.size()
        dtype = output_grad.data.type()

        log_alpha = ctx.log_alpha
        log_beta = _crf_backward(input.data, P.data,
                                 S.data if S is not None else None,
                                 E.data if E is not None else None)
        log_Z = ctx.log_Z.unsqueeze(-1)

        # storage for gradients
        input_grad = Variable(input.data.new(input.size()).fill_(0))
        P_grad = Variable(P.data.new(P.size()).fill_(0))

        if S is not None:
            S_grad = Variable(S.data.new(S.size()).fill_(0))
        else:
            S_grad = None

        if E is not None:
            E_grad = Variable(E.data.new(E.size()).fill_(0))
        else:
            E_grad = None

        # end boundary
        if E is not None:
            log_psi = E.data.unsqueeze(0).expand(batch_size, voc)
            delta = one_hot([batch_size, voc], labels.data[:, -1:]).type(dtype)
            delta_log_psi = torch.exp(log_alpha[:, -1, :] - log_Z + log_psi) - delta

            E_grad.data += delta_log_psi.sum(0)

        # normal cases
        for t in range(1, seq_len):
            for i in range(voc):
                log_psi = emit_score.data[:, t, :] + P.data[i, :].unsqueeze(0)
                left = (labels.data[:, t - 1] == i).unsqueeze(-1).expand(batch_size, voc) 
                right = labels.data[:, t].unsqueeze(-1).expand(batch_size, voc) == torch.arange(0, voc).type(torch.LongTensor).unsqueeze(0).expand(batch_size, voc)
                delta = (left * right).type(dtype)
                delta_log_psi = torch.exp(ctx.log_alpha[:, t - 1, i:i+1] + log_beta[:, t, :] - log_Z + log_psi) - delta
                input_grad.data[:, t, :] += delta_log_psi
                P_grad.data[i, :] += delta_log_psi.sum(0)

        # start boundary
        log_psi = emit_score.data[:, 0, :] + (S.data.unsqueeze(0).expand(batch_size, voc) if S is not None else 0)
        delta = one_hot([batch_size, voc], labels.data[:, :1]).type(dtype)
        delta_log_psi = torch.exp(log_beta[:, 0, :] - log_Z + log_psi) - delta

        input_grad.data[:, 0, :] += delta_log_psi
        if S is not None:
            S_grad.data += delta_log_psi.sum(0)

        if ctx.size_average:
            input_grad.data.div_(y.size(0))
            P_grad.data.div_(y.size(0))
            if S_grad is not None:
                S_grad.data.div_(y.size(0))
            if E_grad is not None:
                E_grad.data.div_(y.size(0))

        return input_grad, None, P_grad, S_grad, E_grad, None


def test_crf_forward():
    batch_size = 3
    seq_len = 10
    voc = 7

    torch.manual_seed(1111)

    emit_score = torch.rand(batch_size, seq_len, voc)
    P = torch.randn(voc, voc) * 0.1
    S = torch.randn(voc) * 0.1
    E = torch.randn(voc) * 0.1

    log_alpha = _crf_forward(emit_score, P, S, E)
    print(log_alpha[0])


def test_crf_backward():
    batch_size = 3
    seq_len = 10
    voc = 7

    torch.manual_seed(1111)

    emit_score = torch.rand(batch_size, seq_len, voc)
    P = torch.randn(voc, voc) * 0.1
    S = torch.randn(voc) * 0.1
    E = torch.randn(voc) * 0.1

    log_beta = _crf_backward(emit_score, P, S, E)
    print(log_beta[0])


def test_forward():
    batch_size = 3
    seq_len = 10
    voc = 7

    torch.manual_seed(1111)

    labels = torch.multinomial(torch.rand(batch_size, voc), seq_len, replacement=True)
    emit_score = torch.rand(batch_size, seq_len, voc)
    P = torch.randn(voc, voc) * 0.1
    S = torch.randn(voc) * 0.1
    E = torch.randn(voc) * 0.1

    emit_score = Variable(emit_score, requires_grad=True)
    labels = Variable(labels, requires_grad=True)
    P = Variable(P, requires_grad=True)
    S = Variable(S, requires_grad=True)
    E = Variable(E, requires_grad=True)

    nll = CRFF.apply(emit_score, labels, P, S, E)
    print(nll)


def test_backward():
    batch_size = 3
    seq_len = 10
    voc = 7

    torch.manual_seed(1111)

    labels = torch.multinomial(torch.rand(batch_size, voc), seq_len, replacement=True)
    emit_score = torch.rand(batch_size, seq_len, voc)
    P = torch.randn(voc, voc) * 0.1
    S = torch.randn(voc) * 0.1
    E = torch.randn(voc) * 0.1

    emit_score = Variable(emit_score, requires_grad=True)
    labels = Variable(labels, requires_grad=True)
    P = Variable(P, requires_grad=True)
    S = Variable(S, requires_grad=True)
    E = Variable(E, requires_grad=True)

    nll = CRFF.apply(emit_score, labels, P, S, E)
    nll.backward()

    print(emit_score.grad.data)
    print(P.grad.data)
    print(S.grad.data)
    print(E.grad.data)


def test_grad():
    batch_size = 3
    seq_len = 10
    voc = 7

    torch.manual_seed(1111)

    labels = torch.multinomial(torch.rand(batch_size, voc), seq_len, replacement=True)
    emit_score = torch.rand(batch_size, seq_len, voc)
    P = torch.randn(voc, voc) * 0.1
    S = torch.randn(voc) * 0.1
    E = torch.randn(voc) * 0.1

    emit_score = Variable(emit_score, requires_grad=True)
    labels = Variable(labels, requires_grad=True)
    P = Variable(P, requires_grad=True)
    S = Variable(S, requires_grad=True)
    E = Variable(E, requires_grad=True)

    nll = CRFF.apply(emit_score, labels, P, S, E)
    nll.backward()

    delta = 1e-4
    toleration = 1e-2

    # P
    print('P')
    for i in range(voc):
        for j in range(voc):
            V = P.data.numpy()

            o = V[i, j]
            grad_1 = P.grad.data.numpy()[i, j]

            V[i, j] = o + delta
            l1 = CRFF.apply(emit_score, labels, P, S, E).data.numpy()[0]
            V[i, j] = o - delta
            l2 = CRFF.apply(emit_score, labels, P, S, E).data.numpy()[0]
            V[i, j] = o

            grad_2 = (l1 - l2) / (2 * delta)

            diff = np.abs((grad_1 - grad_2) / grad_1)
            if diff > toleration:
                print("%.2e, %.2e, %.2e" % (grad_1, grad_2, diff))

    # emit
    print('emit_score')
    for i in range(batch_size):
        for j in range(seq_len):
            for k in range(voc):
                V = emit_score.data.numpy()

                o = V[i, j, k]
                grad_1 = emit_score.grad.data.numpy()[i, j, k]

                V[i, j, k] = o + delta
                l1 = CRFF.apply(emit_score, labels, P, S, E).data.numpy()[0]
                V[i, j, k] = o - delta
                l2 = CRFF.apply(emit_score, labels, P, S, E).data.numpy()[0]
                V[i, j, k] = o

                grad_2 = (l1 - l2) / (2 * delta)

                diff = np.abs((grad_1 - grad_2) / grad_1)
                if diff > toleration:
                    print("%.2e, %.2e, %.2e" % (grad_1, grad_2, diff))



    nll = CRFF.apply(emit_score, labels, P, S, E)
    nll.backward()



if __name__ == '__main__':
    #test_crf_forward()
    #test_crf_backward()
    #test_forward()
    #test_backward()

    test_grad()
