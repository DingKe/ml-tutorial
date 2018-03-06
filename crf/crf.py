'''
Pytorch CRF
'''
import torch
from torch.autograd import Function, Variable
import torch.nn as nn


class CRFLoss(nn.Module):

    def __init__(self, vocab_size, start=False, end=False, size_average=True):
        super(CRFLoss, self).__init__()
        self.vocab_size = vocab_size
        self.start = start
        self.end = end
        self.size_average = size_average

        self.P = nn.Parameter(torch.Tensor(vocab_size, vocab_size))

        if self.start:
            self.S = nn.Parameter(torch.Tensor(vocab_size))
        else:
            self.register_parameter('S', None)

        if self.end:
            self.E = nn.Parameter(torch.Tensor(vocab_size))
        else:
            self.register_parameter('E', None)

    def reset_parameters(self):
        nn.init.normal(self.P.data, 0, 1)
        if self.S is not None:
            nn.init.constant(self.S, 0)
        if self.E is not None:
            nn.init.constant(self.E, 0)

    def forward(self, logits, labels, auto_grad=True):
        if auto_grad:
            return self._forward(logits, labels)
        else:
            return crf(logits, labels, self.P, self.S,
                       self.E, self.size_average)

    def _forward(self, logits, labels):
        batch_size, seq_len, voc = logits.size()
        log_alpha = Variable(logits.data.new(batch_size, voc).fill_(0))

        if self.S is not None:
            log_alpha = log_alpha + self.S.unsqueeze(0).expand(batch_size, voc)
        log_alpha = log_alpha + logits[:, 0, :]

        for t in range(1, seq_len):
            trans = self.P.unsqueeze(0).expand(
                batch_size, voc, voc)  # transmit score
            emit = logits[:, t, :].unsqueeze(1).expand(
                batch_size, voc, voc)  # emit score
            log_alpha_tm1 = log_alpha.unsqueeze(2).expand(batch_size, voc, voc)
            log_alpha = reduce_logsumexp(trans + emit + log_alpha_tm1, dim=1)

        if self.E is not None:
            log_Z = reduce_logsumexp(
                log_alpha + self.E.unsqueeze(0).expand(batch_size, voc), dim=1)
        else:
            log_Z = reduce_logsumexp(log_alpha, dim=1)

        # score for y
        labels_l = labels[:, :-1]
        expanded_size = labels_l.size() + (voc,)
        labels_l = labels_l.unsqueeze(-1).expand(expanded_size)

        labels_r = labels[:, 1:]
        labels_r = labels_r.unsqueeze(-1)

        P_row = self.P.unsqueeze(0).expand(
            batch_size, voc, voc).gather(1, labels_l)
        y_transmit_score = P_row.gather(2, labels_r).squeeze(-1)
        y_emit_score = logits.gather(2, labels.unsqueeze(2)).squeeze(-1)

        log_M = torch.sum(y_emit_score, dim=1) + \
            torch.sum(y_transmit_score, dim=1)

        if self.S is not None:
            log_M = log_M + self.S.gather(0, labels[:, 0])

        if self.E is not None:
            log_M = log_M + self.E.gather(0, labels[:, -1])

        # negative likelihood
        nll = log_Z - log_M
        nll = nll.sum(0).view(1)

        if self.size_average:
            nll.div_(batch_size)

        return nll


def reduce_logsumexp(input, dim=None):
    if dim is None:
        max_val = torch.max(input)
        ret = max_val + torch.log(torch.sum(torch.exp(input - max_val)))
        return ret
    else:
        max_val, _ = torch.max(input, dim=dim, keepdim=True)
        ret = max_val.squeeze(dim=dim) + \
            torch.log(torch.sum(torch.exp(input - max_val), dim=dim))
        return ret


def logsumexp(a, b):
    max_val = torch.max(a, b)

    dtype = a.type()
    tmp = (b - a) * (a > b).type(dtype) + (a - b) * (a <= b).type(dtype)

    return max_val + torch.log1p(torch.exp(tmp))


def one_hot(size, index):
    '''
    voc = size[-1]
    ret = index.expand(*size) == \
        torch.arange(0, voc).type(torch.LongTensor).unsqueeze(0).expand(*size)
    return ret
    '''
    mask = torch.LongTensor(*size).fill_(0)
    ret = mask.scatter_(1, index, 1)
    return ret


def _crf_forward(logits, P, S=None, E=None):
    batch_size, seq_len, voc = logits.size()
    log_alpha = logits.new(batch_size, seq_len, voc).fill_(0)

    if S is not None:
        log_alpha[:, 0, :] = S.unsqueeze(0).expand(batch_size, voc)
    else:
        log_alpha[:, 0, :] = 0
    log_alpha[:, 0, :] += logits[:, 0, :]

    for t in range(1, seq_len):
        trans = P.unsqueeze(0).expand(batch_size, voc, voc)  # transmit score
        emit = logits[:, t, :].unsqueeze(1).expand(
            batch_size, voc, voc)  # emit score

        log_alpha_tm1 = log_alpha[:, t - 1,
                                  :].unsqueeze(2).expand(batch_size, voc, voc)
        log_alpha[:, t, :] = reduce_logsumexp(
            trans + emit + log_alpha_tm1, dim=1)

    return log_alpha


def _crf_backward(logits, P, S=None, E=None):
    batch_size, seq_len, voc = logits.size()
    log_beta = logits.new(batch_size, seq_len, voc)

    if E is not None:
        log_beta[:, -1, :] = E.unsqueeze(0).expand(batch_size, voc)
    else:
        log_beta[:, -1, :] = 0

    for t in range(seq_len - 2, -1, -1):
        trans = P.unsqueeze(0).expand(
            batch_size, voc, voc)  # transmit score
        emit = logits[:, t + 1, :].unsqueeze(1).expand(
            batch_size, voc, voc)  # emit score
        log_beta_tp1 = log_beta[:, t + 1,
                                :].unsqueeze(1).expand(batch_size, voc, voc)

        log_beta[:, t, :] = reduce_logsumexp(
            trans + emit + log_beta_tp1, dim=2)

    return log_beta


class CRFF(Function):
    '''
    '''
    @staticmethod
    def forward(ctx, logits, labels, P, S=None, E=None, size_average=True):
        batch_size, seq_len, voc = logits.size()

        ctx.size_average = size_average
        ctx.log_alpha = log_alpha = _crf_forward(logits, P, S, E)
        ctx.S = S
        ctx.E = E

        # norm
        if E is not None:
            ctx.log_Z = log_Z = reduce_logsumexp(
                log_alpha[:, -1, :] +
                E.unsqueeze(0).expand(batch_size, voc), dim=1)
        else:
            ctx.log_Z = log_Z = reduce_logsumexp(log_alpha[:, -1, :], dim=1)

        # score for y
        labels_l = labels[:, :-1]
        expanded_size = labels_l.size() + (voc,)
        labels_l = labels_l.unsqueeze(-1).expand(expanded_size)

        labels_r = labels[:, 1:]
        labels_r = labels_r.unsqueeze(-1)

        P_row = P.unsqueeze(0).expand(batch_size, voc, voc).gather(1, labels_l)
        y_trans = P_row.gather(2, labels_r).squeeze(-1)
        y_emit = logits.gather(2, labels.unsqueeze(2)).squeeze(-1)

        log_M = torch.sum(y_emit, dim=1) + torch.sum(y_trans, dim=1)

        if S is not None:
            log_M += S.gather(0, labels[:, 0])

        if E is not None:
            log_M += E.gather(0, labels[:, -1])

        # negative likelihood
        nll = log_Z - log_M
        nll = nll.sum(0).view(1)

        if size_average:
            nll.div_(batch_size)

        ctx.save_for_backward(logits, labels, P)

        return nll

    @staticmethod
    def backward(ctx, output_grad):
        logits, labels, P = ctx.saved_variables
        logits = logits.data
        labels = labels.data
        P = P.data
        S, E = ctx.S, ctx.E

        batch_size, seq_len, voc = logits.size()
        dtype = output_grad.data.type()

        log_alpha = ctx.log_alpha
        log_beta = _crf_backward(logits, P, S, E)
        log_Z = ctx.log_Z.unsqueeze(-1)

        # storage for gradients
        logits_grad = Variable(logits.new(logits.size()).fill_(0))
        P_grad = Variable(P.new(P.size()).fill_(0))
        if S is not None:
            S_grad = Variable(S.new(S.size()).fill_(0))
        else:
            S_grad = None
        if E is not None:
            E_grad = Variable(E.new(E.size()).fill_(0))
        else:
            E_grad = None

        # end boundary
        if E_grad is not None:
            log_psi = E.unsqueeze(0).expand(batch_size, voc)
            delta = one_hot([batch_size, voc], labels[:, -1:]).type(dtype)
            delta_log_psi = torch.exp(
                log_alpha[:, -1, :] - log_Z + log_psi) - delta

            E_grad.data += delta_log_psi.sum(0)

        # normal cases
        for t in range(1, seq_len):
            for i in range(voc):
                emit = logits[:, t, :]
                trans = P[i, :].unsqueeze(0)
                log_psi = emit + trans

                left = (labels[:, t - 1] ==
                        i).unsqueeze(-1).expand(batch_size, voc)
                right = labels[:, t].unsqueeze(-1).expand(batch_size, voc) ==\
                    torch.arange(0, voc).type(torch.LongTensor).\
                    unsqueeze(0).expand(batch_size, voc)
                delta = (left * right).type(dtype)

                delta_log_psi = torch.exp(log_alpha[:, t - 1, i:i + 1] +
                                          log_beta[:, t, :] -
                                          log_Z + log_psi) - delta

                logits_grad.data[:, t, :] += delta_log_psi
                P_grad.data[i, :] += delta_log_psi.sum(0)

        # start boundary
        log_psi = logits[:, 0, :] + \
            (S.unsqueeze(0) if S is not None else 0)
        delta = one_hot([batch_size, voc], labels[:, :1]).type(dtype)
        delta_log_psi = torch.exp(log_beta[:, 0, :] - log_Z + log_psi) - delta
        logits_grad.data[:, 0, :] += delta_log_psi
        if S_grad is not None:
            S_grad.data += delta_log_psi.sum(0)

        if ctx.size_average:
            logits_grad.data.div_(batch_size)
            P_grad.data.div_(batch_size)
            if S_grad is not None:
                S_grad.data.div_(batch_size)
            if E_grad is not None:
                E_grad.data.div_(batch_size)

        return logits_grad, None, P_grad, S_grad, E_grad, None


crf = CRFF.apply


def test_crf_forward():
    import numpy as np

    logits = torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    P = torch.Tensor(np.ones([3, 3]))
    S = torch.Tensor([1, 2, 3])
    E = None

    log_alpha = _crf_forward(logits, P, S, E)
    print(log_alpha)


def test_crf_backward():
    import numpy as np

    logits = torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    P = torch.Tensor(np.ones([3, 3]))
    S = None
    E = torch.Tensor([1, 2, 3])
    E = None

    log_beta = _crf_backward(logits, P, S, E)
    print(log_beta)


def test_forward():
    batch_size = 3
    seq_len = 10
    voc = 7

    torch.manual_seed(1111)

    labels = torch.multinomial(torch.rand(
        batch_size, voc), seq_len, replacement=True)
    logits = torch.rand(batch_size, seq_len, voc)
    P = torch.randn(voc, voc) * 0.1
    S = torch.randn(voc) * 0.1
    E = torch.randn(voc) * 0.1

    logits = Variable(logits, requires_grad=True)
    labels = Variable(labels, requires_grad=True)
    P = Variable(P, requires_grad=True)
    S = Variable(S, requires_grad=True)
    E = Variable(E, requires_grad=True)

    nll = CRFF.apply(logits, labels, P, S, E)
    print(nll)


def test_backward():
    batch_size = 3
    seq_len = 10
    voc = 7

    torch.manual_seed(1111)

    labels = torch.multinomial(torch.rand(
        batch_size, voc), seq_len, replacement=True)
    logits = torch.rand(batch_size, seq_len, voc)
    P = torch.randn(voc, voc) * 0.1
    S = torch.randn(voc) * 0.1
    E = torch.randn(voc) * 0.1

    logits = Variable(logits, requires_grad=True)
    labels = Variable(labels, requires_grad=True)
    P = Variable(P, requires_grad=True)
    S = Variable(S, requires_grad=True)
    E = Variable(E, requires_grad=True)

    nll = CRFF.apply(logits, labels, P, S, E)
    nll.backward()

    print(logits.grad.data)
    print(P.grad.data)
    print(S.grad.data)
    print(E.grad.data)


def test_grad():
    import numpy as np

    batch_size = 3
    seq_len = 10
    voc = 7

    torch.manual_seed(1111)

    labels = torch.multinomial(torch.rand(
        batch_size, voc), seq_len, replacement=True)
    logits = torch.rand(batch_size, seq_len, voc)
    emit_score = torch.rand(batch_size, seq_len, voc)
    P = torch.randn(voc, voc) * 0.1
    S = torch.randn(voc) * 0.1
    E = torch.randn(voc) * 0.1

    logits = Variable(logits, requires_grad=True)
    labels = Variable(labels, requires_grad=True)
    P = Variable(P, requires_grad=True)
    S = Variable(S, requires_grad=True)
    E = Variable(E, requires_grad=True)

    nll = CRFF.apply(logits, labels, P, S, E)
    nll.backward()

    delta = 1e-3
    toleration = 5e-3

    # P
    print('P')
    for i in range(voc):
        for j in range(voc):
            V = P.data.numpy()

            o = V[i, j]
            grad_1 = P.grad.data.numpy()[i, j]

            V[i, j] = o + delta
            l1 = CRFF.apply(logits, labels, P, S, E).data.numpy()[0]
            V[i, j] = o - delta
            l2 = CRFF.apply(logits, labels, P, S, E).data.numpy()[0]
            V[i, j] = o

            grad_2 = (l1 - l2) / (2 * delta)

            diff = np.abs((grad_1 - grad_1))
            if diff > toleration:
                print("%.2e, %.2e, %.2e" % (grad_1, grad_2, diff))

    # logits
    print('logits')
    for i in range(batch_size):
        for j in range(seq_len):
            for k in range(voc):
                V = logits.data.numpy()

                o = V[i, j, k]
                grad_1 = logits.grad.data.numpy()[i, j, k]

                V[i, j, k] = o + delta
                l1 = CRFF.apply(logits, labels, P, S, E).data.numpy()[0]
                V[i, j, k] = o - delta
                l2 = CRFF.apply(logits, labels, P, S, E).data.numpy()[0]
                V[i, j, k] = o

                grad_2 = (l1 - l2) / (2 * delta)

                diff = np.abs((grad_1 - grad_2))
                if diff > toleration:
                    print("%.2e, %.2e, %.2e" % (grad_1, grad_2, diff))


def test_module():
    import numpy as np
    torch.manual_seed(1111)

    batch_size = 3
    seq_len = 10
    voc = 7

    labels = torch.multinomial(torch.rand(
        batch_size, voc), seq_len, replacement=True)
    logits = torch.rand(batch_size, seq_len, voc)

    logits = Variable(logits, requires_grad=True)
    labels = Variable(labels)

    crf_model = CRFLoss(vocab_size=voc, start=True, end=True)
    crf_model.reset_parameters()

    l1 = crf_model(logits, labels, auto_grad=False)
    l1.backward()
    grads_1 = [logits.grad.data.clone()] +\
        [param.grad.data.clone() for param in crf_model.parameters()]

    crf_model.zero_grad()
    logits.grad.data.fill_(0)
    l2 = crf_model(logits, labels, auto_grad=True)
    l2.backward()
    grads_2 = [logits.grad.data.clone()] +\
        [param.grad.data.clone() for param in crf_model.parameters()]

    toleration = 5e-7
    delta = 1e-4
    for g1, g2 in zip(grads_1, grads_2):
        g1 = g1.view(-1)
        g2 = g2.view(-1)
        for i in range(g1.size(0)):
            # print('%.2e, %.2e, %.2e' % (g1[i], g2[i], g1[i] - g2[i]))
            pass

        print('\nOverview:')
        print('%.5e, %.5e, %.15e, %.5e' % (l1.data.sum(), l2.data.sum(),
                                           (l1 - l2).data.sum(),
                                           torch.sum(torch.abs(g1 - g2))))

    # logits
    print('logits')
    grad_1 = grads_1[0]
    grad_2 = grads_2[0]
    for i in range(batch_size):
        for j in range(seq_len):
            for k in range(voc):
                V = logits.data.numpy()

                o = V[i, j, k]
                g1 = grad_1.numpy()[i, j, k]
                g2 = grad_2.numpy()[i, j, k]

                V[i, j, k] = o + delta
                l1 = crf_model(logits, labels).data.sum()
                V[i, j, k] = o - delta
                l2 = crf_model(logits, labels).data.sum()
                V[i, j, k] = o

                g3 = (l1 - l2) / (2 * delta)

                diff = np.abs((g2 - g1))
                if diff > toleration:
                    print("%.2e, %.2e, %.2e, %.2e" % (g1, g2, g3, diff))

    # P
    print('P')
    grad_1 = grads_1[1]
    grad_2 = grads_2[1]
    for i in range(voc):
        for j in range(voc):
            V = crf_model.P.data.numpy()

            o = V[i, j]
            g1 = grad_1.numpy()[i, j]
            g2 = grad_2.numpy()[i, j]

            V[i, j] = o + delta
            l1 = crf_model(logits, labels).data.sum()
            V[i, j] = o - delta
            l2 = crf_model(logits, labels).data.sum()
            V[i, j] = o

            g3 = (l1 - l2) / (2 * delta)

            diff = np.abs((g2 - g1))
            if diff > toleration:
                print("%.2e, %.2e, %.2e, %.2e" % (g1, g2, g3, diff))

    # S
    print('S')
    grad_1 = grads_1[2]
    grad_2 = grads_2[2]
    for i in range(voc):
        V = crf_model.S.data.numpy()

        o = V[i]
        g1 = grad_1.numpy()[i]
        g2 = grad_2.numpy()[i]

        V[i] = o + delta
        l1 = crf_model(logits, labels).data.sum()
        V[i] = o - delta
        l2 = crf_model(logits, labels).data.sum()
        V[i] = o

        g3 = (l1 - l2) / (2 * delta)

        diff = np.abs((g2 - g1))
        if diff > toleration:
            print("%.2e, %.2e, %.2e, %.2e" % (g1, g2, g3, diff))

    # E
    print('E')
    grad_1 = grads_1[3]
    grad_2 = grads_2[3]
    for i in range(voc):
        V = crf_model.E.data.numpy()

        o = V[i]
        g1 = grad_1.numpy()[i]
        g2 = grad_2.numpy()[i]

        V[i] = o + delta
        l1 = crf_model(logits, labels).data.sum()
        V[i] = o - delta
        l2 = crf_model(logits, labels).data.sum()
        V[i] = o

        g3 = (l1 - l2) / (2 * delta)

        diff = np.abs((g2 - g1))
        if diff > toleration:
            print("%.2e, %.2e, %.2e, %.2e" % (g1, g2, g3, diff))


if __name__ == '__main__':
    test_crf_forward()
    test_crf_backward()
    test_forward()
    test_backward()
    test_grad()
    test_module()
