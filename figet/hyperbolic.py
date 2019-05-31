from figet import utils
from figet.Constants import EPS
import torch
from torch.autograd import Function
from numpy import clip
from numpy.linalg import norm
import math


log = utils.get_logging()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def hyperbolic_distance_numpy(p, q):
    return hyperbolic_distance(norm(p), norm(q), norm(p - q))


def hyperbolic_distance(p_norm, q_norm, p_minus_q_norm):
    numerator = 2 * p_minus_q_norm * p_minus_q_norm
    denominator = (1 - p_norm * p_norm) * (1 - q_norm * q_norm)
    denominator = clip(denominator, EPS, denominator)
    return math.acosh(1 + numerator / denominator)


def hyperbolic_distance_torch(p, q):
    return poincare_distance(torch.from_numpy(p).to(DEVICE), torch.from_numpy(q).to(DEVICE))


def poincare_distance(u, v):
    """
    From: https://github.com/facebookresearch/poincare-embeddings/blob/master/model.py#L48
    """
    boundary = 1 - 1e-5
    squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, boundary)
    sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, boundary)
    sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
    x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
    # arcosh
    z = torch.sqrt(torch.pow(x, 2) - 1)
    return torch.log(x + z)


class PoincareDistance(Function):
    boundary = 1 - EPS

    @staticmethod
    def forward(ctx, u, v):
        squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, PoincareDistance.boundary)
        sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, PoincareDistance.boundary)
        sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        # arcosh
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = PoincareDistance.grad(u, v, squnorm, sqvnorm, sqdist)
        gv = PoincareDistance.grad(v, u, sqvnorm, squnorm, sqdist)

        grad_u = g.expand_as(gu) * gu
        grad_v = g.expand_as(gv) * gv

        corrected_u = PoincareDistance.apply_riemannian_correction(squnorm, grad_u)
        corrected_v = PoincareDistance.apply_riemannian_correction(sqvnorm, grad_v)

        return corrected_u, corrected_v

    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) / torch.pow(alpha, 2)).unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = torch.sqrt(torch.pow(z, 2) - 1)
        z = torch.clamp(z * beta, min=EPS).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

    @staticmethod
    def apply_riemannian_correction(sqxnorm, gradient):
        # corrected_gradient = gradient * ((1 - sqxnorm.unsqueeze(-1)) ** 2 / 4).expand_as(gradient)
        return gradient.clamp(min=-10000.0, max=10000.0)


def polarization_identity(u, v):
    """
    :param u, v: (n x embed_dim) Tensors with hyperbolic embeddings
    :return: Tensor of shape (n x 1) with results of applying the function

    Formula taken from: https://en.wikipedia.org/wiki/Polarization_identity#Other_forms_for_real_vector_spaces

    This function is the equivalent of the dot product expressed using the norm in the given geometry (normed space).
    This function is applicable only in geometries where the Parallelogram law holds.
    """
    squnorm = hyperbolic_norm(u) ** 2
    sqvnorm = hyperbolic_norm(v) ** 2
    u_minus_v = hyperbolic_norm(u - v) ** 2
    return (squnorm + sqvnorm - u_minus_v) * 0.5


def hyperbolic_norm(u):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    origin = torch.zeros(u.size()).to(device)
    return PoincareDistance.apply(origin, u)


def normalize(predicted_emb):
    """
    Projects the embedding with a norm above 1 (one) inside the Poincare ball
    """
    return torch.renorm(predicted_emb, 2, 0, 1 - EPS)
