# Imports
import copy
import torch
import numpy as np
from scipy.stats import logistic
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.utils.multiclass import type_of_target
from sklearn.decomposition import TruncatedSVD
from torch.nn import functional as F

GLOBAL_MODEL_PETURB = 'relative'
SOMEWHAT_BIG_NUMBER = 10

"""
=========================================
Fairness Metric Learning + Approximations
=========================================
"""


def get_ell_infy_bounds(X_train, sens_inds):
    """
    Given X_train return weighted ell_p coefs corresponding to each
    feature dims correlation with the sensitive feature
    """
    corr_coefs = np.abs(np.corrcoef(X_train, rowvar=False))
    ell_coefs = []
    num_cols = X_train.shape[1]
    for i in range(num_cols):
        coef = 0.0
        if (i in sens_inds):
            pass
        else:
            for j in sens_inds:
                coef += np.abs(corr_coefs[i][j])
        if (coef == 0.0):
            ell_coefs.append(coef)
        else:
            ell_coefs.append(coef)
    ell_coefs = np.where(np.isnan(ell_coefs), 0, ell_coefs)  # numerical for when 0 data is available
    return ell_coefs


def get_mahalanobis_fairness(X_train, sens_inds):
    X = copy.deepcopy(X_train)
    
    
def compute_SenSR_matrix_source(data, protected_idxs, keep_protected_idxs=True):
    dtype = torch.Tensor(data).dtype

    # data = datautils.convert_tensor_to_numpy(data)
    basis_vectors_ = []
    num_attr = data.shape[1]

    # Get input data excluding the protected attributes
    protected_idxs = sorted(protected_idxs)
    free_idxs = [idx for idx in range(num_attr) if idx not in protected_idxs]
    X_train = data[:, free_idxs]
    Y_train = data[:, protected_idxs]

    # Update: extended support for continuous target type
    coefs = []
    for idx in range(len(protected_idxs)):
        y_arr = np.array(Y_train[:, idx])
        ttype = type_of_target(y_arr)
        if ttype == 'continuous':
            coefs.append(Lasso()
                         .fit(X_train, Y_train[:, idx])
                         .coef_.squeeze())
        # binary or multiclass or multilabel-indicator (this would break if the ttype is multiclass-multioutput,
        # continuous-multioutput)
        else:
            coefs.append(LogisticRegression(solver="lbfgs", penalty="l2")
                         .fit(X_train, Y_train[:, idx])
                         .coef_.squeeze())
    coefs = np.array(coefs)

    if keep_protected_idxs:
        # To keep protected indices, we add two basis vectors
        # First, with logistic regression coefficients with 0 in
        # protected indices. Second, with one-hot vectors with 1 in
        # protected indices.

        basis_vectors_ = np.empty(shape=(2 * len(protected_idxs), num_attr))

        for i, protected_idx in enumerate(protected_idxs):
            protected_basis_vector = np.zeros(shape=(num_attr))
            protected_basis_vector[protected_idx] = 1.0

            unprotected_basis_vector = np.zeros(shape=(num_attr))
            np.put_along_axis(
                unprotected_basis_vector, np.array(free_idxs), coefs[i], axis=0
            )

            basis_vectors_[2 * i] = unprotected_basis_vector
            basis_vectors_[2 * i + 1] = protected_basis_vector
    else:
        # Protected indices are to be discarded. Therefore, we can
        # simply return back the logistic regression coefficients
        basis_vectors_ = coefs

    basis_vectors_ = torch.tensor(basis_vectors_, dtype=dtype).T
    basis_vectors_ = basis_vectors_.detach()

    def get_span_of_sensitive_subspace(sensitive_subspace):
        """
        sensitive_subspace: the redundant sensitive subspace
        return: the span of the sensitive subspace
        """
        tSVD = TruncatedSVD(n_components=sensitive_subspace.shape[0])
        tSVD.fit(sensitive_subspace)
        span = tSVD.components_
        return span

    def complement_projector(span):
        """
        span: the span of the sensitive directions
        return: the orthogonal complement projector of the span
        """
        basis = span.T
        proj = np.linalg.pinv(basis.T @ basis)
        proj = basis @ proj @ basis.T
        proj_compl = np.eye(proj.shape[0]) - proj
        return proj_compl

    span = get_span_of_sensitive_subspace(basis_vectors_.T)
    metric_matrix = complement_projector(span)
    metric_matrix += np.eye(len(metric_matrix), dtype=np.float64) * 1  # Inflating the variance to help numerically
    return metric_matrix


def __grad_likelihood__(X, Y, sigma):
    """Computes the gradient of the likelihood function using sigmoidal link"""

    diag = np.einsum("ij,ij->i", np.matmul(X, sigma), X)
    diag = np.maximum(diag, 1e-10)
    prVec = logistic.cdf(diag)
    sclVec = 2.0 / (np.exp(diag) - 1)
    vec = (Y * prVec) - ((1 - Y) * prVec * sclVec)
    grad = np.matmul(X.T * vec, X) / X.shape[0]
    return grad


def __projPSD__(sigma):
    """Computes the projection onto the PSD cone"""

    try:
        L = np.linalg.cholesky(sigma)
        sigma_hat = np.dot(L, L.T)
    except np.linalg.LinAlgError:
        d, V = np.linalg.eigh(sigma)
        sigma_hat = np.dot(
            V[:, d >= 1e-8], d[d >= 1e-8].reshape(-1, 1) * V[:, d >= 1e-8].T
        )
    return sigma_hat


def learn_EXPLORE_hyperplane(X, Y, iters, batchsize):
    N = X.shape[0]
    P = X.shape[1]

    sigma_t = np.random.normal(0, 1, P ** 2).reshape(P, P)
    sigma_t = np.matmul(sigma_t, sigma_t.T)
    sigma_t = sigma_t / np.linalg.norm(sigma_t)

    curriter = 0

    while curriter < iters:
        batch_idxs = np.random.choice(N, size=batchsize, replace=False)
        X_batch = X[batch_idxs]
        Y_batch = Y[batch_idxs]

        grad_t = __grad_likelihood__(X_batch, Y_batch, sigma_t)
        t = 0.0001 #1.0 / (1 + curriter // 100)
        sigma_t = __projPSD__(sigma_t - t * grad_t)

        curriter += 1
        
    sigma = torch.FloatTensor(sigma_t).detach()
    return sigma


def get_explore_intervals(features, labels, sens_maj_inds, sens_min_inds, use_sens=True, eps=0.025):
    """
    Given a dataset of features differences compute the EXPLORE sensitive hyperplane, then 
    over-approximate the polytope with an orthotope
    """
    M = learn_EXPLORE_hyperplane(features, labels, 10000, 128)
    M = M + np.eye(len(M))*1e-3

    if (use_sens):
        l, U = np.linalg.eigh(M)
        ones = np.ones_like(l)

        # translate eps to distance space
        # translated eps = eps * max eigen value * sqrt(dimension)
        # eps = eps * np.max(l) * np.sqrt(2*M.shape[-1])

        # Zero sensitive columns
        for i in sens_maj_inds + sens_min_inds:
            U[:, i] = 0

        z_l = np.matmul(ones, U.T) - eps / np.sqrt(l)
        z_u = np.matmul(ones, U.T) + eps / np.sqrt(l)

        z_mu = (z_l + z_u) / 2
        z_rad = (z_u - z_l) / 2

        # Propagate the rotated variables through the matmul:
        x_center = np.matmul(z_mu, U)
        x_rad = np.matmul(z_rad, np.abs(U))

        # Now we have the interval transformed into the original space
        x_l = x_center - x_rad - 1
        x_u = x_center + x_rad - 1

        # Add large value to sens so classifier is insensitive
        for j in sens_min_inds + sens_maj_inds:
            x_l[j] = -10
            x_u[j] = 10
    else:
        M = np.delete(M, sens_maj_inds + sens_min_inds, 0)
        M = np.delete(M, sens_maj_inds + sens_min_inds, 1)

        l, U = np.linalg.eigh(M)
        ones = np.ones_like(l)

        # translate eps to distance space
        # translated eps = eps * max eigen value * sqrt(dimension)
        # eps = eps * np.max(l) * np.sqrt(2*M.shape[-1])

        z_l = np.matmul(ones, U.T) - eps / np.sqrt(l)
        z_u = np.matmul(ones, U.T) + eps / np.sqrt(l)

        z_mu = (z_l + z_u) / 2
        z_rad = (z_u - z_l) / 2

        # Propagate the rotated variables through the matmul:
        x_center = np.matmul(z_mu, U)
        x_rad = np.matmul(z_rad, np.abs(U))

        x_rad = x_rad  # / np.linalg.norm(x_rad)
        # Now we have the interval transformed to the origin of the original space
        x_l = x_center - x_rad - 1
        x_u = x_center + x_rad - 1

    return (x_u - x_l) / 2

def get_bounds_from_mahalanobis(M: np.ndarray) -> np.ndarray:
    """
    Approximates mahalanobis distance interval with axis aligned orthotope
    :param M: a pd matrix
    :return: interval that is an array of lengths such that [-interval, interval] (closely) includes points that are
             unit distance according to MH distance
    """
    #U, l, V = np.linalg.svd(M)
    #Lambda = np.diag(l)
    l, U = np.linalg.eigh(M)
    ones = np.ones_like(l)
    # U and V must be related by transpose since M is sq
    # over-approximation
    A = np.matmul(ones/np.sqrt(l), np.abs(U))
    # slightly better approximation
    # A = np.abs(np.linalg.pinv(np.diag(np.sqrt(l)) @ np.abs(U).transpose()) @ ones)
    # A = np.max(1/np.matmul(np.diag(np.sqrt(l)), np.abs(U).transpose()), axis=0)
    # norm of rows
    # interval_lens = np.linalg.norm(A, axis=1)
    interval_lens = A
    return interval_lens


def get_fairness_intervals(X_train, sens_inds, metric="LP", use_sens=True, eps=1):
    """
    Given a dataset and sensitive feature indexes, we return interval bounds over the level sets
    of each fairness metric type. We highlight that each level set is normalized in order that training
    epsilons and deltas all be comparable. 
    
    metric options - {"LP", "SENSR"}
    """
    if metric.lower() == "sensr":
        M = compute_SenSR_matrix_source(X_train, sens_inds, keep_protected_idxs=use_sens)

        interval_lens = get_bounds_from_mahalanobis(M)
        x_l, x_u = -eps*interval_lens, eps*interval_lens

        # Add large value to sens so classifier is insensitive
        if use_sens:
            for j in sens_inds:
                x_l[j] = -SOMEWHAT_BIG_NUMBER
                x_u[j] = SOMEWHAT_BIG_NUMBER
        else:
            x_l = np.delete(x_l, [sens_inds])
            x_u = np.delete(x_u, [sens_inds])

    elif metric.lower() == "lp":
        w = get_ell_infy_bounds(X_train, sens_inds)
        x_u = eps * w;
        x_l = -eps * w;

        if use_sens:
            for j in sens_inds:
                x_l[j] = -SOMEWHAT_BIG_NUMBER
                x_u[j] = SOMEWHAT_BIG_NUMBER
        else:
            x_l = np.delete(x_l, [sens_inds])
            x_u = np.delete(x_u, [sens_inds])


    # We can do this because all the interval bounds are symmetric
    # so can be  sufficiently described just by their width.
    return (x_u - x_l) / 2


"""
=============================================================
Interval Propagation Code for Learning Certifiably Fair NNs
=============================================================
"""


# Define forward propagation through a nn
def affine_forward(W, b, x_l, x_u):
    """
    This function uses pytorch to compute upper and lower bounds
    on a matrix multiplication given bounds on the matrix 'x' 
    as given by x_l (lower) and x_u (upper)
    """
    try:
        x_mu = (x_u + x_l) / 2
        x_r = (x_u - x_l) / 2
        h_mu = torch.matmul(x_mu, W.T)
        x_rad = torch.matmul(x_r, torch.abs(W).T)
        h_u = h_mu + x_rad + b
        h_l = h_mu - x_rad + b
        return h_l, h_u
    except:
        x_mu = (x_u.float() + x_l.float()) / 2
        x_r = (x_u.float() - x_l.float()) / 2
        h_mu = torch.matmul(x_mu, W.T.float())
        x_rad = torch.matmul(x_r, torch.abs(W.float()).T)
        h_u = h_mu + x_rad + b.float()
        h_l = h_mu - x_rad + b.float()
        return h_l, h_u

def interval_bound_forward(model, weights, inp, vec, eps):
    h_l = inp - (vec * eps);
    h_u = inp + (vec * eps)
    assert ((h_l <= h_u).all())
    num_layers = len(model.layers) 
    for i in range(len(model.layers)):
        if "LINEAR" in model.layers[i].upper():
            w, b = weights[2 * (i)], weights[(2 * (i)) + 1]
            h_l, h_u = affine_forward(w, b, h_l, h_u)
            #assert((h_l <= h_u).all())
            if i < num_layers - 1:  # Return Logits not Softmax Activation
                h_l = model.activations[i](h_l)
                h_u = model.activations[i](h_u)
        else:
            assert False, "Layers that are note linear layers are not supported at this time"
    return h_l, h_u


# Define standard robustness bounds as a sanity check

def fairness_regularizer(model, inp, lab, vec, eps, nclasses=10):
    """
    This class only works for binary classification at the moment. Can be generalized
    with a bit of effort modifying the for loop.
    """
    weights = [t for t in model.parameters()]
    logit_l, logit_u = interval_bound_forward(model, weights, inp, vec, eps)
    worst_delta = 0
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses)
    min_logit = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    max_logit = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
    min_i_softmax = F.softmax(min_logit, dim=-1)
    max_i_softmax = F.softmax(max_logit, dim=-1)
    worst_delta = torch.sum(torch.abs(max_i_softmax - min_i_softmax))
    return worst_delta  # F.cross_entropy(worst_case, lab)


def fairness_bounds(model, inp, lab, vec, eps, nclasses):
    weights = [t for t in model.parameters()]
    logit_l, logit_u = interval_bound_forward(model, weights, inp, vec, eps)
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses)
    worst_case = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    best_case = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
    y_l = F.softmax(worst_case, dim=-1)
    y_u = F.softmax(best_case, dim=-2)
    return y_l, y_u


def fairness_delta(model, inp, lab, vec, eps, nclasses):
    """
    This class only works for binary classification at the moment. Can be generalized
    with a bit of effort modifying the for loop.
    """
    weights = [t for t in model.parameters()]
    logit_l, logit_u = interval_bound_forward(model, weights, inp, vec, eps)    
    worst_delta = 0
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses)
    min_logit = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    max_logit = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
    min_i_softmax = F.softmax(min_logit, dim=-1)
    max_i_softmax = F.softmax(max_logit, dim=-1)
    delta = (max_i_softmax - min_i_softmax)
    delta = delta.detach().numpy()
    return np.max(delta, axis=1)


def fair_PGD(model, x_natural, lab, vec, eps, nclasses, iterations=10):
    x = x_natural.detach()
    eps_vec = vec*eps
    #noise = (eps_vec*torch.zeros_like(x).uniform_(0, 1))
    noise = (-2*eps_vec) * torch.zeros_like(x).uniform_(0, 1) + eps_vec
    x = x + (noise*eps)
    for i in range(iterations):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, lab)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x + (eps_vec/iterations * torch.sign(grad.detach()))
        #x = x.detach() + 0.5 * torch.sign(grad.detach())
        #x = torch.min(torch.max(x, x_natural - eps_vec), x_natural + eps_vec)
        x = torch.clip(x, x_natural-eps_vec, x_natural+eps_vec)      
    return x 

def fairness_delta_PGD(model, inp, lab, vec, eps, nclasses, iterations=10):
    y_pred = model(inp)
    x_adv = fair_PGD(model, inp, lab, vec, eps, nclasses, iterations)
    y_adv = model(x_adv)
    pgd_delta = torch.max(torch.abs(y_pred - y_adv), axis=1)
    return pgd_delta

def fairness_regularizer_PGD(model, inp, lab, vec, eps, nclasses, iterations=10):
    y_pred = model(inp)
    x_adv = fair_PGD(model, inp, lab, vec, eps, nclasses, iterations)
    y_adv = model(x_adv)
    regval = torch.sum(torch.abs(y_pred - y_adv))
    return regval
    
def RobustnessRegularizer(model, inp, lab, vec, eps, nclasses):
    inp_dim = inp.shape[-1]
    # vec = torch.ones([inp_dim]).to(inp.device).double()
    weights = [t for t in model.parameters()]
    logit_l, logit_u = interval_bound_forward(model, weights, inp, vec, eps)
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - v1
    worst_case = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    return F.cross_entropy(worst_case, lab)


def RobustnessBounds(model, inp, lab, vec, eps, nclasses):
    weights = [t for t in model.parameters()]
    logit_l, logit_u = interval_bound_forward(model, weights, inp, vec, eps)
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses)
    worst_case = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    best_case = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
    y_l = worst_case
    y_u = best_case
    return y_l, y_u


def global_pgd_ibp(model, vec, eps, npoints=128, iterations=50, lr=0.1, nclasses=2):
    points = torch.randn((npoints, model.in_dim))
    points.requires_grad_()
    lab = torch.randint(0, 1, (1,npoints)).long() #torch.ones((npoints)).long()
    vec = torch.Tensor(vec)
    for i in range(iterations):
        points.requires_grad_()
        with torch.enable_grad():
            weights = [t for t in model.parameters()]
            logit_l, logit_u = interval_bound_forward(model, weights, points, vec, eps)
            v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
            v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses)
            min_logit = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
            max_logit = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
            min_i_softmax = F.softmax(min_logit, dim=-1)
            max_i_softmax = F.softmax(max_logit, dim=-1)
            delta = torch.abs((max_i_softmax - min_i_softmax))
            loss = torch.mean(delta)
        grad = torch.autograd.grad(loss, [points])[0]
        #print("Grad: ", grad)
        points = points + (lr *  torch.sign(grad.detach()))
        points = torch.clip(points, -2.5, 2.5)
    return points

def global_pgd_ibp_s(model, vec, eps, gamma, X, iterations=10, lr=0.01, nclasses=2):
    points = copy.deepcopy(X) #torch.randn((npoints, model.in_dim))
    npoints = len(points)
    #print(npoints)
    points.requires_grad_()
    lab = torch.randint(0, 1, (1,npoints)).long() #torch.ones((npoints)).long()
    vec = torch.Tensor(vec)
    for i in range(iterations):
        points.requires_grad_()
        with torch.enable_grad():
            weights = [t for t in model.parameters()]
            logit_l, logit_u = interval_bound_forward(model, weights, points, vec, eps)
            v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
            v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses)
            min_logit = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
            max_logit = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
            min_i_softmax = F.softmax(min_logit, dim=-1)
            max_i_softmax = F.softmax(max_logit, dim=-1)
            delta = torch.abs((max_i_softmax - min_i_softmax))
            loss = torch.mean(delta)
        grad = torch.autograd.grad(loss, [points])[0]
        #print("Grad: ", grad)
        points = points + (lr *  torch.sign(grad.detach()))
        points = torch.clip(points, X-gamma, X+gamma)
    return points

"""
=============================================================
MULTIPLE OUTPUT Interval Propagation Code for Learning Certifiably Fair NNs
=============================================================
"""

def global_pgd_ibp_MO(model, vec, eps, npoints=128, iterations=100, lr=0.1, nclasses=2, index=0, seed_inputs=-1):
    if(type(seed_inputs) == int):
        points = torch.randn((npoints, 7))
        points.requires_grad_()
    else:
        points = seed_inputs
    lab = torch.randint(0, 1, (1,npoints)).long() #torch.ones((npoints)).long()
    vec = torch.Tensor(vec)
    for i in range(iterations):
        points.requires_grad_()
        with torch.enable_grad():
            weights = model.get_weights(index)
            logit_l, logit_u = interval_bound_forward(model, weights, points, vec, eps)
            v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
            v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses)
            min_logit = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
            max_logit = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
            min_i_softmax = F.softmax(min_logit, dim=-1)
            max_i_softmax = F.softmax(max_logit, dim=-1)
            delta = torch.abs((max_i_softmax - min_i_softmax))
            loss = torch.mean(delta)
        grad = torch.autograd.grad(loss, [points])[0]
        #print("Grad: ", grad)
        points = points + (lr *  torch.sign(grad.detach()))
        points = torch.clip(points, -2.5, 2.5)
    return points

def fair_PGD_MO(model, x_natural, lab, vec, eps, nclasses, iterations=10, index=0):
    x = x_natural.detach()
    eps_vec = vec*eps
    #noise = (eps_vec*torch.zeros_like(x).uniform_(0, 1))
    noise = (-2*eps_vec) * torch.zeros_like(x).uniform_(0, 1) + eps_vec
    x = x + (noise*eps)
    for i in range(iterations):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model.forward(x, index)
            loss = F.cross_entropy(logits, lab)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x + (eps_vec/iterations * torch.sign(grad.detach()))
        x = torch.clip(x, x_natural-eps_vec, x_natural+eps_vec)      
    return x 

def fairness_regularizer_PGD_MO(model, inp, lab, vec, eps, nclasses, iterations=10, index=0):
    y_pred = F.softmax(model.forward(inp, index))
    x_adv = fair_PGD_MO(model, inp, lab, vec, eps, nclasses, iterations, index)
    y_adv = F.softmax(model.forward(x_adv, index))
    regval = torch.sum(torch.abs(y_pred - y_adv))
    return regval


def fairness_regularizer_MO(model, inp, lab, vec, eps, nclasses=10, index=0):
    """
    This class only works for binary classification at the moment. Can be generalized
    with a bit of effort modifying the for loop.
    """
    weights = model.get_weights(index)
    logit_l, logit_u = interval_bound_forward(model, weights, inp, vec, eps)
    worst_delta = 0
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses)
    min_logit = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    max_logit = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
    min_i_softmax = F.softmax(min_logit, dim=-1)
    max_i_softmax = F.softmax(max_logit, dim=-1)
    worst_delta = torch.sum(torch.abs(max_i_softmax - min_i_softmax))
    return worst_delta  # F.cross_entropy(worst_case, lab)


def fairness_delta_PGD_MO(model, inp, lab, vec, eps, nclasses, iterations=10, index=0):
    y_pred = F.softmax(model.forward(inp, index)).detach().numpy()
    x_adv = fair_PGD_MO(model, inp, lab, vec, eps, nclasses, iterations, index)
    y_adv = F.softmax(model.forward(x_adv, index)).detach().numpy()
    regval = np.mean(np.abs(y_pred - y_adv), axis=1)
    return regval
    #y_pred = model(inp, index=index)
    #x_adv = fair_PGD_MO(model, inp, lab, vec, eps, nclasses, iterations, index)
    #y_adv = model.forward(x_adv, index=index)
    #regval = torch.max(y_pred - y_adv, axis=1)
    #return regval

def fairness_delta_MO(model, inp, lab, vec, eps, nclasses=10, index=0):
    """
    This class only works for binary classification at the moment. Can be generalized
    with a bit of effort modifying the for loop.
    """
    weights = model.get_weights(index)
    logit_l, logit_u = interval_bound_forward(model, weights, inp, vec, eps)
    worst_delta = 0
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses)
    min_logit = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    max_logit = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
    min_i_softmax = F.softmax(min_logit, dim=-1)
    max_i_softmax = F.softmax(max_logit, dim=-1)
    worst_delta = torch.max(max_i_softmax - min_i_softmax, axis=1)
    return worst_delta 



# ----------------- FOLKTABLES UTILS ----------------- 

from folktables import ACSDataSource, ACSIncome, generate_categories

ACS_categories = {
    "COW": {
        1.0: (
            "Employee of a private for-profit company or"
            "business, or of an individual, for wages,"
            "salary, or commissions"
        ),
        2.0: (
            "Employee of a private not-for-profit, tax-exempt,"
            "or charitable organization"
        ),
        3.0: "Local government employee (city, county, etc.)",
        4.0: "State government employee",
        5.0: "Federal government employee",
        6.0: (
            "Self-employed in own not incorporated business,"
            "professional practice, or farm"
        ),
        7.0: (
            "Self-employed in own incorporated business,"
            "professional practice or farm"
        ),
        8.0: "Working without pay in family business or farm",
        9.0: "Unemployed and last worked 5 years ago or earlier or never worked",
    },
    "SCHL": {
        1.0: "No schooling completed",
        2.0: "Nursery school, preschool",
        3.0: "Kindergarten",
        4.0: "Grade 1",
        5.0: "Grade 2",
        6.0: "Grade 3",
        7.0: "Grade 4",
        8.0: "Grade 5",
        9.0: "Grade 6",
        10.0: "Grade 7",
        11.0: "Grade 8",
        12.0: "Grade 9",
        13.0: "Grade 10",
        14.0: "Grade 11",
        15.0: "12th grade - no diploma",
        16.0: "Regular high school diploma",
        17.0: "GED or alternative credential",
        18.0: "Some college, but less than 1 year",
        19.0: "1 or more years of college credit, no degree",
        20.0: "Associate's degree",
        21.0: "Bachelor's degree",
        22.0: "Master's degree",
        23.0: "Professional degree beyond a bachelor's degree",
        24.0: "Doctorate degree",
    },
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
    },
    "SEX": {1.0: "Male", 2.0: "Female"},
    "RAC1P": {
        1.0: "White alone",
        2.0: "Black or African American alone",
        3.0: "American Indian alone",
        4.0: "Alaska Native alone",
        5.0: (
            "American Indian and Alaska Native tribes specified;"
            "or American Indian or Alaska Native,"
            "not specified and no other"
        ),
        6.0: "Asian alone",
        7.0: "Native Hawaiian and Other Pacific Islander alone",
        8.0: "Some Other Race alone",
        9.0: "Two or More Races",
    },
}