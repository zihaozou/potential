import torch
import torch.nn as nn
from torch.autograd import grad as torch_grad

def nesterov(f, x0,max_iter=150):
    """ nesterov acceleration for fixed point iteration. """
    res = []
    imgs = []

    x = x0
    s = x.clone()
    t = torch.tensor(1., dtype=torch.float32)
    for k in range(max_iter):

        xnext = f(s).detach()
        
        # acceleration

        tnext = 0.5*(1+torch.sqrt(1+4*t*t))

        s = xnext + ((t-1)/tnext)*(xnext-x)
        
        # update
        t = tnext
        x = xnext
        

    return x

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]

        # (bsz x n)
        alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break

    return X[:,k%m].view_as(x0)



class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver_img, solver_grad, **kwargs):
        super().__init__()
        self.f = f
        self.solver_img = solver_img
        self.solver_grad = solver_grad
        self.kwargs = kwargs
        
    def forward(self, n_y, kernel,sigma):
        self.f.initialize_prox(n_y,kernel)
        n_ipt=self.f.calculate_prox(n_y)
        z= self.solver_img(lambda z : self.f(z,sigma,False), n_ipt, **self.kwargs)
        z = self.f(z.requires_grad_(), sigma)
        # set up Jacobian vector product (without additional forward calls)
        if self.training:
            z0 = z.clone().detach().requires_grad_()
            f0 = self.f(z0, sigma)
            def backward_hook(grad):
                g = self.solver_grad(lambda y : torch_grad(f0, z0, y, retain_graph=True)[0] + grad,
                                                grad, **self.kwargs)
                return g
            
            z.register_hook(backward_hook)
        
        return z