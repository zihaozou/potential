import torch
import torch.nn as nn
from torch.autograd import grad as torch_grad
from skimage.metrics import peak_signal_noise_ratio as PSNR


def mypsnr(img1,img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))
def nesterov(f, x0,gt,max_iter=30):
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
        print(PSNR(x[0,...].detach().permute(1,2,0).cpu().numpy(),gt[0,...].detach().permute(1,2,0).cpu().numpy(),data_range=1.0))

    return x

def anderson(f, x0, m=5, lam=1e-4, max_iter=30, tol=1e-5, beta = 1.0):
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
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:, 1:n+1, 0]
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break

    return X[:,k%m].view_as(x0)

def simpleIter(f, x0, gt,max_iter=30, tol=1e-5):
    x=x0
    lastpsnr=mypsnr(gt,x0)
    for k in range(max_iter):
        xnext = f(x).detach()
        nowpsnr=mypsnr(gt,xnext)
        #print(nowpsnr)
        if mypsnr(gt,xnext)<lastpsnr:
            break
        x = xnext
        lastpsnr=nowpsnr
    return x
class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver_img, solver_grad, **kwargs):
        super().__init__()
        self.f = f
        self.solver_img = solver_img
        self.solver_grad = solver_grad
        self.kwargs = kwargs
        self.sigmaFactor=torch.nn.parameter.Parameter(torch.tensor([1.8]))
    def forward(self, n_y, kernel,sigma,gt):
        n_y.requires_grad_()
        sigma=sigma*self.sigmaFactor
        self.f.initialize_prox(n_y,kernel)
        n_ipt=self.f.calculate_prox(n_y)
        z= self.solver_img(lambda z : self.f(z,sigma,False), n_ipt, gt,**self.kwargs)
        z = self.f(z, sigma,self.training)
        # set up Jacobian vector product (without additional forward calls)
        if self.training:
            z0 = z.clone().detach().requires_grad_()
            f0 = self.f(z0, sigma,True)
            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()
                g = self.solver_grad(lambda y : torch_grad(f0, z0, y, retain_graph=True)[0] + grad,
                                                grad, **self.kwargs)
                
                return g

            self.hook=z.register_hook(backward_hook)
        output=self.f.denoise(z,sigma,self.training)
        # print(PSNR(output[0,...].detach().permute(1,2,0).cpu().numpy(),gt[0,...].detach().permute(1,2,0).cpu().numpy(),data_range=1.0))
        return output