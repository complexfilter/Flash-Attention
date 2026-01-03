import torch 
import torch.nn.functional as F

def my_softmax(x: torch.Tensor): 
    assert x.ndim == 1
    x_max = torch.amax(x)
    exp_ = torch.exp(x - x_max)
    return exp_ / torch.sum(exp_)

class Softmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        assert x.ndim == 1
        x_max = torch.amax(x)
        exp = torch.exp(x - x_max)
        y = exp / torch.sum(exp)
        ctx.y = y
        return y

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y 
        assert dy.ndim == 1 
        dx = y * dy - torch.sum(y * dy) * y
        return dx

if __name__ == "__main__":
    x = torch.randn(4096, dtype=torch.bfloat16, device="cuda").requires_grad_(True)
    y = F.softmax(x, dim=0)
    dy = torch.rand_like(y)
    y.backward(dy)
    dx = x.grad 

    y_ = my_softmax(x)
    # numerically verify the correctness of softmax forward
    print(torch.norm(y - y_, p='fro') / torch.norm(y, p='fro'))

    x_copy = x.detach().clone().requires_grad_(True) # need to add requires grad True, since clone() by default does not require grad. 
    y_softmax = Softmax.apply(x_copy)
    y_softmax.backward(dy)
    dx_copy = x_copy.grad
    # numerically verify the correctness of softmax backwward
    print(torch.norm(dx_copy - dx, p='fro') / torch.norm(dx, p='fro'))



    