import matplotlib as plt
from fastai.basics import *

n = 100
x = torch.ones(n,2)

x[:,0].uniform_(-1.,1)
x[:5]

theta = tensor(3.,2); theta
y = x@theta + torch.rand(n)
plt.scatter(x[:, 0], y)


def mse(y_hat, y):
    return ((y_hat-y)**2).mean()


a = tensor(-2.5, 1); a
y_hat = x@a

plt.scatter(x[:, 0], y)
plt.scatter(x[:, 0], y_hat)


nn.Parameter(a)

def update():
    y_hat = x@a
    loss = mse(y, y_hat)
    if t % 10 == 0: print(loss)
    loss.backward()
    with torch.no_grad():
        a.sub_(lr * a.grad)
        a.grad.zero_()
