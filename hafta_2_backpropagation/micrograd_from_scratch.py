import math
import os
from threading import local
from turtle import forward
from mpmath import pade
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
from numpy._core import ubyte
import matplotlib.pyplot as plt
from sympy.geometry.polygon import y
import torch

# çıktılar her zaman bu dosyanın yanındaki output/ klasörüne yazılsın,
# script'in nereden çalıştırıldığından bağımsız olarak
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# for graph visualization stuff
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, filename='output/graph'):
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    out_path = os.path.join(BASE_DIR, filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dot.render(out_path, view=False, cleanup=True)
    return dot

# 1. Kendi `Value` sınıfını yaz — toplama ve çarpma ile başla. Her yeni `Value`,
# kendini üreten `Value`'ları ve hangi işlemden çıktığını saklasın. İstersen computation graph'i graphviz ile çizdir.
class Value:

    def __init__(self, data, _children=(), _op='', label='') -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})" ###, _op={self._op}, _prev={self._prev}, label={self.label})"

    def __add__(self, other):
        # for addition to integer by value
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+', label='')
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out

    def __mul__(self, other):
        # for multiplication to integer by value
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*', label='')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    # reverse multiplication for error handling
    def __rmul__(self, other):
        return self * other


    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh', label='')
        def _backward():
            self.grad += out.grad * (1 - t**2)
        out._backward = _backward
        return out


    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp', label='')
        def _backward():
            self.grad += out.grad * math.exp(self.data)
        out._backward = _backward
        return out

    def __pow__(self, other):
        out = Value(self.data ** other, (self,), f'**{other}', label='')
        def _backward():
            self.grad += out.grad * other * self.data ** (other - 1)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rtruediv__(self, other):
        return other * self**-1


    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for node in reversed(topo):
            node.grad = 0.0


def sigmoid(x):
    return 1 / (1 + (-x).exp())


# .grad değerlerini hesaplamak için finite difference yöntemi
def local_variable():
    h = 0.001
    a = Value(0.5, label='a')
    b = Value(0.3, label='b')
    c = Value(0.4, label='c')
    d = a * b; d.label = 'd'
    e = d + c; e.label = 'e'
    f = Value(0.2,label = 'f')
    L = e * f; L.label = 'L'
    L1 = L.data

    a = Value(0.76, label='a')
    b = Value(0.87, label='b')
    # a.data += h
    #b.data += h
    c = Value(0.98, label='c')
    d = a * b; d.label = 'd'
    #d.data += h
    e = d + c; e.label = 'e'
    #e.data += h
    f = Value(0.2,label = 'f')
    # f.data += h
    L = e * f; L.label = 'L'
    L2 = L.data

    print((L2 - L1)/h)
    p = Value(2.0)
    q = Value(3.0)
    L.grad = 1.0
    f.grad = -2.0
    e.data = -7.0
    d.grad = -7.0
    c.grad = -7.0
    b.grad = -14.0
    a.grad = 21.0
    return L

local_variable()

def noron():
    x1 = Value(3, label= 'x1')
    x2 = Value(0.3, label='x2')
    w1 = Value(0.9, label='w1')
    w2 = Value(1, label='w2')
    b = Value(2, label='b')
    x1w1 = x1 * w1; x1w1.label = 'x1w1'
    x2w2 = x2 * w2; x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
    n = x1w1x2w2 + b; n.label = 'n'
    #n = x1w1 + b
    # -----
    e = (2*n).exp(); e.label = 'e'
    o = (e - 1) / (e + 1) ; o.label = 'tanh'
    o.backward()
    draw_dot(o, filename='output/noron_tanh')
    n.zero_grad()
    o = n.tanh()
    o.backward()
    draw_dot(o,filename='output/noron_tanh_zero_grad')
    # ----
    # we should reset the grads before backward
    n.zero_grad()
    s = 1 / (1 + (-1 * n).exp())
    s.label = 's'
    s.backward()
    draw_dot(s, filename='output/noron_sigmoid')
    n.zero_grad()


    s = sigmoid(n)
    s.backward()
    draw_dot(s, filename='output/noron_sigmoid_zero_grad')


noron()
# graph visualization for local_variable() function
# print(draw_dot(local_variable()))


def torch_test():
    x1 = torch.tensor([3.0]).double(); x1.requires_grad = True
    x2 = torch.tensor([0.3]).double(); x2.requires_grad = True
    w1 = torch.tensor([0.9]).double(); w1.requires_grad = True
    w2 = torch.tensor([1.0]).double(); w2.requires_grad = True
    b = torch.tensor([2.0]).double(); b.requires_grad = True
    x1w1 = x1 * w1; x1w1.label = 'x1w1'
    x2w2 = x2 * w2; x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
    n = x1w1x2w2 + b; n.label = 'n'

    o = torch.tanh(n)
    print(o.data.item())
    o.backward()

    print("----")
    print("x2", x2.grad.item())
    print("w2", w2.grad.item())
    print("x1", x1.grad.item())
    print("w1", w1.grad.item())

torch_test()

class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(np.random.uniform(-1, 1))

    def __call__(self, x):
        return (np.dot(self.w, x) + self.b).tanh()

    def parameters(self):
        return self.w + [self.b]



class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]


    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        print(sz)
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


x = [2, 3, -1]
c = MLP(3, [4, 4, 1])

print(c(x))
draw_dot(c(x), 'output/mlp_graph')

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]


for k in range(30):
    ypred = [c(x) for x in xs]
    loss = sum([(yp - y)**2 for y, yp in zip(ys, ypred)])
    for p in c.parameters():
        p.grad = 0.0
    loss.backward()
    for p in c.parameters():
        p.data += -0.07 * p.grad
    print(f"k={k}, loss={loss.data}")

draw_dot(loss, 'output/mlp_loss')

print("ypred", ypred)
