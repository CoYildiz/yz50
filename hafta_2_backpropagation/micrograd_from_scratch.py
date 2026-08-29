import math
from threading import local
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

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
    dot.render(filename, view=False, cleanup=True)
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
        return f"Value(data={self.data}, _op={self._op}, _prev={self._prev}, label={self.label})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+', label='')
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*', label='')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh', label='')
        def _backward():
            self.grad = out.grad * (1 - t**2)
        out._backward = _backward
        return out



# .grad değerlerini hesaplamak için finite difference yöntemi
def local_variable():
    h = 0.001
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(4.0, label='c')
    d = a * b; d.label = 'd'
    e = d + c; e.label = 'e'
    f = Value(-7.0,label = 'f')
    L = e * f; L.label = 'L'
    L1 = L.data

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    # a.data += h
    #b.data += h
    c = Value(4.0, label='c')
    d = a * b; d.label = 'd'
    #d.data += h
    e = d + c; e.label = 'e'
    #e.data += h
    f = Value(-7.0,label = 'f')
    # f.data += h
    L = e * f; L.label = 'L'
    L2 = L.data

    print((L2 - L1)/h)
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
    x = Value(1.0, label= 'x')
    x1 = x + x
    x1.label = 'x1'
    x2 = Value(3.0, label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    b = Value(6.0, label='b')
    x1w1 = x1 * w1; x1w1.label = 'x1w1'
    x2w2 = x2 * w2; x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
    n = x1w1x2w2 + b; n.label = 'n'
    o = n.tanh(); o.label = 'o'
    o.grad = 1.0
    o._backward()
    n._backward()
    x1w1x2w2._backward()
    x2w2._backward()
    x1w1._backward()
    b._backward()
    w2._backward()
    w1._backward()
    x1._backward()
    x2._backward()
    draw_dot(o, filename='output/noron')


noron()
print(draw_dot(local_variable()))
