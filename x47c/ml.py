from __future__ import annotations
from typing import Any

class Tensor:
    def __init__(self, data: Any, requires_grad: bool = False, _children=(), _op: str = ""):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = 0.0 if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor({self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op="+")

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad or 0) + (out.grad or 0)
            if other.requires_grad:
                other.grad = (other.grad or 0) + (out.grad or 0)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op="*")

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad or 0) + other.data * (out.grad or 0)
            if other.requires_grad:
                other.grad = (other.grad or 0) + self.data * (out.grad or 0)
        out._backward = _backward
        return out

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


def relu(x: Tensor) -> Tensor:
    val = max(0, x.data) if isinstance(x.data, (int, float)) else x.data
    out = Tensor(val, requires_grad=x.requires_grad, _children=(x,), _op="relu")
    def _backward():
        if x.requires_grad:
            x.grad = (x.grad or 0) + ((out.grad or 0) if out.data > 0 else 0)
    out._backward = _backward
    return out


class Linear:
    def __init__(self, in_features: int, out_features: int):
        import random
        self.weight = Tensor([[random.uniform(-0.1, 0.1) for _ in range(out_features)] for _ in range(in_features)], requires_grad=True)
        self.bias = Tensor([0.0] * out_features, requires_grad=True)

    def __call__(self, x: Tensor):
        return x * self.weight + self.bias


class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.requires_grad and p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.requires_grad:
                p.grad = 0.0


def train(model, x, y, optimizer, loss_fn):
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    return loss.data
