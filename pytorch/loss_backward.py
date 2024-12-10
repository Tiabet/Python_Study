import torch
from torchviz import make_dot

# 간단한 예제를 위해 랜덤한 텐서 정의
x = torch.randn(3, requires_grad=True)  # 입력 텐서
w = torch.randn(3, requires_grad=True)  # 가중치 텐서
b = torch.randn(1, requires_grad=True)  # 편향 텐서

# Forward 연산: y = w * x + b
y = w * x + b
# Loss 함수 예제: sum 연산을 사용하여 스칼라 loss로 만듦
loss = y.sum()

print("===== Before backward =====")
print("x.grad_fn:", x.grad_fn)  # x는 leaf tensor라 grad_fn은 None일 수 있음
print("w.grad_fn:", w.grad_fn)
print("b.grad_fn:", b.grad_fn)
print("y.grad_fn:", y.grad_fn)
print("loss.grad_fn:", loss.grad_fn)

# torchviz를 이용한 연산 그래프 시각화
dot = make_dot(loss, params={'x': x, 'w': w, 'b': b})
dot.render("graph", format="png")
print("연산 그래프가 graph.png로 렌더링되었습니다.")

# Backward 호출
loss.backward()

print("\n===== After backward =====")
print("x.grad:", x.grad)
print("w.grad:", w.grad)
print("b.grad:", b.grad)
