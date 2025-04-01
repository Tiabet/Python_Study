import numpy as np
import matplotlib.pyplot as plt

# 1. 목적함수 정의
def f(x1, x2):
    return (x1 - 3)**2 + (x2 - 3)**2

# 2. (x1, x2)에 대한 격자 생성
x1_vals = np.linspace(-1, 6, 300)
x2_vals = np.linspace(-1, 6, 300)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = f(X1, X2)

# 3. 등고선(Contour) 플롯
plt.figure(figsize=(8,6))
contours = plt.contour(X1, X2, Z, levels=20, cmap='jet')
plt.clabel(contours, inline=True, fontsize=8)
plt.title("Objective Function Contours and Constraints")

# 4. 제약식 표현
# (1) x1 + x2 = 4 -> x2 = 4 - x1
x2_line1 = 4 - x1_vals
plt.plot(x1_vals, x2_line1, 'r--', label='x1 + x2 = 4')

# (2) x1 - 3x2 = 1 -> x2 = (x1 - 1)/3
x2_line2 = (x1_vals - 1)/3
plt.plot(x1_vals, x2_line2, 'g--', label='x1 - 3x2 = 1')

# 5. x1 + x2 <= 4 영역을 약간 음영 처리 (optional)
#   - 예시로 x1_vals 중 실제로 x2_line1이 화면 안에 있는 구간만 채웁니다.
plt.fill_between(x1_vals, x2_line1, 6,
                 where=(x2_line1<=6), color='red', alpha=0.1)

# 6. 그래프 설정
plt.xlim(-1, 6)
plt.ylim(-1, 6)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()
