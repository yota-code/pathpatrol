import sympy

ax, ay, bx, by, cx, cy, dx, dy = sympy.symbols('a_x a_y b_x b_y c_x c_y d_x d_y')

k1, k2 = sympy.symbols('k_1 k_2')

x1 = (bx - ax) * k1 + ax
y1 = (by - ay) * k1 + ay

x2 = (dx - cx) * k2 + cx
y2 = (dy - cy) * k2 + cy


res = sympy.solve([x1 - x2, y1 - y2], [k1, k2])
print(res)
# équation cartésienne : (ay - by) * ax + (bx - ax) * ay + (a_x*b_y - a_y*b_x) = 0
# intersection à : {x: (B_1*C_2 - B_2*C_1)/(A_1*B_2 - A_2*B_1), y: (-A_1*C_2 + A_2*C_1)/(A_1*B_2 - A_2*B_1)}