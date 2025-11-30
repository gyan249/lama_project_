import numpy as np
import pygame

def edge_function(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - \
           (c[1] - a[1]) * (b[0] - a[0])

def draw_triangle(screen, zbuffer, verts, color):
    A, B, C = verts

    min_x = max(int(min(A[0], B[0], C[0])), 0)
    max_x = min(int(max(A[0], B[0], C[0])), screen.get_width() - 1)

    min_y = max(int(min(A[1], B[1], C[1])), 0)
    max_y = min(int(max(A[1], B[1], C[1])), screen.get_height() - 1)

    area = edge_function(A, B, C)
    if area == 0:
        return

    for y in range(min_y, max_y + 1):
        row = zbuffer[y]
        for x in range(min_x, max_x + 1):
            P = (x + 0.5, y + 0.5)
            w0 = edge_function(B, C, P)
            w1 = edge_function(C, A, P)
            w2 = edge_function(A, B, P)

            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or \
               (w0 <= 0 and w1 <= 0 and w2 <= 0):

                alpha = w0 / area
                beta  = w1 / area
                gamma = w2 / area

                z = alpha*A[2] + beta*B[2] + gamma*C[2]

                if z < row[x]:
                    row[x] = z
                    screen.set_at((x,y), color)
