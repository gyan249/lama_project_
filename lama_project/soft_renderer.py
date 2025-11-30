import os
import pygame, sys, math
import numpy as np
from pygame.locals import *
from rasterizer import draw_triangle
from obj_loader import load_obj

WIDTH, HEIGHT = 960, 640

# ----- transform matrices -----
def rot_x(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=np.float32)

def rot_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=np.float32)

def translate(tx,ty,tz):
    T = np.eye(4, dtype=np.float32)
    T[0,3]=tx; T[1,3]=ty; T[2,3]=tz
    return T

def scale_matrix(s):
    S = np.eye(4, dtype=np.float32)
    S[0,0]=S[1,1]=S[2,2]=s
    return S

def perspective(fov, aspect, znear, zfar):
    f = 1.0/math.tan(math.radians(fov)/2)
    P = np.zeros((4,4), dtype=np.float32)
    P[0,0]=f/aspect; P[1,1]=f
    P[2,2]=(zfar+znear)/(znear-zfar)
    P[2,3]=(2*zfar*znear)/(znear-zfar)
    P[3,2] = -1.0
    return P

# ----------------- fallback cube -----------------
cube_verts = np.array([
    [-1,-1,-1,1],[1,-1,-1,1],[1,1,-1,1],[-1,1,-1,1],
    [-1,-1,1,1],[1,-1,1,1],[1,1,1,1],[-1,1,1,1]
], dtype=np.float32)

triangles_cube = [
    (0,1,2),(0,2,3),
    (4,6,5),(4,7,6),
    (0,4,5),(0,5,1),
    (3,2,6),(3,6,7),
    (1,5,6),(1,6,2),
    (0,3,7),(0,7,4)
]

# ----------------- normalize model -----------------
def normalize_model(vertices):
    pts = np.array([v[:3] for v in vertices], dtype=np.float32)
    mn = pts.min(axis=0); mx = pts.max(axis=0)
    center = (mn + mx)/2.0
    pts = pts - center
    extent = (mx-mn).max()

    scale = 1.0/extent if extent>0 else 1.0
    verts = [np.array([*(p*scale),1.0], dtype=np.float32) for p in pts]
    return verts

# ----------------- load OBJ or cube -----------------
def load_model():
    if not os.path.isdir("models"):
        print("models/ folder missing → using cube")
        return cube_verts, triangles_cube

    obj_files = [f for f in os.listdir("models") if f.lower().endswith(".obj")]

    if not obj_files:
        print("No OBJ found → using cube")
        return cube_verts, triangles_cube

    path = os.path.join("models", obj_files[0])
    print(f"Loading OBJ: {path}")

    verts, faces = load_obj(path)

    print(f"Loaded: {len(verts)} vertices, {len(faces)} faces")

    if len(verts)==0 or len(faces)==0:
        print("OBJ empty → using cube")
        return cube_verts, triangles_cube

    verts = normalize_model(verts)
    return np.array(verts, dtype=np.float32), faces


# ========================= MAIN ===============================

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH,HEIGHT))
    clock = pygame.time.Clock()

    vertices, triangles = load_model()

    P = perspective(60, WIDTH/HEIGHT, 0.1, 100)
    angle = 0

    zbuffer = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float32)
    base_color = np.array([160,200,255], dtype=np.float32)

    while True:
        for ev in pygame.event.get():
            if ev.type == QUIT:
                pygame.quit()
                sys.exit()

        angle += 0.01

        # model → rotate → push forward
        S = scale_matrix(1.8)
        R = rot_y(angle) @ rot_x(angle*0.6)
        T = translate(0,0,-3.0)

        M = T @ (R @ S)
        MVP = P @ M

        screen.fill((10,10,20))
        zbuffer[:] = np.inf

        projected = []
        cam_positions = []

        for v in vertices:
            p = M @ v
            cam_positions.append(p[:3])

            clip = MVP @ v
            if clip[3] <= 1e-6:
                projected.append(None); continue

            ndc = clip[:3] / clip[3]
            x = int((ndc[0]*0.5+0.5)*WIDTH)
            y = int((1-(ndc[1]*0.5+0.5))*HEIGHT)
            z = ndc[2]
            projected.append((x,y,z))

        visible = 0
        for a,b,c in triangles:
            A=projected[a]; B=projected[b]; C=projected[c]
            if A is None or B is None or C is None:
                continue

            v0=cam_positions[a]; v1=cam_positions[b]; v2=cam_positions[c]
            normal=np.cross(v1-v0, v2-v0)
            nl=np.linalg.norm(normal)
            if nl==0: continue
            normal/=nl

            center=(v0+v1+v2)/3.0
            if np.dot(normal, center)>=0:  # correct facing
                continue

            lambert=max(0, -np.dot(normal, center)/(np.linalg.norm(center)+1e-9))
            brightness=0.25+0.75*lambert
            color=tuple((base_color*brightness).astype(np.uint8))

            draw_triangle(screen, zbuffer, (A,B,C), color)
            visible+=1

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
