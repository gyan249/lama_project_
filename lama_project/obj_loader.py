import numpy as np

def load_obj(path):
    vertices = []
    faces = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append(np.array([float(x), float(y), float(z), 1.0]))

            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                # collect vertex indices (ignore vt/vn)
                indices = []
                for p in parts:
                    v = p.split("/")[0]
                    indices.append(int(v) - 1)

                # TRIANGULATE:
                # any face with N vertices becomes N-2 triangles
                if len(indices) >= 3:
                    for i in range(1, len(indices) - 1):
                        faces.append((indices[0], indices[i], indices[i + 1]))

    return vertices, faces
