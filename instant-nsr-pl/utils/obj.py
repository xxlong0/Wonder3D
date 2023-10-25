import numpy as np


def load_obj(filename):
    # Read entire file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # load vertices
    vertices, texcoords  = [], []
    for line in lines:
        if len(line.split()) == 0:
            continue
        
        prefix = line.split()[0].lower()
        if prefix == 'v':
            vertices.append([float(v) for v in line.split()[1:]])
        elif prefix == 'vt':
            val = [float(v) for v in line.split()[1:]]
            texcoords.append([val[0], 1.0 - val[1]])
    
    uv = len(texcoords) > 0
    faces, tfaces = [], []
    for line in lines:
        if len(line.split()) == 0:
            continue
        prefix = line.split()[0].lower()
        if prefix == 'usemtl': # Track used materials
            pass
        elif prefix == 'f': # Parse face
            vs = line.split()[1:]
            nv = len(vs)
            vv = vs[0].split('/')
            v0 = int(vv[0]) - 1
            if uv:
                t0 = int(vv[1]) - 1 if vv[1] != "" else -1
            for i in range(nv - 2): # Triangulate polygons
                vv1 = vs[i + 1].split('/')
                v1 = int(vv1[0]) - 1
                vv2 = vs[i + 2].split('/')
                v2 = int(vv2[0]) - 1
                faces.append([v0, v1, v2])
                if uv:
                    t1 = int(vv1[1]) - 1 if vv1[1] != "" else -1
                    t2 = int(vv2[1]) - 1 if vv2[1] != "" else -1
                    tfaces.append([t0, t1, t2])
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int64)
    if uv:
        assert len(tfaces) == len(faces)
        texcoords = np.array(texcoords, dtype=np.float32)
        tfaces = np.array(tfaces, dtype=np.int64)
    else:
        texcoords, tfaces = None, None 

    return vertices, faces, texcoords, tfaces


def write_obj(filename, v_pos, t_pos_idx, v_tex, t_tex_idx):
    with open(filename, "w") as f:
        for v in v_pos:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))
       
        if v_tex is not None:
            assert(len(t_pos_idx) == len(t_tex_idx))
            for v in v_tex:
                f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        # Write faces
        for i in range(len(t_pos_idx)):
            f.write("f ")
            for j in range(3):
                f.write(' %s/%s' % (str(t_pos_idx[i][j]+1), '' if v_tex is None else str(t_tex_idx[i][j]+1)))
            f.write("\n")
