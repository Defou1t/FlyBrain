"""flybody (TuragaLab / Google DeepMind, Apache-2.0) MJCF + OBJ meshes -> one glTF for the page.
Nodes: thorax, body (head, mouthparts, antennae, abdomen), eyes, lWing, rWing, legs. Vertex colour =
a shade factor per flybody material (black parts dark, underside lighter), the page's skin colour
multiplies it. Coordinates: glTF x = MJCF y (lateral), y = MJCF z (up), z = MJCF x (forward)."""
import sys, xml.etree.ElementTree as ET
import numpy as np, trimesh, fast_simplification

BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 260_000
root = ET.parse('fruitfly.xml').getroot()
meshfile = {m.get('name'): m.get('file') for m in root.iter('mesh') if m.get('name')}
defaults = {}
def walk_def(el, parent=None):
    for d in el.findall('default'):
        c = d.get('class'); g = d.find('geom')
        defaults[c] = (g.get('material') if g is not None else None, parent)
        walk_def(d, c)
walk_def(root)
def material_of(geom, cls):
    if geom.get('material'):
        return geom.get('material')
    c = geom.get('class') or cls
    while c is not None:
        m, p = defaults.get(c, (None, None))
        if m: return m
        c = p
    return 'body'
SHADE = {'body': 1.0, 'lower': 0.92, 'black': 0.22, 'brown': 0.38, 'ocelli': 0.3, 'bristle-brown': 0.22, 'red': 1.0, 'membrane': 1.0}
def quat_to_mat(q):   # MuJoCo (w, x, y, z)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def pose(el):
    p = np.array([float(v) for v in (el.get('pos') or '0 0 0').split()])
    q = [float(v) for v in (el.get('quat') or '1 0 0 0').split()]
    T = np.eye(4); T[:3, :3] = quat_to_mat(q); T[:3, 3] = p; return T

def group_of(body_name):
    n = body_name
    if n == 'thorax' or n.startswith('haltere'): return 'thorax'
    if n.startswith('wing_left'): return 'wing_L'
    if n.startswith('wing_right'): return 'wing_R'
    if any(n.startswith(k) for k in ('coxa', 'femur', 'tibia', 'tarsus', 'claw')): return 'legs'
    return 'body'

parts = {}   # group -> list of (vertices, faces, shade)
def walk(body, T, cls):
    cls = body.get('childclass') or cls
    T = T @ pose(body)
    grp = group_of(body.get('name'))
    for g in body.findall('geom'):
        m = g.get('mesh')
        if not m or m not in meshfile: continue
        mat = material_of(g, cls)
        if mat in ('blue', 'pink'): continue          # collision helpers
        mesh = trimesh.load(meshfile[m], force='mesh', process=True); mesh.merge_vertices()   # the OBJs are unwelded (one component per triangle)
        v = mesh.vertices * 0.1
        Tg = T @ pose(g)
        v = (Tg[:3, :3] @ v.T).T + Tg[:3, 3]
        key = 'eyes' if mat == 'red' else grp
        parts.setdefault(key, []).append((v.astype(np.float64), mesh.faces.astype(np.int64), SHADE.get(mat, 1.0), mat))
    for c in body.findall('body'):
        walk(c, T, cls)
walk(root.find('worldbody').find('body'), np.eye(4), None)

total = sum(len(f) for ps in parts.values() for _, f, _, _ in ps)
ratio = min(1.0, BUDGET / total)
print(f'{total:,} triangles in, target {BUDGET:,} (keep {ratio:.2f})')
scene = trimesh.Scene()
NAMES = {'thorax': 'thorax', 'body': 'body', 'eyes': 'eyes', 'legs': 'legs'}
for key, ps in parts.items():
    vs, fs, cs, off = [], [], [], 0
    for v, f, shade, mat in ps:
        if len(f) > 300 and ratio < 1.0:
            keep = ratio * (0.5 if mat in ('black', 'brown', 'lower', 'red') else 1.0)
            v2, f2 = fast_simplification.simplify(v, f, target_reduction=1 - max(0.03, keep))
            v, f = v2, f2.astype(np.int64)
        vs.append(v); fs.append(f + off); off += len(v)
        c = np.full((len(v), 4), 255, np.uint8); c[:, :3] = int(shade * 255); cs.append(c)
    V = np.concatenate(vs); F = np.concatenate(fs); C = np.concatenate(cs)
    V = np.stack([V[:, 1], V[:, 2], V[:, 0]], axis=1)          # MJCF (x fwd, y left, z up) -> glTF (x, y up, z fwd)
    m = trimesh.Trimesh(V, F, vertex_colors=C, process=False)
    if key.startswith('wing'):
        name = 'lWing' if m.bounds.mean(axis=0)[0] < 0 else 'rWing'
    else:
        name = NAMES[key]
    m.metadata['name'] = name
    scene.add_geometry(m, node_name=name, geom_name=name)
    print(f'  {name:7s} {len(F):8,} tris  shade groups {sorted(set(round(s, 2) for _, _, s, _ in ps))}')
# the two wings are posed differently in the MJCF (one is flipped); use one wing mirrored for both so the page's
# fold / flap animation treats them symmetrically
wings = {n: g for n, g in scene.geometry.items() if n.endswith('Wing')}
src = wings['rWing']
V = src.vertices.copy(); V[:, 0] *= -1; F = src.faces[:, ::-1].copy()
mirror = trimesh.Trimesh(V, F, vertex_colors=src.visual.vertex_colors.copy(), process=False)
scene.delete_geometry('lWing'); scene.add_geometry(mirror, node_name='lWing', geom_name='lWing')
scene.export('drosophila-flybody.glb', include_normals=False)
import os; print('written', os.path.getsize('drosophila-flybody.glb') // 1024, 'KB')
