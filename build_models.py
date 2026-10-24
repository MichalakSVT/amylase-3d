"""Construit les modèles 3D (GLB + USDZ) du parcours « L'amylase en trois dimensions ».

Usage : python3 build_models.py 3L2L.pdb site/models

Données expérimentales
- 3L2L (Larson et al., 2010) : amylase pancréatique de porc et fragments
  glucidiques observés dans le sillon du site actif.
- 1PPI (Qian et al., 1994) : même enzyme liée à l'acarbose ; fournit la position
  des cycles occupant les sous-sites +1 et +2 (extraits dans data_1ppi.py).

Modèles produits (un par étape du parcours)
- etape1-structure : surface semi-transparente + chaîne principale (trace des
  carbones alpha) colorée du 1er au 496e acide aminé.
- etape2-site : surface ; site actif en vert (chaînes latérales d'Asp197,
  Glu233, Asp300 et des histidines His101, His201, His299, His305).
- etape3-fixation (animé) : un fragment d'amylose de 7 glucoses arrive dans le
  sillon, forme le complexe enzyme-substrat, la liaison -1/+1 est coupée,
  le maltose (+1, +2) et la chaîne raccourcie (-5 à -1) s'éloignent.
  Position des glucoses -5 à -1 : fragment de 3L2L (chaîne C, ramification
  alpha-1,6 retirée) ; glucoses +1 et +2 : gabarit glucose placé sur les cycles
  de 1PPI après superposition des deux enzymes. Approche et départ : trajectoires
  simplifiées (translations), sans valeur de mécanisme.
- etape5-cellulose : amylose et cellulose (beta-1,4) libres, à côté de l'enzyme.
  Cellulose : chaîne régulière construite à partir du beta-glucose de 1PPI
  (vis d'ordre 2, contraintes de liaison C1-O-C4').
Unités : 1 Å -> SCALE mètres ; une seule translation globale (modèle posé sur y = 0).
"""
import json, sys, pathlib, base64
import numpy as np
import gemmi, trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from skimage import measure
import fast_simplification
from data_1ppi import CA_1PPI, RING_1PPI, BGC_1PPI

SRC, OUT = sys.argv[1], pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 0.006                      # 1 Å = 6 mm (agrandissement ~ 60 millions)
PROBE, GRID = 1.4, 0.5
CATALYTIC = (197, 233, 300)        # Asp197 nucléophile, Glu233 acide/base, Asp300
SITE_HIS = (101, 201, 299, 305)    # histidines liées au ligand dans le site (Qian et al., 1994)
SITE_RES = CATALYTIC + SITE_HIS
VDW = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80}
RING = ['C1', 'C2', 'C3', 'C4', 'C5', 'O5']

COL = {
    'protein': (0.416, 0.239, 0.722),    # violet  #6A3DB8
    'site':    (0.106, 0.776, 0.537),    # vert    #1BC689
    'glc_a':   (0.957, 0.651, 0.000),    # amidon  #F4A600
    'glc_b':   (1.000, 0.820, 0.400),    #         #FFD166
    'cel_a':   (0.184, 0.427, 0.710),    # cellulose #2F6DB5
    'cel_b':   (0.353, 0.627, 0.863),    #           #5AA0DC
    'cut':     (0.890, 0.110, 0.180),    # liaison coupée #E31C2E
}
TUBE_FROM, TUBE_TO = np.array([0.169, 0.702, 0.851]), np.array([0.294, 0.122, 0.561])  # cyan -> violet foncé


def kabsch(P, Q):
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    pc, qc = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def apply(R, t, X):
    return (R @ np.asarray(X, float).T).T + t


# ---------------------------------------------------------------- lecture
st = gemmi.read_structure(SRC)
st.remove_alternative_conformations()
st.remove_hydrogens()
m = st[0]
A = m['A']

prot, prot_res = [], []
for r in A:
    if r.het_flag != 'A' and r.name != 'PCA':
        continue
    for a in r:
        prot.append((a.pos.x, a.pos.y, a.pos.z, VDW.get(a.element.name, 1.7)))
        bb = a.name in ('N', 'CA', 'C', 'O')
        prot_res.append(r.seqid.num if (r.seqid.num in SITE_RES and not bb) else -1)
prot, prot_res = np.array(prot), np.array(prot_res)
xyz, rad = prot[:, :3], prot[:, 3]
tree = cKDTree(xyz)
ca = np.array([r['CA'][0].pos.tolist() for r in A if (r.het_flag == 'A' or r.name == 'PCA') and r.find_atom('CA', '*')])
print(f'amylase : {len(prot)} atomes lourds, {len(ca)} acides aminés')

def atom(res, name):
    return np.array(A[str(res)][0][name][0].pos.tolist())

# ---------------------------------------------------------------- surface (SES)
lo = xyz.min(0) - rad.max() - 2 * PROBE - 2
hi = xyz.max(0) + rad.max() + 2 * PROBE + 2
shape = np.ceil((hi - lo) / GRID).astype(int) + 1
gx = [lo[i] + GRID * np.arange(shape[i]) for i in range(3)]
G = np.stack(np.meshgrid(*gx, indexing='ij'), -1).reshape(-1, 3)
sas = np.zeros(len(G), bool)
for i, nb in enumerate(tree.query_ball_point(G, rad.max() + PROBE)):
    if nb:
        sas[i] = np.any(np.linalg.norm(xyz[nb] - G[i], axis=1) < rad[nb] + PROBE)
dist_out = ndimage.distance_transform_edt(sas.reshape(shape), sampling=GRID)
verts, faces, _, _ = measure.marching_cubes(dist_out, level=PROBE, spacing=(GRID,) * 3)
mesh = trimesh.Trimesh(verts + lo, faces, process=True)
mesh = max(mesh.split(only_watertight=False), key=lambda x: len(x.faces))
trimesh.smoothing.filter_taubin(mesh, iterations=10)
v, f = fast_simplification.simplify(mesh.vertices, mesh.faces, target_reduction=0.75)
mesh = trimesh.Trimesh(v, f, process=True)
# marching cubes sur un champ croissant vers l'intérieur : faces orientées vers l'intérieur -> on les retourne
if mesh.volume < 0:
    mesh.invert()
_, idx = tree.query(mesh.triangles_center)
site_face = prot_res[idx] > 0
print('surface : %d triangles, site actif : %d' % (len(mesh.faces), site_face.sum()))

area = mesh.area_faces[site_face]
site_c = np.average(mesh.triangles_center[site_face], axis=0, weights=area)
site_n = np.average(mesh.face_normals[site_face], axis=0, weights=area)
site_n /= np.linalg.norm(site_n)
golden = np.pi * (3 - np.sqrt(5)); k = np.arange(800)
zz = 1 - 2 * (k + 0.5) / 800; rr = np.sqrt(1 - zz ** 2)
dirs = np.stack([rr * np.cos(golden * k), zz, rr * np.sin(golden * k)], 1)
hit = mesh.ray.intersects_any(np.repeat((site_c + site_n)[None], len(dirs), 0), dirs)
open_dir = dirs[~hit].mean(0) if (~hit).any() else site_n
open_dir /= np.linalg.norm(open_dir)

# ---------------------------------------------------------------- substrat (amylose, 7 glucoses)
C = m['C']
def glc(res_id, drop=()):
    return {a.name: np.array(a.pos.tolist()) for a in C[str(res_id)][0] if a.name not in drop}
units_minus = [glc(i, drop=('O1',) if i == 1 else ()) for i in (5, 4, 3, 2, 1)]   # -5 ... -1
R1, t1 = kabsch([CA_1PPI[k] for k in CA_1PPI], [atom(k, 'CA') for k in CA_1PPI])
ca_rmsd = np.sqrt((np.linalg.norm(apply(R1, t1, [CA_1PPI[k] for k in CA_1PPI]) - [atom(k, 'CA') for k in CA_1PPI], axis=1) ** 2).mean())
check = {nm: float(np.linalg.norm(apply(R1, t1, RING_1PPI[nm]) - np.array([glc(ci)[a] for a in RING]), axis=1).mean())
         for nm, ci in (('minus2', 2), ('minus3', 3))}
template = glc(2)                                    # glucose interne de 3L2L (sans O1)
units_plus = []
for nm in ('plus1', 'plus2'):
    target = apply(R1, t1, RING_1PPI[nm])
    R2, t2 = kabsch([template[a] for a in RING], target)
    units_plus.append({a: apply(R2, t2, [p])[0] for a, p in template.items()})
units = units_minus + units_plus                     # ordre : -5 -4 -3 -2 -1 +1 +2
sub_xyz = np.array([p for u in units for p in u.values()])
dsub, _ = tree.query(sub_xyz)
bond_len = float(np.linalg.norm(units_plus[0]['O4'] - units_minus[-1]['C1']))
cut_pos = (units_plus[0]['O4'] + units_minus[-1]['C1']) / 2
print('superposition 1PPI/3L2L : RMSD CA %.2f Å ; cycles -2/-3 : %.2f / %.2f Å' % (ca_rmsd, check['minus2'], check['minus3']))
print('substrat : distance min à l\'amylase %.1f Å ; C1(-1)…O4(+1) = %.2f Å' % (dsub.min(), bond_len))

# ---------------------------------------------------------------- cellulose (beta-1,4, 4 glucoses)
B = {k: np.array(v) for k, v in BGC_1PPI.items()}
def rotm(u, a):
    u = u / np.linalg.norm(u); K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
def screw(p, x):
    th, ph, px, py, pz, rise = p
    u = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    c0 = np.array([px, py, pz])
    return (rotm(u, np.pi) @ (np.asarray(x) - c0).T).T + c0 + rise * u
def resid(p):
    v1 = B['C1'] - B['O1']; v2 = screw(p, B['C4']) - B['O1']
    ang = np.degrees(np.arccos(np.clip(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1)))
    return list((screw(p, B['O4']) - B['O1']) * 3) + [(np.linalg.norm(v2) - 1.43) * 3, (ang - 116) / 10]
rng = np.random.default_rng(0)
cB = np.array([B[a] for a in RING]).mean(0)
best = None
for _ in range(300):
    s = least_squares(resid, np.concatenate([rng.uniform([0, -np.pi], [np.pi, np.pi]), cB + rng.normal(0, 3, 3), [rng.uniform(4, 6)]]))
    if s.cost < 5e-3:
        nxt = np.array([screw(s.x, B[a]) for a in B if a != 'O4']); cur = np.array([B[a] for a in B if a != 'O1'])
        dmin = np.min(np.linalg.norm(nxt[:, None] - cur[None], axis=2))
        if best is None or dmin > best[0]:
            best = (dmin, s.x)
cel_units, u = [], {a: p.copy() for a, p in B.items()}
for i in range(4):
    cel_units.append({a: p for a, p in u.items() if not (i > 0 and a == 'O4')})
    u = {a: screw(best[1], p) for a, p in u.items()}
print('cellulose : montée %.2f Å par glucose, contact min entre glucoses voisins %.2f Å' % (best[1][5], best[0]))

def axis_of(units_):
    X = np.array([p for uu in units_ for p in uu.values()]); c = X.mean(0)
    return c, np.linalg.svd(X - c)[2][0]

# placement des chaînes libres (étape 5) : à côté de l'enzyme, verticales, côte à côte,
# dans le plan vu depuis l'ouverture du site (les deux restent visibles avec le sillon)
up = np.array([0, 1, 0.0])
lateral = np.cross(up, open_dir); lateral /= np.linalg.norm(lateral)
s_c, s_ax = axis_of(units)
c_c, c_ax = axis_of(cel_units)
def align(ax, target):
    ax = ax / np.linalg.norm(ax)
    if abs(ax @ target) > 0.999:
        return np.eye(3) if ax @ target > 0 else rotm(np.cross(target, lateral), np.pi)
    return rotm(np.cross(ax, target), np.arccos(np.clip(ax @ target, -1, 1)))
Ra, Rc = align(s_ax, up), align(c_ax, up)
def place(units_, R, src_c, dst_c):
    return [{a: R @ (p - src_c) + dst_c for a, p in uu.items()} for uu in units_]
prot_c = xyz.mean(0)
ext = np.max((xyz - prot_c) @ lateral)                 # bord de la protéine du côté « lateral »
base = prot_c + up * ((site_c - prot_c) @ up) + open_dir * 6
amy_free = place(units, Ra, s_c, base + lateral * (ext + 12))
cel_free = place(cel_units, Rc, c_c, base + lateral * (ext + 30))
dist = float(tree.query(np.array([p for uu in amy_free + cel_free for p in uu.values()]))[0].min())
print('étape 5 : chaînes libres à côté de l\'enzyme (distance min %.0f Å)' % dist)

# ---------------------------------------------------------------- tube de la chaîne principale
def catmull(P, n=4):
    P = np.vstack([P[0], P, P[-1]]); out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for tt in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = tt * tt, tt * tt * tt
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * tt + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2]); return np.array(out)
def tube(path, radius=0.9, sides=8):
    T = np.gradient(path, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
    nrm = np.cross(T[0], [0, 0, 1.0]); nrm /= np.linalg.norm(nrm); frames = []
    for t in T:
        nrm = nrm - (nrm @ t) * t; nrm /= np.linalg.norm(nrm); frames.append((nrm, np.cross(t, nrm)))
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    V = np.array([p + radius * (np.cos(a) * n1 + np.sin(a) * n2) for p, (n1, n2) in zip(path, frames) for a in ang])
    F = []
    for i in range(len(path) - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            c, d = a + sides, b + sides
            F += [[a, b, c], [b, d, c]]          # faces orientées vers l'extérieur du tube
    return trimesh.Trimesh(V, F, process=False)
NSEG = 8
spl = catmull(ca, 4)
seg_meshes = []
for sgm in range(NSEG):
    i0 = int(len(spl) * sgm / NSEG); i1 = min(len(spl), int(len(spl) * (sgm + 1) / NSEG) + 1)
    seg_meshes.append(tube(spl[i0:i1]))

# ---------------------------------------------------------------- sphères atomiques
_ico = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
def spheres(points, scale=0.85, elem_radius=None):
    out = []
    for name, p in points:
        r = VDW.get(name[0], 1.7) * scale
        s = _ico.copy(); s.vertices = s.vertices * r + p; out.append(s)
    return trimesh.util.concatenate(out)
def chain_meshes(units_, col_a, col_b):
    a = [(n, p) for i, uu in enumerate(units_) if i % 2 == 0 for n, p in uu.items()]
    b = [(n, p) for i, uu in enumerate(units_) if i % 2 == 1 for n, p in uu.items()]
    return [(spheres(a), col_a), (spheres(b), col_b)]

# ---------------------------------------------------------------- repère global
allv = np.vstack([mesh.vertices] + [p for uu in units for p in [np.array(list(uu.values()))]])
shift = -np.array([(allv[:, 0].min() + allv[:, 0].max()) / 2, allv[:, 1].min(), (allv[:, 2].min() + allv[:, 2].max()) / 2])
def tf(p):
    return (np.asarray(p) + shift) * SCALE
def tfm(mm):
    mm = mm.copy(); mm.vertices = tf(mm.vertices); return mm

surf_rest = tfm(mesh.submesh([np.where(~site_face)[0]], append=True))
surf_site = tfm(mesh.submesh([np.where(site_face)[0]], append=True))
surf_all = tfm(mesh)
OFF_IN = open_dir * 30 * SCALE                          # départ du substrat (30 Å devant le site)
# les deux produits s'écartent de part et d'autre de la liaison coupée, le long de la chaîne
_ab = np.array([p for uu in units_plus for p in uu.values()]).mean(0) - np.array([p for uu in units_minus for p in uu.values()]).mean(0)
_ab -= (_ab @ open_dir) * open_dir; _ab /= np.linalg.norm(_ab)
OFF_A = (open_dir * 22 - _ab * 16) * SCALE              # chaîne raccourcie
OFF_B = (open_dir * 22 + _ab * 16) * SCALE              # maltose

# ---------------------------------------------------------------- description des étapes
# chaque partie : (nom, maillage, couleur, opacité)
def step_parts(step):
    P = []
    if step == 'etape1-structure':
        P.append(('surface', surf_all, COL['protein'], 0.22))
        for i, sm in enumerate(seg_meshes):
            col = tuple(TUBE_FROM + (TUBE_TO - TUBE_FROM) * i / (NSEG - 1))
            P.append(('chaine_%d' % (i + 1), tfm(sm), col, 1.0))
    else:
        P.append(('amylase', surf_rest, COL['protein'], 1.0))
        P.append(('site_actif', surf_site, COL['site'], 1.0))
    if step == 'etape3-fixation':
        # géométrie centrée sur le centre du groupe : la translation animée la replace
        for nm, part in (('pieceA', units_minus), ('pieceB', units_plus)):
            for j, (mm, col) in enumerate(chain_meshes(part, COL['glc_a'], COL['glc_b'])):
                mm = tfm(mm); mm.vertices = mm.vertices - GROUP_C[nm]
                P.append((f'{nm}_{j}', mm, col, 1.0))
        mk = _ico.copy(); mk.vertices = mk.vertices * 1.6 * SCALE
        P.append(('liaison', mk, COL['cut'], 1.0))
    if step == 'etape5-cellulose':
        for j, (mm, col) in enumerate(chain_meshes(amy_free, COL['glc_a'], COL['glc_b'])):
            P.append((f'amidon_{j}', tfm(mm), col, 1.0))
        for j, (mm, col) in enumerate(chain_meshes(cel_free, COL['cel_a'], COL['cel_b'])):
            P.append((f'cellulose_{j}', tfm(mm), col, 1.0))
    return P

# animation (étape 3) : groupes et images clés (secondes)
A_c0 = np.array([p for uu in units_minus for p in uu.values()]).mean(0)
B_c0 = np.array([p for uu in units_plus for p in uu.values()]).mean(0)
GROUP_C = {'pieceA': tf(A_c0), 'pieceB': tf(B_c0), 'liaison': tf(cut_pos)}
ANIM_T = [0, 1.5, 3.0, 5.0, 5.4, 5.8, 6.4, 6.6, 8.0, 9.4, 9.6, 9.7, 10.0, 10.5]
def anim_tracks():
    z = np.zeros(3); f = lambda a: [tuple(x) for x in a]
    ease_in = [OFF_IN, OFF_IN * 0.35, z]
    trA = ease_in + [z, z, z, z, z, OFF_A * 0.6, OFF_A, OFF_A, OFF_IN, OFF_IN, OFF_IN]
    trB = ease_in + [z, z, z, z, z, OFF_B * 0.6, OFF_B, OFF_B, OFF_IN, OFF_IN, OFF_IN]
    one, zero = (1, 1, 1), (0, 0, 0)
    scA = [one] * 9 + [one, zero, zero, one, one]
    scM = [zero, zero, zero, zero, (1.3,) * 3, one, one, zero, zero, zero, zero, zero, zero, zero]
    add = lambda tr, g: [tuple(np.asarray(x) + GROUP_C[g]) for x in tr]
    return {'pieceA': (add(trA, 'pieceA'), scA), 'pieceB': (add(trB, 'pieceB'), scA),
            'liaison': (add([z] * len(ANIM_T), 'liaison'), scM)}

# ---------------------------------------------------------------- export GLB
import pygltflib
def material(rgb, name, alpha):
    return trimesh.visual.material.PBRMaterial(name=name, baseColorFactor=list(rgb) + [alpha], metallicFactor=0.0,
                                               roughnessFactor=0.75, alphaMode='BLEND' if alpha < 1 else 'OPAQUE')

def group_of(name):
    return name.split('_')[0] if name.split('_')[0] in ('pieceA', 'pieceB') else name

def export_glb(step, path):
    parts = step_parts(step)
    scene = trimesh.Scene()
    for nm, mm, col, al in parts:
        mm = mm.copy(); mm.visual = trimesh.visual.TextureVisuals(material=material(col, nm, al))
        scene.add_geometry(mm, node_name=nm, geom_name=nm)
    scene.export(path)
    if step != 'etape3-fixation':
        return
    g = pygltflib.GLTF2().load(str(path))
    blob = bytearray(g.binary_blob())
    tracks = anim_tracks()
    def add_acc(arr, typ):
        arr = np.asarray(arr, np.float32); off = len(blob)
        while off % 4: blob.append(0); off += 1
        blob.extend(arr.tobytes())
        g.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=off, byteLength=arr.nbytes))
        acc = pygltflib.Accessor(bufferView=len(g.bufferViews) - 1, componentType=pygltflib.FLOAT, count=len(arr), type=typ)
        if typ == 'SCALAR':
            acc.min, acc.max = [float(arr.min())], [float(arr.max())]
        g.accessors.append(acc); return len(g.accessors) - 1
    t_acc = add_acc(ANIM_T, 'SCALAR')
    anim = pygltflib.Animation(name='fixation-hydrolyse', samplers=[], channels=[])
    for ni, node in enumerate(g.nodes):
        grp = group_of(node.name or '')
        if grp not in tracks:
            continue
        node.matrix = None
        tr, sc = tracks[grp]
        for path_, data in (('translation', tr), ('scale', sc)):
            anim.samplers.append(pygltflib.AnimationSampler(input=t_acc, output=add_acc(data, 'VEC3'), interpolation='LINEAR'))
            anim.channels.append(pygltflib.AnimationChannel(sampler=len(anim.samplers) - 1,
                                                            target=pygltflib.AnimationChannelTarget(node=ni, path=path_)))
    g.animations = [anim]
    g.buffers[0].byteLength = len(blob)
    g.set_binary_blob(bytes(blob))
    g.save_binary(str(path))

# ---------------------------------------------------------------- export USDZ
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt, UsdUtils
FPS = 24
def export_usdz(step, path):
    tmp = path.with_suffix('.usdc')
    stage = Usd.Stage.CreateNew(str(tmp))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/Amylase'); stage.SetDefaultPrim(root.GetPrim())
    animated = step == 'etape3-fixation'
    if animated:
        stage.SetTimeCodesPerSecond(FPS); stage.SetStartTimeCode(0); stage.SetEndTimeCode(ANIM_T[-1] * FPS)
        tracks = anim_tracks(); groups = {}
        for grp, (tr, sc) in tracks.items():
            x = UsdGeom.Xform.Define(stage, f'/Amylase/{grp}')
            top, sop = x.AddTranslateOp(), x.AddScaleOp()
            for t, a, b in zip(ANIM_T, tr, sc):
                top.Set(Gf.Vec3d(*a), t * FPS); sop.Set(Gf.Vec3f(*b), t * FPS)
            groups[grp] = f'/Amylase/{grp}'
    for nm, mm, col, al in step_parts(step):
        parent = '/Amylase'
        if animated and group_of(nm) in ('pieceA', 'pieceB', 'liaison'):
            parent = groups[group_of(nm)]
        mat = UsdShade.Material.Define(stage, f'/Amylase/Mat_{nm}')
        sh = UsdShade.Shader.Define(stage, f'/Amylase/Mat_{nm}/Surface')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*col))
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.75)
        sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
        if al < 1:
            sh.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(float(al))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        gm = UsdGeom.Mesh.Define(stage, f'{parent}/{nm}')
        gm.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mm.vertices.astype(np.float32)))
        gm.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mm.faces)))
        gm.CreateFaceVertexIndicesAttr(Vt.IntArray(mm.faces.flatten().tolist()))
        gm.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(mm.vertex_normals.astype(np.float32)))
        gm.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        gm.CreateSubdivisionSchemeAttr('none')
        UsdShade.MaterialBindingAPI.Apply(gm.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()
    UsdUtils.CreateNewARKitUsdzPackage(Sdf.AssetPath(str(tmp)), str(path))
    tmp.unlink()

STEPS = ['etape1-structure', 'etape2-site', 'etape3-fixation', 'etape5-cellulose']
for stp in STEPS:
    export_glb(stp, OUT / f'{stp}.glb')
    export_usdz(stp, OUT / f'{stp}.usdz')
    # copie intégrée pour l'ouverture de la page depuis un fichier local
    (OUT / f'{stp}.glb.js').write_text('(window.__GLB=window.__GLB||{})["%s"]="%s";' % (stp, base64.b64encode((OUT / f'{stp}.glb').read_bytes()).decode()))
    print(stp, 'GLB %.1f Mo, USDZ %.1f Mo' % ((OUT / f'{stp}.glb').stat().st_size / 1e6, (OUT / f'{stp}.usdz').stat().st_size / 1e6))

# ---------------------------------------------------------------- repères pour les légendes et la caméra
def mv(p):
    return [round(float(x), 4) for x in tf(p)]
def orbit(d):
    return [round(float(np.degrees(np.arctan2(d[0], d[2]))), 1), round(float(np.degrees(np.arccos(np.clip(d[1], -1, 1)))), 1)]
sub_c = np.array([p for uu in units for p in uu.values()]).mean(0)
A_c = np.array([p for uu in units_minus for p in uu.values()]).mean(0)
B_c = np.array([p for uu in units_plus for p in uu.values()]).mean(0)
top = mesh.vertices[mesh.vertices[:, 1].argmax()]
meta = {
    'scale_m_per_angstrom': SCALE,
    'site_target_m': mv(site_c),
    'site_orbit_deg': orbit(open_dir),
    'anim': {'name': 'fixation-hydrolyse', 'bound': 5.0, 'cut_end': 9.4, 'end': ANIM_T[-1]},
    'hotspots': {
        'proteine': mv(top + np.array([0, 2, 0])),
        'debut': mv(ca[0]), 'fin': mv(ca[-1]),
        'site': mv(site_c + open_dir * 2),
        'asp197': mv(atom(197, 'CG')), 'glu233': mv(atom(233, 'CD')), 'asp300': mv(atom(300, 'CG')),
        'substrat': mv(sub_c + open_dir * 4),
        'liaison': mv(cut_pos + open_dir * 3),
        'produitA': mv(A_c + OFF_A / SCALE + open_dir * 3),
        'produitB': mv(B_c + OFF_B / SCALE + open_dir * 3),
        'amidon': mv(np.array([p for uu in amy_free for p in uu.values()])[:, 1].max() * up + np.array([p for uu in amy_free for p in uu.values()]).mean(0) * (1 - up) + up * 7),
        'cellulose': mv(np.array([p for uu in cel_free for p in uu.values()]).min(0) * up + np.array([p for uu in cel_free for p in uu.values()]).mean(0) * (1 - up) - up * 7),
    },
    'controles': {
        'rmsd_CA_1PPI_3L2L_A': round(float(ca_rmsd), 2),
        'ecart_cycles_moins2_moins3_A': [round(check['minus2'], 2), round(check['minus3'], 2)],
        'substrat_distance_min_proteine_A': round(float(dsub.min()), 2),
        'C1_moins1__O4_plus1_A': round(bond_len, 2),
        'cellulose_montee_A': round(float(best[1][5]), 2),
    },
    'atoms': {'amylase': int(len(prot)), 'acides_amines': int(len(ca))},
}
(OUT / 'meta.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False))
print(json.dumps(meta['controles'], indent=1))
(OUT / 'meta.js').write_text('window.__META=' + json.dumps(meta, ensure_ascii=False) + ';')
