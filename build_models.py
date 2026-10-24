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

# ---------------------------------------------------------------- visite guidée : une seule animation, 5 étapes
# Segments (secondes, vitesse 1) : chaque étape occupe un intervalle de la frise.
SEG = {1: (0.0, 12.0), 2: (12.0, 22.0), 3: (22.0, 34.0), 4: (34.0, 46.0), 5: (46.0, 58.0)}
END = 58.0
T_BOUND, T_CUT0, T_CUT1, T_GONE = 27.0, 34.0, 35.8, 39.0     # fixation terminée, marqueur de coupure, produits partis
CAPTIONS = {
    1: ("1/5 · Structure", "L’amylase est une protéine : une chaîne de 496 acides aminés repliée en une forme 3D précise."),
    2: ("2/5 · Site actif (vert)", "Il est formé d’acides aminés éloignés dans la séquence mais rapprochés par le repliement."),
    3: ("3/5 · Fixation du substrat", "L’amidon (substrat) se fixe dans le site actif, de forme complémentaire : c’est le complexe enzyme-substrat."),
    4: ("4/5 · Catalyse : hydrolyse", "Une liaison entre deux glucoses est coupée : maltose et chaîne raccourcie sont libérés, l’enzyme est intacte."),
    5: ("5/5 · Et la cellulose ?", "Amidon (liaisons α) et cellulose (liaisons β) : la cellulose peut-elle se fixer dans ce site actif ?"),
}
EPS = 0.02

def vis_keys(intervals):
    """Visibilité (échelle 0 ou 1) avec bascules quasi instantanées."""
    keys = [(0.0, 0.0)]
    for a, b in intervals:
        keys += [(max(a - EPS, 0.0), 0.0), (a, 1.0), (b - EPS, 1.0), (b, 0.0)]
    keys.append((END, 0.0))
    d = {}
    for t, v in keys:                       # à temps égal, la dernière valeur l'emporte
        d[round(t, 4)] = v
    return sorted(d.items())

# panneaux de texte (pour la réalité augmentée des applications natives)
from PIL import Image, ImageDraw, ImageFont
FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_R = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def wrap(draw, text, font, width):
    lines, cur = [], ''
    for w in text.split():
        t = (cur + ' ' + w).strip()
        if draw.textlength(t, font=font) <= width: cur = t
        else: lines.append(cur); cur = w
    return lines + [cur]
def panel_image(k):
    W, H = 1600, 420
    img = Image.new('RGB', (W, H), (255, 255, 255)); d = ImageDraw.Draw(img)
    d.rounded_rectangle((6, 6, W - 7, H - 7), radius=36, outline=(106, 61, 184), width=12)
    fb, fr = ImageFont.truetype(FONT_B, 66), ImageFont.truetype(FONT_R, 58)
    title, text = CAPTIONS[k]
    d.text((60, 44), title, font=fb, fill=(90, 51, 158))
    y = 140
    for line in wrap(d, text, fr, W - 120)[:3]:
        d.text((60, y), line, font=fr, fill=(31, 27, 43)); y += 78
    return img
top_y = max(float(surf_all.vertices[:, 1].max()), 0)
PW, PH = 0.42, 0.42 * 420 / 1600
def panel_mesh(back=False):
    v = np.array([[-PW / 2, 0, 0], [PW / 2, 0, 0], [PW / 2, PH, 0], [-PW / 2, PH, 0]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    if back:   # face arrière : même texte, lisible de l'autre côté
        v[:, 0] *= -1; v[:, 2] -= 0.002
    v[:, 1] += top_y + 0.03
    return trimesh.Trimesh(v, f, process=False), uv

# toutes les parties de la visite : (nom, maillage, couleur, opacité, texture, groupe)
PARTS = [('surface_transparente', surf_all, COL['protein'], 0.22, None, 'trans')]
for i, sm in enumerate(seg_meshes):
    PARTS.append(('chaine_%d' % (i + 1), tfm(sm), tuple(TUBE_FROM + (TUBE_TO - TUBE_FROM) * i / (NSEG - 1)), 1.0, None, 'tube'))
PARTS += [('amylase', surf_rest, COL['protein'], 1.0, None, 'opaque'), ('site_actif', surf_site, COL['site'], 1.0, None, 'opaque')]
A_c0 = np.array([p for uu in units_minus for p in uu.values()]).mean(0)
B_c0 = np.array([p for uu in units_plus for p in uu.values()]).mean(0)
GROUP_C = {'pieceA': tf(A_c0), 'pieceB': tf(B_c0), 'liaison': tf(cut_pos)}
for nm, part in (('pieceA', units_minus), ('pieceB', units_plus)):
    for j, (mm, col) in enumerate(chain_meshes(part, COL['glc_a'], COL['glc_b'])):
        mm = tfm(mm); mm.vertices = mm.vertices - GROUP_C[nm]
        PARTS.append((f'{nm}_{j}', mm, col, 1.0, None, nm))
mk = _ico.copy(); mk.vertices = mk.vertices * 1.6 * SCALE
PARTS.append(('liaison', mk, COL['cut'], 1.0, None, 'liaison'))
for j, (mm, col) in enumerate(chain_meshes(amy_free, COL['glc_a'], COL['glc_b'])):
    PARTS.append((f'amidon_{j}', tfm(mm), col, 1.0, None, 'libres'))
for j, (mm, col) in enumerate(chain_meshes(cel_free, COL['cel_a'], COL['cel_b'])):
    PARTS.append((f'cellulose_{j}', tfm(mm), col, 1.0, None, 'libres'))
for k in range(1, 6):
    img = panel_image(k); img.save(OUT / f'panneau_{k}.png')
    for side in (0, 1):
        pm, uv = panel_mesh(back=bool(side))
        PARTS.append((f'panneau_{k}_{side}', pm, (1, 1, 1), 1.0, (uv, img, f'panneau_{k}.png'), f'panneau_{k}'))

def tracks_for(group):
    """(clés de translation, clés d'échelle) pour un groupe ; translation None = fixe."""
    S = SEG
    if group == 'trans' or group == 'tube':
        return None, vis_keys([S[1]])
    if group == 'opaque':
        return None, vis_keys([(S[2][0], END)])
    if group == 'libres':
        return None, vis_keys([S[5]])
    if group.startswith('panneau_'):
        return None, vis_keys([S[int(group[-1])]])
    if group in ('pieceA', 'pieceB'):
        off = OFF_A if group == 'pieceA' else OFF_B
        c = GROUP_C[group]
        tr = [(0.0, OFF_IN), (S[3][0], OFF_IN), (S[3][0] + 2.5, OFF_IN * 0.35), (T_BOUND, 0 * OFF_IN), (T_CUT1, 0 * OFF_IN),
              (T_CUT1 + 1.7, off * 0.6), (T_GONE, off), (END, off)]
        return [(t, tuple(np.asarray(v) + c)) for t, v in tr], vis_keys([(S[3][0], S[4][1])])
    if group == 'liaison':
        sc = [(0.0, 0.0), (T_CUT0, 0.0), (T_CUT0 + 0.4, 1.3), (T_CUT0 + 0.8, 1.0), (T_CUT1 - 0.2, 1.0), (T_CUT1, 0.0), (END, 0.0)]
        return [(0.0, tuple(GROUP_C['liaison'])), (END, tuple(GROUP_C['liaison']))], sc
    raise ValueError(group)

# ---------------------------------------------------------------- export GLB animé
import pygltflib
def material(rgb, name, alpha, tex=None):
    kw = dict(name=name, baseColorFactor=list(rgb) + [alpha], metallicFactor=0.0, roughnessFactor=0.75,
              alphaMode='BLEND' if alpha < 1 else 'OPAQUE')
    if tex is not None:
        kw['baseColorTexture'] = tex
        kw['emissiveFactor'] = [0.0, 0.0, 0.0]
    return trimesh.visual.material.PBRMaterial(**kw)

def export_glb(path):
    scene = trimesh.Scene()
    for nm, mm, col, al, tex, grp in PARTS:
        mm = mm.copy()
        if tex is not None:
            uv, img, _ = tex
            mm.visual = trimesh.visual.TextureVisuals(uv=uv, material=material(col, nm, al, img))
        else:
            mm.visual = trimesh.visual.TextureVisuals(material=material(col, nm, al))
        scene.add_geometry(mm, node_name=nm, geom_name=nm)
    scene.export(path)
    g = pygltflib.GLTF2().load(str(path))
    blob = bytearray(g.binary_blob())
    def add_acc(arr, typ):
        arr = np.asarray(arr, np.float32); off = len(blob)
        while off % 4: blob.append(0); off += 1
        blob.extend(arr.tobytes())
        g.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=off, byteLength=arr.nbytes))
        acc = pygltflib.Accessor(bufferView=len(g.bufferViews) - 1, componentType=pygltflib.FLOAT, count=len(arr), type=typ)
        if typ == 'SCALAR':
            acc.min, acc.max = [float(arr.min())], [float(arr.max())]
        g.accessors.append(acc); return len(g.accessors) - 1
    grp_of = {p[0]: p[5] for p in PARTS}
    anim = pygltflib.Animation(name='visite', samplers=[], channels=[])
    for ni, node in enumerate(g.nodes):
        grp = grp_of.get(node.name or '')
        if grp is None:
            continue
        node.matrix = None
        tr, sc = tracks_for(grp)
        chans = [('scale', [t for t, _ in sc], [(v, v, v) for _, v in sc])]
        if tr is not None:
            chans.append(('translation', [t for t, _ in tr], [v for _, v in tr]))
        for path_, times, vals in chans:
            anim.samplers.append(pygltflib.AnimationSampler(input=add_acc(times, 'SCALAR'), output=add_acc(vals, 'VEC3'), interpolation='LINEAR'))
            anim.channels.append(pygltflib.AnimationChannel(sampler=len(anim.samplers) - 1, target=pygltflib.AnimationChannelTarget(node=ni, path=path_)))
    g.animations = [anim]
    g.buffers[0].byteLength = len(blob)
    g.set_binary_blob(bytes(blob))
    g.save_binary(str(path))

# ---------------------------------------------------------------- export USDZ animé (Quick Look)
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt, UsdUtils
FPS = 24
def export_usdz(path):
    tmp = path.with_suffix('.usdc')
    stage = Usd.Stage.CreateNew(str(tmp))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/Amylase'); stage.SetDefaultPrim(root.GetPrim())
    stage.SetTimeCodesPerSecond(FPS); stage.SetStartTimeCode(0); stage.SetEndTimeCode(END * FPS)
    for nm, mm, col, al, tex, grp in PARTS:
        x = UsdGeom.Xform.Define(stage, f'/Amylase/{nm}_x')
        tr, sc = tracks_for(grp)
        if tr is not None:
            top = x.AddTranslateOp()
            for t, v in tr: top.Set(Gf.Vec3d(*v), t * FPS)
        sop = x.AddScaleOp()
        for t, v in sc: sop.Set(Gf.Vec3f(v, v, v), t * FPS)
        mat = UsdShade.Material.Define(stage, f'/Amylase/Mat_{nm}')
        sh = UsdShade.Shader.Define(stage, f'/Amylase/Mat_{nm}/Surface')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.75)
        sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
        if tex is not None:
            uvr = UsdShade.Shader.Define(stage, f'/Amylase/Mat_{nm}/uv')
            uvr.CreateIdAttr('UsdPrimvarReader_float2')
            uvr.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
            tx = UsdShade.Shader.Define(stage, f'/Amylase/Mat_{nm}/tex')
            tx.CreateIdAttr('UsdUVTexture')
            tx.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(tex[2])
            tx.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uvr.ConnectableAPI(), 'result')
            sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), 'rgb')
        else:
            sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*col))
        if al < 1:
            sh.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(float(al))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        gm = UsdGeom.Mesh.Define(stage, f'/Amylase/{nm}_x/{nm}')
        gm.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mm.vertices.astype(np.float32)))
        gm.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mm.faces)))
        gm.CreateFaceVertexIndicesAttr(Vt.IntArray(mm.faces.flatten().tolist()))
        gm.CreateSubdivisionSchemeAttr('none')
        if tex is not None:
            pv = UsdGeom.PrimvarsAPI(gm).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec2fArray.FromNumpy(tex[0].astype(np.float32)))
            gm.CreateDoubleSidedAttr(False)
        else:
            gm.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(mm.vertex_normals.astype(np.float32)))
            gm.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        UsdShade.MaterialBindingAPI.Apply(gm.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()
    UsdUtils.CreateNewARKitUsdzPackage(Sdf.AssetPath(str(tmp)), str(path))
    tmp.unlink()

export_glb(OUT.resolve() / 'visite.glb')
export_usdz(OUT.resolve() / 'visite.usdz')        # textures des panneaux : chemins relatifs au dossier des modèles
(OUT / 'visite.glb.js').write_text('(window.__GLB=window.__GLB||{})["visite"]="%s";' % base64.b64encode((OUT / 'visite.glb').read_bytes()).decode())
print('visite GLB %.1f Mo, USDZ %.1f Mo' % ((OUT / 'visite.glb').stat().st_size / 1e6, (OUT / 'visite.usdz').stat().st_size / 1e6))

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
    'anim': {'name': 'visite', 'end': END, 'segments': {str(k): list(v) for k, v in SEG.items()},
             'bound': T_BOUND, 'cut': [T_CUT0, T_CUT1], 'gone': T_GONE},
    'captions': {str(k): list(v) for k, v in CAPTIONS.items()},
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
        'cellulose_montee_A': round(abs(float(best[1][5])), 2),
    },
    'atoms': {'amylase': int(len(prot)), 'acides_amines': int(len(ca))},
}
(OUT / 'meta.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False))
print(json.dumps(meta['controles'], indent=1))
(OUT / 'meta.js').write_text('window.__META=' + json.dumps(meta, ensure_ascii=False) + ';')
