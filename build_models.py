"""Construit les modèles 3D (GLB + USDZ) de l'amylase pancréatique porcine
à partir des coordonnées expérimentales PDB 3L2L.

Usage : python3 build_models.py 3L2L.pdb site/models

Choix de représentation
- Amylase (chaîne A, polymère) : surface moléculaire exclue du solvant (SES,
  sonde 1,4 Å) calculée sur les atomes lourds, conformation alternative A.
- Fragments glucidiques (chaînes B, C et GLC A801) : sphères atomiques
  (rayons de van der Waals, atomes lourds).
- Site actif : partie de la surface la plus proche des chaînes latérales des
  trois résidus catalytiques Asp197, Glu233, Asp300 (numérotation de la
  protéine mature ; UniProt P00690 : 212, 248, 315) et des histidines du
  site His101, His201, His299, His305 (Qian et al., 1994).
- Eau, ions Ca2+ et Cl- : non représentés.
Unités : 1 Å -> SCALE mètres. Positions relatives des atomes conservées
(seule une translation globale est appliquée : modèle posé sur y = 0).
"""
import json, sys, pathlib
import numpy as np
import gemmi, trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure
import fast_simplification

SRC, OUT = sys.argv[1], pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 0.006          # 1 Å = 6 mm  -> agrandissement d'environ 60 millions
PROBE, GRID = 1.4, 0.5
CATALYTIC = (197, 233, 300)          # Asp197 nucléophile, Glu233 acide/base, Asp300
SITE_HIS = (101, 201, 299, 305)      # histidines liées à l'inhibiteur dans le site (Qian et al., 1994)
SITE_RES = CATALYTIC + SITE_HIS
VDW = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80}

COL = {
    'protein': (0.416, 0.239, 0.722),   # violet  #6A3DB8
    'sugar':   (0.957, 0.651, 0.000),   # ambre   #F4A600
    'site':    (0.106, 0.776, 0.537),   # vert    #1BC689
}

st = gemmi.read_structure(SRC)
st.remove_alternative_conformations()   # conserve la conformation A
st.remove_hydrogens()
m = st[0]

prot, prot_res, sugar = [], [], []
for ch in m:
    for r in ch:
        if r.name == 'HOH' or r.name in ('CA', 'CL'):
            continue
        for a in r:
            p = (a.pos.x, a.pos.y, a.pos.z, VDW.get(a.element.name, 1.7))
            if r.name == 'GLC':
                sugar.append(p)
            elif r.het_flag == 'A' or r.name == 'PCA':
                prot.append(p)
                bb = a.name in ('N', 'CA', 'C', 'O')
                prot_res.append(r.seqid.num if (r.seqid.num in SITE_RES and not bb) else -1)
prot, sugar, prot_res = np.array(prot), np.array(sugar), np.array(prot_res)
print(f'atomes lourds : amylase {len(prot)}, glucides {len(sugar)}')

# ---- Surface exclue du solvant (SES) par champ de distance ----
xyz, rad = prot[:, :3], prot[:, 3]
lo = xyz.min(0) - rad.max() - 2 * PROBE - 2
hi = xyz.max(0) + rad.max() + 2 * PROBE + 2
shape = np.ceil((hi - lo) / GRID).astype(int) + 1
gx = [lo[i] + GRID * np.arange(shape[i]) for i in range(3)]
G = np.stack(np.meshgrid(*gx, indexing='ij'), -1).reshape(-1, 3)
tree = cKDTree(xyz)
sas = np.zeros(len(G), bool)
for i, nb in enumerate(tree.query_ball_point(G, rad.max() + PROBE)):
    if nb:
        d = np.linalg.norm(xyz[nb] - G[i], axis=1)
        sas[i] = np.any(d < rad[nb] + PROBE)
sas = sas.reshape(shape)
# distance (Å) de chaque point intérieur au solvant accessible au centre de la sonde
dist_out = ndimage.distance_transform_edt(sas, sampling=GRID)
verts, faces, _, _ = measure.marching_cubes(dist_out, level=PROBE, spacing=(GRID,) * 3)
verts += lo
mesh = trimesh.Trimesh(verts, faces, process=True)
mesh = max(mesh.split(only_watertight=False), key=lambda x: len(x.faces))  # supprime les cavités internes
trimesh.smoothing.filter_taubin(mesh, iterations=10)
v, f = fast_simplification.simplify(mesh.vertices, mesh.faces, target_reduction=0.75)
mesh = trimesh.Trimesh(v, f, process=True)
print('surface : %d triangles' % len(mesh.faces))

# ---- Attribution des triangles à la région du site actif ----
_, idx = tree.query(mesh.triangles_center)
site_face = prot_res[idx] > 0
print('triangles du site actif :', int(site_face.sum()))

# ---- Repère géométrique du site actif ----
face_area = mesh.area_faces[site_face]
site_c = np.average(mesh.triangles_center[site_face], axis=0, weights=face_area)
site_n = np.average(mesh.face_normals[site_face], axis=0, weights=face_area)
site_n /= np.linalg.norm(site_n)
# Direction d'ouverture du site : rayons lancés depuis le centre de la région ;
# on garde les directions qui sortent sans traverser la surface (sillon ouvert).
golden = np.pi * (3 - np.sqrt(5)); k = np.arange(800)
zz = 1 - 2 * (k + 0.5) / 800; rr = np.sqrt(1 - zz ** 2)
dirs = np.stack([rr * np.cos(golden * k), zz, rr * np.sin(golden * k)], 1)
start = site_c + site_n * 1.0
hit = mesh.ray.intersects_any(np.repeat(start[None], len(dirs), 0), dirs)
free = dirs[~hit]
print('directions libres :', len(free), '/', len(dirs))
if len(free):
    open_dir = free.mean(0); open_dir /= np.linalg.norm(open_dir)
else:
    open_dir = site_n
prot_c = xyz.mean(0)

# ---- Sphères des fragments glucidiques ----
spheres = []
for x, y, z, r in sugar:
    s = trimesh.creation.icosphere(subdivisions=2, radius=r * 0.85)
    s.apply_translation((x, y, z))
    spheres.append(s)
sugar_mesh = trimesh.util.concatenate(spheres)

# ---- Translation globale : centre en x/z, posé sur y = 0 ; puis Å -> m ----
allv = np.vstack([mesh.vertices, sugar_mesh.vertices])
shift = -np.array([(allv[:, 0].min() + allv[:, 0].max()) / 2,
                   allv[:, 1].min(),
                   (allv[:, 2].min() + allv[:, 2].max()) / 2])
def tf(p):
    return (np.asarray(p) + shift) * SCALE

surf_site = mesh.submesh([np.where(site_face)[0]], append=True)
surf_rest = mesh.submesh([np.where(~site_face)[0]], append=True)
parts = {}
for name, mm in (('protein', surf_rest), ('site', surf_site), ('sugar', sugar_mesh)):
    mm = mm.copy(); mm.vertices = tf(mm.vertices); parts[name] = mm

# ---- Export GLB (4 variantes) ----
def material(rgb, name):
    return trimesh.visual.material.PBRMaterial(
        name=name, baseColorFactor=list(rgb) + [1.0], metallicFactor=0.0, roughnessFactor=0.75)

variants = {}
for sugars in (True, False):
    for site in (False, True):
        key = f"amylase{'-glucides' if sugars else ''}{'-site' if site else ''}"
        scene = trimesh.Scene()
        for name in ('protein', 'site') + (('sugar',) if sugars else ()):
            mm = parts[name].copy()
            colour = COL['site'] if (name == 'site' and site) else COL['protein'] if name == 'site' else COL[name]
            mm.visual = trimesh.visual.TextureVisuals(material=material(colour, name))
            scene.add_geometry(mm, node_name=name, geom_name=name)
        path = OUT / f'{key}.glb'
        scene.export(path)
        variants[key] = {'sugars': sugars, 'site': site}
        print(key, '%.1f Mo' % (path.stat().st_size / 1e6))

# ---- Export USDZ (Quick Look iOS) ----
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt, UsdUtils

def write_usdz(key, sugars, site):
    usda = OUT / f'{key}.usdc'
    stage = Usd.Stage.CreateNew(str(usda))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/Amylase')
    stage.SetDefaultPrim(root.GetPrim())
    for name in ('protein', 'site') + (('sugar',) if sugars else ()):
        mm = parts[name]
        colour = COL['site'] if (name == 'site' and site) else COL['protein'] if name == 'site' else COL[name]
        mat = UsdShade.Material.Define(stage, f'/Amylase/Mat_{name}')
        sh = UsdShade.Shader.Define(stage, f'/Amylase/Mat_{name}/Surface')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.75)
        sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        g = UsdGeom.Mesh.Define(stage, f'/Amylase/{name}')
        g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mm.vertices.astype(np.float32)))
        g.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mm.faces)))
        g.CreateFaceVertexIndicesAttr(Vt.IntArray(mm.faces.flatten().tolist()))
        g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(mm.vertex_normals.astype(np.float32)))
        g.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        g.CreateSubdivisionSchemeAttr('none')
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()
    UsdUtils.CreateNewARKitUsdzPackage(Sdf.AssetPath(str(usda)), str(OUT / f'{key}.usdz'))
    usda.unlink()
    print(key + '.usdz', '%.1f Mo' % ((OUT / f'{key}.usdz').stat().st_size / 1e6))

for key, o in variants.items():
    write_usdz(key, o['sugars'], o['site'])

# ---- Métadonnées pour la page (repère du site actif) ----
sc, pc = tf(site_c), tf(prot_c)
d = open_dir  # axe moyen d'ouverture du sillon du site actif
theta = float(np.degrees(np.arctan2(d[0], d[2])))   # azimut model-viewer
phi = float(np.degrees(np.arccos(np.clip(d[1], -1, 1))))
bbox = np.vstack([p.vertices for p in parts.values()])
meta = {
    'scale_m_per_angstrom': SCALE,
    'enlargement': SCALE / 1e-10,
    'size_cm': ((bbox.max(0) - bbox.min(0)) * 100).round(1).tolist(),
    'site_target_m': sc.round(4).tolist(),
    'site_orbit_deg': [round(theta, 1), round(phi, 1)],
    'center_m': pc.round(4).tolist(),
    'atoms': {'amylase': int(len(prot)), 'glucides': int(len(sugar))},
}
(OUT / 'meta.json').write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
