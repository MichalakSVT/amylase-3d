"""Fiche imprimée servant de cible au mode « réalité augmentée avec la fiche ».

Usage : python3 build_fiche.py 3L2L.pdb site https://michalaksvt.github.io/amylase-3d/

Produit :
- site/models/fiche-cible.png : image de la fiche (cible suivie par la caméra) ;
- site/fiche-a4.pdf : la même fiche, à imprimer en A4 paysage, sans marge de mise à l'échelle.
Le dessin est une projection des atomes de la structure 3L2L (amylase en violet, site actif
en vert, fragment glucidique du site en orange) : il fournit les détails dont la caméra
a besoin pour reconnaître la fiche.
"""
import sys, pathlib
import numpy as np
import gemmi, segno
from PIL import Image, ImageDraw, ImageFont

SRC, SITE, BASE = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3].rstrip('/') + '/'
(SITE / 'models').mkdir(parents=True, exist_ok=True)
W, H = 3508, 2480                     # A4 paysage, 300 ppp
FB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FR = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
F = lambda size, bold=False: ImageFont.truetype(FB if bold else FR, size)

st = gemmi.read_structure(SRC); st.remove_alternative_conformations(); st.remove_hydrogens()
m = st[0]
SITE_RES = {197, 233, 300, 101, 201, 299, 305}
atoms = []
for ch in m:
    for r in ch:
        if r.name in ('HOH', 'CA', 'CL'):
            continue
        kind = 'glc' if (r.name == 'GLC' and ch.name == 'C') else ('skip' if r.name == 'GLC' else ('site' if r.seqid.num in SITE_RES else 'prot'))
        if kind == 'skip':
            continue
        for a in r:
            atoms.append((a.pos.x, a.pos.y, a.pos.z, kind))
X = np.array([a[:3] for a in atoms]); kinds = [a[3] for a in atoms]
X -= X.mean(0)
# vue : on regarde le sillon (axe z de projection = direction du fragment glucidique depuis le centre)
g = X[[k == 'glc' for k in kinds]].mean(0); vz = g / np.linalg.norm(g)
vx = np.cross([0, 1, 0], vz); vx /= np.linalg.norm(vx); vy = np.cross(vz, vx)
P = np.stack([X @ vx, X @ vy, X @ vz], 1)

img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
# cadre épais à motifs (aide la détection des bords et des coins)
d.rectangle((40, 40, W - 41, H - 41), outline=(31, 27, 43), width=24)
for i in range(0, W - 260, 120):
    d.rectangle((i + 70, 70, i + 130, 110), fill=(106, 61, 184) if (i // 120) % 2 else (31, 27, 43))
    d.rectangle((i + 70, H - 110, i + 130, H - 70), fill=(31, 27, 43) if (i // 120) % 2 else (106, 61, 184))

# dessin moléculaire
cx, cy, R = 1040, 1440, 860
scale = R / np.abs(P[:, :2]).max()
order = np.argsort(P[:, 2])
zmin, zmax = P[:, 2].min(), P[:, 2].max()
base = {'prot': np.array([106, 61, 184]), 'site': np.array([27, 198, 137]), 'glc': np.array([244, 166, 0])}
for idx in order:
    x, y, z = P[idx]; k = kinds[idx]
    shade = 0.55 + 0.45 * (z - zmin) / (zmax - zmin)
    col = tuple(int(c) for c in np.clip(base[k] * shade + 255 * (1 - shade) * 0.25, 0, 255))
    r = (1.9 if k != 'glc' else 1.7) * scale * 0.55
    px, py = cx + x * scale, cy - y * scale
    d.ellipse((px - r, py - r, px + r, py + r), fill=col, outline=(20, 16, 30), width=2)

# textes et QR codes
d.text((140, 170), "L’amylase en trois dimensions", font=F(118, True), fill=(31, 27, 43))
d.text((140, 320), "Fiche de réalité augmentée · structure PDB 3L2L", font=F(64), fill=(93, 87, 112))
qr_x = 2250
for i, (url, label) in enumerate(((BASE + 'fiche.html', 'Réalité augmentée avec cette fiche'), (BASE, 'Site complet (vue 3D)'))):
    q = segno.make(url, error='m'); tmp = SITE / 'models' / f'_qr{i}.png'
    q.save(str(tmp), scale=22, border=2); qi = Image.open(tmp).convert('RGB'); tmp.unlink()
    size = 760 if i == 0 else 480
    qi = qi.resize((size, size), Image.NEAREST)
    y0 = 520 if i == 0 else 1520
    img.paste(qi, (qr_x + (900 - size) // 2, y0))
    d.text((qr_x + 450, y0 + size + 60), label, font=F(56, True), fill=(90, 51, 158), anchor='mm')
d.text((140, 420), "Scannez le grand QR code, autorisez la caméra, puis filmez toute la fiche posée à plat, bien éclairée.",
       font=F(46), fill=(31, 27, 43))

img.save(SITE / 'fiche-a4.pdf', resolution=300)
cible = img.resize((1754, 1240), Image.LANCZOS)            # cible suivie par la caméra (même cadrage)
cible.save(SITE / 'models' / 'fiche-cible.png', optimize=True)
print('fiche : site/fiche-a4.pdf et site/models/fiche-cible.png')
