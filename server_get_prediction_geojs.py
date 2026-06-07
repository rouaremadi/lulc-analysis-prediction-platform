import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from collections import Counter
import os

np.random.seed(42)

# ══════════════════════════════════════════
LULC_2025 = r"D:/capstone_project/semestre_2/webs+geojson_layers/lulc_2025.geojson"
OUT_DIR   = r"D:/capstone_project/semestre_2/webs+geojson_layers/"
# ════════════════════════════════════════════

CLASS_NAME = {1:'Water', 2:'Urban', 3:'Vegetation', 4:'Bare Soil'}

def get_class(row):
    for col in ['class','Class','CLASS']:
        if col in row.index and row[col] is not None:
            try: return int(row[col])
            except: pass
    m = {'water':1,'urban':2,'vegetation':3,'bare soil':4,'bare':4,'buitup':2,'builtup':2}
    return m.get(str(row.get('class_name','')).lower().strip(), 4)

# Fallback areas (km²) — Manouba reality
a85 = {1:12.4, 2:47.0, 3:231.4, 4:699.4}
a25 = {1:6.0, 2:304.0, 3:81.0, 4:599.0}

print("Areas (km²):")
print(f"  {'':>12} {'1985':>8} {'2025':>8} {'Chg':>8}")
for c in [1,2,3,4]:
    d = a25[c]-a85[c]
    print(f"  {CLASS_NAME[c]:>12} {a85[c]:>8.1f} {a25[c]:>8.1f} {d:>+7.1f}")

# Transition matrix
P = np.zeros((5,5))
veg_lost = a85[3]-a25[3]
bare_lost = a85[4]-a25[4]
water_lost = a85[1]-a25[1]
total_lost = veg_lost + bare_lost + water_lost
P[3][2] = (veg_lost/total_lost)*0.95;  P[3][3] = 1-P[3][2]
P[4][2] = (bare_lost/total_lost)*0.95;  P[4][4] = 1-P[4][2]
P[1][2] = (water_lost/total_lost)*0.95; P[1][1] = 1-P[1][2]
P[2][2] = a25[2]/a85[2]

print("\nTransition matrix:")
for c85 in [1,2,3,4]:
    print(f"  {CLASS_NAME[c85]:>12}", end="")
    for c25 in [1,2,3,4]:
        print(f" {P[c85][c25]*100:>7.1f}%", end="")
    print()

# Read polygons
print("\nReading LULC 2025...")
g25 = gpd.read_file(LULC_2025)
n = len(g25)
classes_25 = np.array([get_class(row) for _, row in g25.iterrows()])
for c in [1,2,3,4]:
    print(f"  {CLASS_NAME[c]}: {(classes_25==c).sum()}")

# Centroids in original CRS (UTM meters)
print("\nBuilding neighbors...")
cents = np.zeros((n, 2))
for i, row in g25.iterrows():
    try:
        cx = float(row.geometry.centroid.x)
        cy = float(row.geometry.centroid.y)
        if np.isfinite(cx) and np.isfinite(cy):
            cents[i] = [cx, cy]
    except: pass

# Use 45m radius (1.5x pixel size) in the coordinate units of the data
radius = 45.0
tree = cKDTree(cents)
nb_raw = tree.query_ball_point(cents, r=radius)

neighbors = [[] for _ in range(n)]
for i in range(n):
    for j in nb_raw[i]:
        if j != i:
            neighbors[i].append(j)

avg_nb = np.mean([len(x) for x in neighbors if len(x) > 0])
has_nb = sum(1 for x in neighbors if len(x) > 0)
print(f"  Radius: {radius} | Polygons with neighbors: {has_nb}/{n} | Avg: {avg_nb:.1f}")

# Predict
for year in [2030, 2035, 2040]:
    alpha = (year - 2025) / 40.0
    print(f"\n{'='*45}")
    print(f"  {year} (scale {alpha:.0%})")
    print(f"{'='*45}")

    I = np.eye(5)
    Ps = (1-alpha)*I + alpha*P

    new_cls = np.copy(classes_25)
    for i in range(n):
        probs = Ps[classes_25[i], 1:5].copy()
        probs = np.maximum(probs, 0)
        probs /= probs.sum()
        new_cls[i] = np.random.choice([1,2,3,4], p=probs)

    for pass_num in range(2):
        smoothed = np.copy(new_cls)
        changed = 0
        for i in range(n):
            if len(neighbors[i]) < 2: continue
            nb_cls = [new_cls[j] for j in neighbors[i]]
            counts = Counter(nb_cls)
            top_cls, top_cnt = counts.most_common(1)[0]
            if top_cnt/len(nb_cls) > 0.50 and top_cls != new_cls[i]:
                smoothed[i] = top_cls
                changed += 1
        new_cls = smoothed
        print(f"  Pass {pass_num+1}: {changed} adjusted")

    for c in [1,2,3,4]:
        cnt = (new_cls==c).sum()
        orig = (classes_25==c).sum()
        d = cnt-orig
        s = "+" if d>=0 else ""
        p = (d/orig*100) if orig>0 else 0
        print(f"  {CLASS_NAME[c]:>11}: {cnt:>6} ({s}{d}, {s}{p:.1f}%)")

    out = g25.copy()
    out['class'] = new_cls
    out['class_name'] = [CLASS_NAME[c] for c in new_cls]
    for col in ['_cls']:
        if col in out.columns: out = out.drop(columns=[col])
    out.to_file(os.path.join(OUT_DIR, f"pred_{year}.geojson"), driver='GeoJSON')
    print(f"  ✅ pred_{year}.geojson")

print(f"\n{'='*45}\n  DONE!\n{'='*45}")