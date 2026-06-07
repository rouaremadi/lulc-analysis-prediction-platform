"""
GeoAI Manouba — CA-Markov Prediction Backend  v4.0
===================================================
Setup:  pip install flask flask-cors numpy shapely
Run:    python server.py

Files expected in same folder as server.py:
  lulc_1985.geojson  — polygonized 1985 classification (gridcode 1=veg,2=builtup,3=water,4=bare)
  lulc_2025.geojson  — polygonized 2025 classification (same gridcode scheme)

What it does:
  1. Loads both GeoJSONs on startup
  2. Computes real transition matrix from 1985→2025 pixel areas
  3. On /api/generate_predictions:
       - Takes transition matrix (or recomputes with user interventions)
       - Assigns each 2025 polygon a predicted class for 2030/2035/2040
       - Returns 3 GeoJSONs ready to display on the map
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import json, os, math, random

app  = Flask(__name__)
CORS(app)

# ── Class definitions ─────────────────────────────────────────────────────────
CLASSES = ['water', 'builtup', 'veg', 'bare']

# gridcode → class for YOUR rasters
# lulc_1985.geojson: 1=veg, 2=builtup, 3=water, 4=bare  (ml_85_rec)
# lulc_2025.geojson: same scheme expected
GRIDCODE_MAP     = {1:'water', 2:'builtup', 3:'veg', 4:'bare'}  # new class integers
OLD_GRIDCODE_MAP = {1:'veg', 13:'veg', 22:'builtup', 48:'bare'}  # old ArcMap polygonize

# also support text class property
CLASS_TEXT_MAP = {
    'water':'water','eau':'water','waterbody':'water','water body':'water',
    'urban':'builtup','builtup':'builtup','built-up':'builtup','buitup':'builtup',
    'buitlup':'builtup','residential':'builtup','urban area':'builtup',
    'vegetation':'veg','veg':'veg','forest':'veg','cropland':'veg','végétation':'veg',
    'bare':'bare','bare soil':'bare','sol nu':'bare','barren':'bare','baresoil':'bare',
}

# ── Real 1985 pixel counts from ml_85_rec ────────────────────────────────────
# Value 1=Water, 2=Builtup, 3=Vegetation, 4=Bare
_PX_KM2 = 0.0009  # 30m pixel = 900 m² = 0.0009 km²
# ml_85_rec pixel counts: Value 1=Water, 2=Builtup, 3=Vegetation, 4=Bare
PIXELS_1985 = {'water':13823, 'builtup':52205, 'veg':257155, 'bare':777100}
AREA_1985   = {c: PIXELS_1985[c]*_PX_KM2 for c in CLASSES}
TOTAL_1985  = sum(AREA_1985.values())
PCT_1985    = {c: round(AREA_1985[c]/TOTAL_1985*100, 3) for c in CLASSES}

# ── Intervention effects (km² per placement) ─────────────────────────────────
INTERVENTION_EFFECTS = {
    'dam':         {'water':+5.0, 'veg':+2.0,  'bare': -4.0, 'builtup':  0.0},
    'agri':        {'veg':  +8.0, 'water':+1.0, 'bare': -9.0, 'builtup':  0.0},
    'forest':      {'veg': +12.0, 'water':+1.0, 'bare':-10.0, 'builtup': -3.0},
    'residential': {'builtup':+10.0,'bare':-5.0,'veg':  -5.0, 'water':    0.0},
    'solar':       {'builtup': +3.0,'bare':-6.0,'veg':  -2.0, 'water':    0.0},
    'industrial':  {'builtup':+15.0,'bare':-8.0,'veg':  -7.0, 'water':    0.0},
}

# ══════════════════════════════════════════════════════════════════════════════
# GEOJSON LOADER
# ══════════════════════════════════════════════════════════════════════════════
GJ_1985 = None
GJ_2025 = None
REAL_MATRIX = None   # computed from real data once both files loaded

def get_class(props):
    """Extract class key from feature properties.
    Supports:
      - class: integer (1=water,2=builtup,3=veg,4=bare)  ← new GeoJSON format
      - class_name: text ('Vegetation','Builtup',etc.)
      - gridcode: integer (old polygonize format)
      - class: text string (old format)
    """
    # 1. Try integer 'class' field (new format: class=1,2,3,4)
    cls_val = props.get('class') or props.get('Class')
    if isinstance(cls_val, (int, float)):
        return GRIDCODE_MAP.get(int(cls_val), 'bare')

    # 2. Try gridcode (old polygonize format)
    gc = props.get('gridcode') or props.get('Gridcode')
    if gc is not None:
        return OLD_GRIDCODE_MAP.get(int(gc), GRIDCODE_MAP.get(int(gc), 'bare'))

    # 3. Try class_name text field (new format)
    cn = props.get('class_name') or props.get('ClassName')
    if cn and isinstance(cn, str):
        return CLASS_TEXT_MAP.get(cn.lower().strip(), 'bare')

    # 4. Try class as text string (old format)
    if isinstance(cls_val, str) and cls_val:
        return CLASS_TEXT_MAP.get(cls_val.lower().strip(), 'bare')

    # 5. Try other text fields
    raw = (props.get('lulc') or props.get('LULC') or '')
    if isinstance(raw, str) and raw:
        return CLASS_TEXT_MAP.get(raw.lower().strip(), 'bare')

    return 'bare'

def feature_area_km2(feat):
    """Get area from property or approximate from geometry."""
    p = feat.get('properties') or {}
    a = (p.get('area_km2') or p.get('Area_km2') or p.get('AREA_KM2'))
    if a:
        a = float(a)
        return a/1e6 if a > 10000 else a
    # approximate using shoelace
    geom  = feat.get('geometry') or {}
    gtype = geom.get('type','')
    coords= geom.get('coordinates',[])
    try:
        if gtype == 'Polygon':
            return _ring_area(coords[0])
        elif gtype == 'MultiPolygon':
            return sum(_ring_area(p[0]) for p in coords)
    except:
        pass
    return 0.0

def _ring_area(ring):
    if not ring or len(ring)<3: return 0
    n = len(ring)
    a = sum(ring[i][0]*ring[(i+1)%n][1] - ring[(i+1)%n][0]*ring[i][1]
            for i in range(n))
    a = abs(a)/2
    lat = sum(c[1] for c in ring)/n
    return a * 111.0 * 111.0*math.cos(math.radians(lat))

def feature_centroid(feat):
    geom  = feat.get('geometry') or {}
    gtype = geom.get('type','')
    coords= geom.get('coordinates',[])
    try:
        ring = coords[0] if gtype=='Polygon' else coords[0][0]
        return sum(c[0] for c in ring)/len(ring), sum(c[1] for c in ring)/len(ring)
    except:
        return None, None

def compute_matrix_from_geojsons(gj1985, gj2025):
    """
    Compute 40-year transition matrix from two GeoJSONs.
    Strategy: match features spatially by centroid proximity,
    then tally FROM→TO class transitions weighted by area.
    """
    # Build area totals per class for each year
    areas_from = {c:0.0 for c in CLASSES}
    areas_to   = {c:0.0 for c in CLASSES}

    for f in gj1985.get('features',[]):
        cls  = get_class(f.get('properties') or {})
        areas_from[cls] += feature_area_km2(f)

    for f in gj2025.get('features',[]):
        cls  = get_class(f.get('properties') or {})
        areas_to[cls] += feature_area_km2(f)

    total_from = sum(areas_from.values())
    total_to   = sum(areas_to.values())

    if total_from < 1e-6:
        # fall back to pixel counts
        areas_from = dict(AREA_1985)
        total_from = TOTAL_1985

    pct_from = {c: areas_from[c]/total_from for c in CLASSES}
    pct_to   = {c: areas_to[c]/total_to     for c in CLASSES} if total_to>1e-6 else pct_from

    print(f"  1985 areas: { {c:round(areas_from[c],1) for c in CLASSES} }")
    print(f"  2025 areas: { {c:round(areas_to[c],1)   for c in CLASSES} }")

    # Build transition matrix from proportional change
    T = np.eye(len(CLASSES))
    from_arr = np.array([pct_from[c] for c in CLASSES])
    to_arr   = np.array([pct_to[c]   for c in CLASSES])
    delta    = to_arr - from_arr

    losers  = {i:-delta[i] for i in range(4) if delta[i]<0}
    gainers = {i:+delta[i] for i in range(4) if delta[i]>0}
    total_lost = sum(losers.values())

    for i, lost in losers.items():
        if from_arr[i] < 1e-6: continue
        T[i][i] = max(0, (from_arr[i]-lost)/from_arr[i])
        for j, gained in gainers.items():
            total_gained = sum(gainers.values())
            T[i][j] = max(0, (gained/total_gained)*(lost/from_arr[i]))

    # normalize rows
    for i in range(4):
        s = T[i].sum()
        if s > 1e-6: T[i] /= s

    return T, areas_from, areas_to

def matrix_to_annual(T, years=40):
    vals, vecs = np.linalg.eig(T)
    vals  = np.real(vals); vecs = np.real(vecs)
    vals_p = np.where(vals>0, vals**(1.0/years), 0)
    R = vecs @ np.diag(vals_p) @ np.linalg.inv(vecs)
    R = np.clip(R, 0, 1)
    s = R.sum(axis=1, keepdims=True)
    s = np.where(s<1e-6, 1, s)
    return R/s

def load_geojsons():
    global GJ_1985, GJ_2025, REAL_MATRIX
    base = os.path.dirname(os.path.abspath(__file__))

    for year, varname in [(1985,'GJ_1985'),(2025,'GJ_2025')]:
        path = os.path.join(base, f'lulc_{year}.geojson')
        if os.path.exists(path):
            with open(path,'r',encoding='utf-8') as f:
                data = json.load(f)
            n = len(data.get('features',[]))
            print(f"  ✓ lulc_{year}.geojson loaded ({n} features)")
            if year==1985: GJ_1985=data
            else:          GJ_2025=data
        else:
            print(f"  ⚠ lulc_{year}.geojson not found")

    if GJ_1985 and GJ_2025:
        print("  Computing real transition matrix from both GeoJSONs...")
        REAL_MATRIX, _, _ = compute_matrix_from_geojsons(GJ_1985, GJ_2025)
        print(f"  ✓ Real transition matrix computed")
        for i,c in enumerate(CLASSES):
            print(f"    FROM {c:8s}: {[round(float(v),3) for v in REAL_MATRIX[i]]}")
    elif GJ_1985:
        print("  ⚠ Only 1985 loaded — using hardcoded matrix until 2025 available")

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def get_base_matrix(interventions):
    """
    Return appropriate transition matrix:
    - If both GeoJSONs loaded AND no interventions → use real matrix
    - If interventions → recompute matrix from 1985 + modified 2025
    - If only 1985 loaded → use hardcoded fallback
    """
    if interventions and GJ_1985:
        # Apply interventions to 2025 areas then recompute matrix
        base = dict(AREA_1985)
        for iv in interventions:
            for cls, delta in INTERVENTION_EFFECTS.get(iv,{}).items():
                base[cls] = max(0, base.get(cls,0) + delta)
        total = sum(base.values())
        pct_2025 = {c: base[c]/total for c in CLASSES}
        pct_1985 = {c: AREA_1985[c]/TOTAL_1985 for c in CLASSES}
        # build matrix
        T = np.eye(4)
        from_arr = np.array([pct_1985[c] for c in CLASSES])
        to_arr   = np.array([pct_2025[c] for c in CLASSES])
        delta    = to_arr - from_arr
        losers   = {i:-delta[i] for i in range(4) if delta[i]<0}
        gainers  = {i:+delta[i] for i in range(4) if delta[i]>0}
        total_gained = sum(gainers.values())
        for i,lost in losers.items():
            if from_arr[i]<1e-6: continue
            T[i][i] = max(0,(from_arr[i]-lost)/from_arr[i])
            for j,gained in gainers.items():
                T[i][j] = max(0,(gained/total_gained)*(lost/from_arr[i]))
        for i in range(4):
            s=T[i].sum()
            if s>1e-6: T[i]/=s
        return matrix_to_annual(T,40), pct_2025
    elif REAL_MATRIX is not None:
        pct_2025 = {c: 0.0 for c in CLASSES}
        if GJ_2025:
            areas={c:0.0 for c in CLASSES}
            for f in GJ_2025.get('features',[]):
                cls=get_class(f.get('properties') or {})
                areas[cls]+=feature_area_km2(f)
            total=sum(areas.values())
            if total>0: pct_2025={c:areas[c]/total for c in CLASSES}
        return matrix_to_annual(REAL_MATRIX,40), pct_2025
    else:
        # hardcoded fallback matrix (from ArcMap tabulate area)
        _T40 = np.array([
            [0.0014,0.0493,0.4007,0.5486],
            [0.0011,0.0921,0.3010,0.6059],
            [0.0007,0.0558,0.3494,0.5942],
            [0.0011,0.0341,0.2386,0.7262],
        ])
        return matrix_to_annual(_T40,40), dict(PCT_1985)

def predict_geojson(source_gj, T1_annual, start_year, target_year, seed=42):
    """
    Given a source GeoJSON and annual transition matrix,
    predict class of each feature at target_year.
    Uses stochastic assignment based on transition probabilities.
    """
    rng   = random.Random(seed + target_year)
    steps = target_year - start_year
    features_out = []

    cls_idx = {c:i for i,c in enumerate(CLASSES)}

    # Raise T1 to the power of steps to get multi-year matrix
    T_steps = np.linalg.matrix_power(
        np.round(T1_annual, 6), steps
    )
    # clip and renormalize
    T_steps = np.clip(T_steps, 0, 1)
    for i in range(4):
        s = T_steps[i].sum()
        if s > 1e-6: T_steps[i] /= s

    for feat in source_gj.get('features', []):
        props = dict(feat.get('properties') or {})
        geom  = feat.get('geometry')

        # get current class
        from_cls = get_class(props)
        i = cls_idx.get(from_cls, 3)

        # sample predicted class based on transition probabilities
        probs     = T_steps[i]
        r         = rng.random()
        cumsum    = 0.0
        to_cls    = CLASSES[-1]  # default
        for j, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                to_cls = CLASSES[j]
                break

        # update properties
        props['class']       = to_cls
        props['class_label'] = to_cls.capitalize()
        props['pred_year']   = target_year
        props['from_class']  = from_cls
        props['gridcode']    = {'water':3,'builtup':2,'veg':1,'bare':4}[to_cls]

        features_out.append({
            'type':       'Feature',
            'properties': props,
            'geometry':   geom,
        })

    return {
        'type':     'FeatureCollection',
        'name':     f'lulc_pred_{target_year}',
        'features': features_out,
    }

def env_score(pct):
    s = (pct.get('veg',0)*0.50 + pct.get('water',0)*0.28
       - pct.get('builtup',0)*0.22 - pct.get('bare',0)*0.12 + 34)
    return int(max(0,min(100,round(s))))

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/health')
def health():
    return jsonify({
        'status':       'ok',
        'model':        'CA-Markov v4.0',
        'lulc_1985':    'loaded' if GJ_1985 else 'missing',
        'lulc_2025':    'loaded' if GJ_2025 else 'missing',
        'real_matrix':  'computed' if REAL_MATRIX is not None else 'using fallback',
        'lulc_1985_pct': PCT_1985,
    })

@app.route('/api/generate_predictions', methods=['POST'])
def generate_predictions():
    """
    POST body:
    {
      "interventions": ["forest","dam"],   (optional)
      "years": [2030, 2035, 2040]          (optional, default these 3)
    }

    Returns:
    {
      "predictions": {
        "2030": { ...GeoJSON... },
        "2035": { ...GeoJSON... },
        "2040": { ...GeoJSON... }
      },
      "stats": {
        "2030": {"water":X,"builtup":X,"veg":X,"bare":X,"env_score":X},
        ...
      },
      "transition_matrix": [...],
      "lulc_source": "real"|"fallback"
    }
    """
    body          = request.get_json(force=True, silent=True) or {}
    interventions = body.get('interventions', [])
    years         = body.get('years', [2030, 2035, 2040])

    # need at least lulc_1985 as source
    source_gj = GJ_2025 if GJ_2025 else GJ_1985
    if not source_gj:
        return jsonify({'error': 'No GeoJSON loaded. Add lulc_1985.geojson to server folder.'}), 400

    start_year  = 2025 if GJ_2025 else 1985
    T1, pct_now = get_base_matrix(interventions)
    lulc_source = 'real' if REAL_MATRIX is not None else 'fallback'

    predictions = {}
    stats       = {}

    for yr in years:
        gj_pred = predict_geojson(source_gj, T1, start_year, yr)
        predictions[str(yr)] = gj_pred

        # compute stats from predicted GeoJSON
        areas = {c:0.0 for c in CLASSES}
        for f in gj_pred['features']:
            cls   = f['properties'].get('class','bare')
            areas[cls] += feature_area_km2(f)
        total = sum(areas.values())
        pct   = {c: round(areas[c]/total*100,2) for c in CLASSES} if total>0 else {c:25.0 for c in CLASSES}
        stats[str(yr)] = {**pct, 'env_score': env_score(pct)}

    # also include 2025 baseline stats
    if GJ_2025:
        areas25={c:0.0 for c in CLASSES}
        for f in GJ_2025.get('features',[]):
            cls=get_class(f.get('properties') or {})
            areas25[cls]+=feature_area_km2(f)
        total25=sum(areas25.values())
        pct25={c:round(areas25[c]/total25*100,2) for c in CLASSES} if total25>0 else pct_now
    else:
        pct25=PCT_1985

    return jsonify({
        'predictions':         predictions,
        'stats':               stats,
        'baseline_2025':       pct25,
        'interventions_used':  interventions,
        'transition_matrix':   T1.tolist(),
        'lulc_source':         lulc_source,
        'years_generated':     years,
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Lightweight predict — returns trajectory only (no GeoJSON)."""
    body          = request.get_json(force=True, silent=True) or {}
    delegation    = body.get('delegation','Manouba')
    interventions = body.get('interventions',[])
    steps         = int(body.get('steps',10))

    T1, pct_2025 = get_base_matrix(interventions)
    state = np.array([pct_2025.get(c,0)/100 for c in CLASSES])
    trajectory = []
    for yr in range(steps):
        state  = state @ T1
        state += np.random.normal(0, 0.003, 4)
        state  = np.clip(state,0,1); state/=state.sum()
        trajectory.append({'year':2025+yr+1,
            **{c:round(float(state[i]*100),2) for i,c in enumerate(CLASSES)}})

    final = trajectory[-1]
    return jsonify({
        'delegation':        delegation,
        'after_interventions':{c:round(pct_2025.get(c,0),2) for c in CLASSES},
        'annual_trajectory': trajectory,
        'predicted_2035':    {c:round(final[c],2) for c in CLASSES},
        'env_score_2025':    env_score(pct_2025),
        'env_score_2035':    env_score({c:final[c] for c in CLASSES}),
        'lulc_source':       'generated',
        'model':             'CA-Markov v4.0',
    })

if __name__ == '__main__':
    print("="*60)
    print("  GeoAI Manouba — CA-Markov Backend  v4.0")
    print("  http://localhost:5000")
    print("="*60)
    print("  GET  /api/health")
    print("  POST /api/generate_predictions  ← NEW: returns map GeoJSONs")
    print("  POST /api/predict               ← trajectory only")
    print("="*60)
    load_geojsons()
    print("="*60)
    app.run(debug=True, port=5000, host='0.0.0.0')
