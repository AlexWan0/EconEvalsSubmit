from flask import Flask, jsonify, render_template, abort, request, Response
from flask_compress import Compress
import json
from pathlib import Path
import pickle
from safe_open import safe_open
from d5_utils import simplify_output


app = Flask(__name__, static_folder='static', template_folder='templates')

app.config.update(
    COMPRESS_MIMETYPES=[
        'text/html',
        'text/css',
        'text/xml',
        'application/json',
        'application/javascript'
    ],
    COMPRESS_LEVEL=6,
    COMPRESS_MIN_SIZE=500
)
Compress(app)

DATA_DIR = Path(__file__).parent / 'data'
MANIFEST_PATH = DATA_DIR / 'manifest.json'
GROUPS_PATH = DATA_DIR / 'groups.json'

def load_manifest() -> list[dict[str, str]]:
    try:
        with safe_open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    except FileNotFoundError:
        return []

def load_groups() -> list[dict[str, str | list[str]]]:
    try:
        with safe_open(GROUPS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    except FileNotFoundError:
        return []

@app.route('/')
def index() -> str:
    return render_template('index.html')

@app.route('/api/leaderboards')
def list_leaderboards() -> Response:
    return jsonify(load_manifest())

@app.route('/api/groups')
def list_groups() -> Response:
    return jsonify(load_groups())

@app.route('/api/leaderboards/<lb_id>')
def get_leaderboard(lb_id: str) -> Response:
    """
    Optional query param:
      - rowsonly=1|true|yes   -> return only {"rows":[...]} to minimize payloads
    """

    # map lb_id to lb file name using manifest
    manifest = load_manifest()
    entry = next((x for x in manifest if x.get('id') == lb_id), None)
    if not entry:
        abort(404, description="Leaderboard id not found")

    file_name = entry.get('file')
    if not file_name:
        abort(404, description="Leaderboard file not specified")

    # read lb file
    path = DATA_DIR / file_name
    try:
        with safe_open(path, 'r', encoding='utf-8') as f:
            # Replace NaN with null so JSON parses cleanly
            data = json.loads(f.read().replace("NaN", "null"))  # type: ignore

    except FileNotFoundError:
        abort(404, description="Leaderboard file missing")

    if not isinstance(data, dict):
        abort(500, description="Leaderboard file is invalid")

    rows_only_raw = (request.args.get('rowsonly') or '').strip().lower()
    rows_only = rows_only_raw in ('1', 'true', 'yes', 'y')

    # maybe only return rows
    if rows_only:
        rows = data.get('rows') or []
        
        return jsonify({
            'rows': rows 
        })

    # return the result
    return jsonify(data)

@app.route('/api/d5/<lb_id>')
def get_d5(lb_id: str) -> Response:
    # map lb_id to lb file name using manifest
    manifest = load_manifest()
    entry = next((x for x in manifest if x.get('id') == lb_id), None)
    if not entry:
        abort(404, description="Leaderboard id not found")

    file_name = entry.get('file')
    if not file_name:
        abort(404, description="Leaderboard file not specified")

    # read lb file
    path = DATA_DIR / 'comparisons' / file_name.replace('.json', '.pkl')
    try:
        with safe_open(path, 'rb') as f:
            data = pickle.load(f)
        
        return jsonify(
            simplify_output(data)
        )

    except FileNotFoundError:
        abort(404, description=f"Comparison data file not found")

    if not isinstance(data, dict):
        abort(500, description=f"Comparison data file is invalid")
