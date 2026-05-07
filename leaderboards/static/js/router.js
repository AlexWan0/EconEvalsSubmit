// Centralized URL management + persistent p_token support.
let AUTH_TOKEN = null;

// Capture p_token from the current URL once at module load.
(function bootstrapAuthToken() {
  try {
    const u = new URL(window.location.href);
    const pt = (u.searchParams.get('p_token') || '').trim();
    if (pt) AUTH_TOKEN = pt;
  } catch {}
})();

export function getAuthToken() {
  return AUTH_TOKEN;
}

export function setAuthToken(token) {
  AUTH_TOKEN = token || null;
}

/**
 * Build a new URL (string) from given params, preserving p_token if present.
 * - params: { left?: string, right?: string, selected?: string[] }
 * - mode: 'push' | 'replace' | 'just-build'
 * Returns the URL string if mode === 'just-build'; otherwise performs history op.
 */
export function setURL(params, mode = 'push') {
  const qs = new URLSearchParams();

  if (params?.left) qs.set('l', params.left);
  if (params?.right) qs.set('r', params.right);

  const sel = params?.selected || [];
  if (sel.length) qs.set('sel', sel.map(encodeURIComponent).join(','));

  if (AUTH_TOKEN) qs.set('p_token', AUTH_TOKEN);

  const url = `${location.pathname}?${qs.toString()}`;

  if (mode === 'just-build') return url;

  if (mode === 'replace') {
    history.replaceState(null, '', url);
  } else {
    history.pushState(null, '', url);
  }
}

/**
 * Read current URL params.
 * Returns: { left?:string, right?:string, selected?:string[], p_token?:string }
 */
export function readURL() {
  const p = new URLSearchParams(window.location.search);
  const left = p.get('l') || '';
  const right = p.get('r') || '';
  const selRaw = p.get('sel') || '';
  const selected = selRaw
    .split(',')
    .map(s => s.trim())
    .filter(Boolean)
    .map(decodeURIComponent);

  const token = p.get('p_token') || null;
  if (token) AUTH_TOKEN = token; // keep in sync if it changes

  return { left, right, selected, p_token: token };
}

/**
 * Listen to back/forward nav.
 * handler: async () => void
 */
export function installPopstate(handler) {
  window.addEventListener('popstate', handler);
}

/**
 * Helper to open a new window with a URL built from a "patch" to current state.
 * patch can be like: { left: 'id' } or { right: 'id' } and keeps p_token.
 * selected array is provided directly.
 */
export function buildURLWithPatch({ left, right, selected }) {
  const qs = new URLSearchParams();
  if (left) qs.set('l', left);
  if (right) qs.set('r', right);
  if (selected && selected.length) qs.set('sel', selected.map(encodeURIComponent).join(','));
  if (AUTH_TOKEN) qs.set('p_token', AUTH_TOKEN);
  return `${location.pathname}?${qs.toString()}`;
}
