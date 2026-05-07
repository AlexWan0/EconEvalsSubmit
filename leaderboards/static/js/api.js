// static/js/api.js
import { getAuthToken } from './router.js';

const cache = new Map();

async function fetchJSON(url) {
  const res = await fetch(url);
  const text = await res.text();
  if (!res.ok) {
    const err = new Error(`Request failed: ${res.status} ${res.statusText}`);
    // Attach status for callers that want to handle 404 as "no data"
    err.status = res.status;
    throw err;
  }
  try {
    return JSON.parse(text);
  } catch (e) {
    console.error('[JSON parse error]', e?.message, { url, preview: text.slice(0, 400) });
    throw e;
  }
}

function withAuthQS(base, extraParams = {}) {
  const pt = getAuthToken();
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(extraParams)) {
    if (v != null && v !== '') qs.set(k, String(v));
  }
  if (pt) qs.set('p_token', pt);
  return qs.toString() ? `${base}?${qs.toString()}` : base;
}

export async function getManifest() {
  const key = `manifest|pt:${getAuthToken() || ''}`;
  if (cache.has(key)) return cache.get(key);
  const url = withAuthQS('/api/leaderboards');
  const data = await fetchJSON(url);
  cache.set(key, data);
  return data;
}

export async function getGroups() {
  const key = `groups|pt:${getAuthToken() || ''}`;
  if (cache.has(key)) return cache.get(key);
  const url = withAuthQS('/api/groups');
  const data = await fetchJSON(url);
  cache.set(key, data);
  return data;
}

/**
 * Fetch a leaderboard (full or rows-only).
 * @param {string} id
 * @param {{ rowsOnly?: boolean }=} opts
 */
export async function getLeaderboard(id, opts = {}) {
  const rowsOnly = !!opts.rowsOnly;
  const key = `lb:${id}|rows:${rowsOnly ? 1 : 0}|pt:${getAuthToken() || ''}`;
  if (cache.has(key)) return cache.get(key);

  const base = `/api/leaderboards/${encodeURIComponent(id)}`;
  const url = withAuthQS(base, rowsOnly ? { rowsonly: '1' } : {});
  const data = await fetchJSON(url);
  cache.set(key, data);
  return data;
}

/**
 * Fetch D5 comparison clusters for a leaderboard.
 * Returns: Array<[string, number, string[]]>
 * Throws on non-404 errors. For 404, caller may treat as "no data".
 */
export async function getD5(id) {
  const key = `d5:${id}|pt:${getAuthToken() || ''}`;
  if (cache.has(key)) return cache.get(key);

  const base = `/api/d5/${encodeURIComponent(id)}`;
  const url = withAuthQS(base);

  try {
    const data = await fetchJSON(url);
    cache.set(key, data);
    return data;
  } catch (e) {
    if (e && e.status === 404) {
      // No D5 file; cache a sentinel so we don't refetch
      cache.set(key, null);
      return null;
    }
    throw e;
  }
}

export function invalidateLeaderboard(id) {
  const pt = getAuthToken() || '';
  cache.delete(`lb:${id}|rows:0|pt:${pt}`);
  cache.delete(`lb:${id}|rows:1|pt:${pt}`);
  cache.delete(`d5:${id}|pt:${pt}`);
}
