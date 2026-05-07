// static/js/utils/sort_modes.js
// -----------------------------------------------------------------------------
// Sort modes for mini leaderboard tiles.
//
// A sort mode is an object:
//   {
//     id: string,
//     label: string,
//     // Compute a score for a given leaderboard.
//     // Higher scores sort first unless descending === false (then smaller first).
//     // lbData: { rows: Array<{ model: string, score: number|string, ... }>, ... }
//     // selectedModels: string[] (tracked models; may be empty)
//     // ctx (optional): {
//     //   otherPanel: {
//     //     isSingle: boolean,
//     //     leaderboard: { rows: ... } | null
//     //   }
//     // }
//     score(lbData, selectedModels, ctx?): number | null,
//     descending?: boolean // default true
//   }
//
// Notes:
// - If a mode returns null/NaN/±Infinity, Grid will push those behind valid scores
//   (and preserve manifest order among equally invalid scores).
// - “Ranking” modes compute ranks from the leaderboard’s numeric scores
//   (higher score => better rank 1). Ties keep array order (stable).
// -----------------------------------------------------------------------------

// ---------- shared helpers ----------

function toNumber(v) {
  if (v == null) return NaN;
  const n = typeof v === 'number' ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : NaN;
}

function rowsWithValidScores(lbData) {
  const rows = Array.isArray(lbData?.rows) ? lbData.rows : [];
  return rows
    .map(r => ({ model: String(r?.model ?? ''), score: toNumber(r?.score) }))
    .filter(r => r.model && Number.isFinite(r.score));
}

function buildRankMap(lbData) {
  // rank 1 = highest score
  const rows = rowsWithValidScores(lbData).sort((a, b) => b.score - a.score);
  const rankByModel = new Map();
  rows.forEach((r, i) => rankByModel.set(r.model, i + 1)); // 1-based
  return rankByModel;
}

function pickSelectedScores(lbData, selectedModels) {
  if (!Array.isArray(selectedModels) || selectedModels.length === 0) return [];
  const wanted = new Set(selectedModels.map(String));
  const out = [];
  for (const r of rowsWithValidScores(lbData)) {
    if (wanted.has(r.model)) out.push(r.score);
  }
  return out;
}

function pickSelectedRanks(rankMap, selectedModels) {
  if (!Array.isArray(selectedModels) || selectedModels.length === 0) return [];
  const out = [];
  for (const m of selectedModels) {
    const rk = rankMap.get(String(m));
    if (rk != null) out.push(rk);
  }
  return out;
}

function rangeOf(nums) {
  if (!nums.length) return NaN;
  let min = Infinity, max = -Infinity;
  for (const n of nums) { if (n < min) min = n; if (n > max) max = n; }
  return max - min;
}

function meanOf(nums) {
  if (!nums.length) return NaN;
  let s = 0;
  for (const n of nums) s += n;
  return s / nums.length;
}

// Sum of absolute differences in RANKS across selected models
// between lbA and lbB. Models missing from either side are ignored.
// Returns +Infinity if nothing to compare (so it sinks to the bottom).
function totalAbsRankDiff(lbA, lbB, selectedModels) {
  if (!lbA || !lbB) return Number.POSITIVE_INFINITY;
  const rkA = buildRankMap(lbA);
  const rkB = buildRankMap(lbB);
  let sum = 0;
  let compared = 0;
  for (const m of selectedModels || []) {
    const a = rkA.get(String(m));
    const b = rkB.get(String(m));
    if (a != null && b != null) {
      sum += Math.abs(a - b);
      compared++;
    }
  }
  return compared > 0 ? sum : Number.POSITIVE_INFINITY;
}

// ---------- modes ----------

export const SORT_MODES = {
  // Keep manifest order
  default: {
    id: 'default',
    label: 'Default',
    score: () => null,
    descending: true,
  },

  // Score-range among selected models: max(score) - min(score). Larger is better.
  range: {
    id: 'range',
    label: 'Range (score)',
    score: (lbData, selectedModels) => {
      const vals = pickSelectedScores(lbData, selectedModels);
      if (!vals.length) return -Infinity; // sink
      return rangeOf(vals);
    },
    descending: true,
  },

  // Rank-range among selected models: max(rank) - min(rank). Smaller is better.
  range_ranking: {
    id: 'range_ranking',
    label: 'Range (rank)',
    score: (lbData, selectedModels) => {
      const rk = buildRankMap(lbData);
      const vals = pickSelectedRanks(rk, selectedModels);
      if (!vals.length) return Number.POSITIVE_INFINITY; // sink
      return rangeOf(vals); // smaller = better
    },
    descending: false, // ASC
  },

  // Mean of selected models' scores. Larger is better.
  mean: {
    id: 'mean',
    label: 'Mean (score)',
    score: (lbData, selectedModels) => {
      const vals = pickSelectedScores(lbData, selectedModels);
      if (!vals.length) return -Infinity;
      return meanOf(vals);
    },
    descending: true,
  },

  // Mean of selected models' ranks. Smaller is better.
  mean_ranking: {
    id: 'mean_ranking',
    label: 'Mean (rank)',
    score: (lbData, selectedModels) => {
      const rk = buildRankMap(lbData);
      const vals = pickSelectedRanks(rk, selectedModels);
      if (!vals.length) return Number.POSITIVE_INFINITY;
      return meanOf(vals); // smaller = better
    },
    descending: false, // ASC
  },

  // OPTIONAL: total absolute difference of ranks vs the other panel’s SINGLE leaderboard.
  // Smaller is better (tighter agreement with the other side).
  // NOTE: This mode only makes sense when the *other* panel is in single mode.
  rank_diff_vs_other: {
    id: 'rank_diff_vs_other',
    label: 'Rank Δ vs other',
    score: (lbData, selectedModels, ctx) => {
      const other = ctx?.otherPanel;
      if (!other?.isSingle || !other?.leaderboard) {
        // No valid "other" to compare; push to bottom.
        return Number.POSITIVE_INFINITY;
      }
      return totalAbsRankDiff(lbData, other.leaderboard, selectedModels);
    },
    descending: false, // ASC (smaller diffs first)
  },
};

// ---------- API ----------

export function listSortModes() {
  // Return in a nice order for the dropdown
  const ids = [
    'default',
    'range', 'range_ranking',
    'mean', 'mean_ranking',
    'rank_diff_vs_other', // keep last since it depends on other panel
  ];
  return ids.map(id => SORT_MODES[id]).filter(Boolean);
}

export function getSortMode(id) {
  return SORT_MODES[id] || SORT_MODES.default;
}
