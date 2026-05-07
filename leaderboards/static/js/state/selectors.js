import { PROMPT_MAX_LINES, PROMPT_MIN_LINES } from '../utils/constants.js';
import { sanitizeMultiline } from '../utils/sanitize.js';

export function getManifestMap(state) { return state.manifestMap; }

export function panel(state, which) { return state.panels[which]; }

export function getPromptCount(lb) {
  return Array.isArray(lb?.prompts) ? lb.prompts.length : 0;
}

export function currentPromptIndex(state, which) {
  return panel(state, which).promptIndex ?? 0;
}

export function currentPromptText(state, which) {
  const p = panel(state, which);
  if (!p?.data) return '';
  const idx = Math.min(Math.max(p.promptIndex || 0, 0), getPromptCount(p.data) - 1);
  return sanitizeMultiline(p.data.prompts?.[idx] || '');
}

export function collectResponsesForPromptIndex(state, which) {
  const p = panel(state, which);
  if (!p?.data) return [];
  const idx = Math.min(Math.max(p.promptIndex || 0, 0), getPromptCount(p.data) - 1);

  const ordered = [
    ['judge_scores', 'Judge scores (GPT-4.1-mini)'],
    ['model_answer', 'Model answer'],
    ['judge', 'Judge notes (GPT-4.1-mini)'],
    ['judge_answer', 'Judge answer (openai:o3-mini-2025-01-31)'],
  ];

  const responseData = (p.data && typeof p.data.response_data === 'object') ? p.data.response_data : {};
  const list = [];

  state.selections.forEach(({ color }, modelName) => {
    const arr = Array.isArray(responseData[modelName]) ? responseData[modelName] : null;
    const obj = arr && arr[idx] ? arr[idx] : null;
    if (!obj) return;

    const seen = new Set();
    const sections = [];
    for (const [key, label] of ordered) {
      if (obj[key] != null && obj[key] !== '') { seen.add(key); sections.push({ key, label, text: sanitizeMultiline(String(obj[key])) }); }
    }
    for (const k of Object.keys(obj)) {
      if (seen.has(k)) continue;
      const val = obj[k];
      if (val != null && val !== '') sections.push({ key: k, label: k, text: sanitizeMultiline(String(val)) });
    }
    list.push({ modelName, bgClass: color, sections });
  });

  return list;
}

export function inferColumns(rows) {
  if (!rows || rows.length === 0) return ['model', 'score'];
  const first = rows[0];
  const extras = Object.keys(first).filter(k => k !== 'model' && k !== 'score');
  return ['model', 'score', ...extras];
}

export { PROMPT_MAX_LINES, PROMPT_MIN_LINES };
