import { COLOR_PALETTE } from '../utils/constants.js';

/** @type {import('./schema.js').AppState} */
const initialState = {
  manifest: [],
  manifestMap: new Map(),
  groups: [],
  panels: {
    left:  {
      mode: 'single',
      id: undefined,
      gridIds: [],
      gridSort: 'default',
      data: undefined,
      promptIndex: 0,
      tableSort: { by: null, dir: 'asc' },
      d5: { data: null, loading: false, error: null },
    },
    right: {
      mode: 'single',
      id: undefined,
      gridIds: [],
      gridSort: 'default',
      data: undefined,
      promptIndex: 0,
      tableSort: { by: null, dir: 'asc' },
      d5: { data: null, loading: false, error: null },
    },
  },
  selections: new Map(),
  expansion: { left: new Map(), right: new Map() },
  ui: {
    lastActivePanel: 'left',
    promptExpanded: { left: false, right: false },
    loading: { left: false, right: false },
    errors: {}
  }
};

const listeners = new Set();
let state = initialState;

export function getState() { return state; }
export function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }
function emit() { for (const fn of listeners) fn(state); }

// ---------- Reducers ----------
export function initLoaded(manifest, groups) {
  state = {
    ...state,
    manifest,
    manifestMap: new Map(manifest.map(m => [m.id, m])),
    groups,
  };
  emit();
}

export function panelSetMode(panel, mode, idsOrId) {
  const isGrid = mode === 'grid';
  const prev = state.panels[panel] || {};
  const panels = structuredClone(state.panels);
  if (isGrid) {
    panels[panel] = {
      mode,
      gridIds: idsOrId || [],
      gridSort: prev.gridSort || 'default',
      id: undefined,
      data: undefined,
      promptIndex: 0,
      tableSort: prev.tableSort || { by: null, dir: 'asc' },
      d5: { data: null, loading: false, error: null }, // reset in grid mode
    };
  } else {
    panels[panel] = {
      mode,
      id: idsOrId,
      gridIds: [],
      gridSort: prev.gridSort || 'default',
      data: undefined,
      promptIndex: 0,
      tableSort: prev.tableSort || { by: null, dir: 'asc' },
      d5: { data: null, loading: false, error: null }, // will load per id
    };
  }
  state = { ...state, panels };
  emit();
}

export function panelSetLoading(panel, isLoading) {
  state = { ...state, ui: { ...state.ui, loading: { ...state.ui.loading, [panel]: !!isLoading } } };
  emit();
}

export function panelSetError(panel, msg) {
  state = { ...state, ui: { ...state.ui, errors: { ...state.ui.errors, [panel]: msg || undefined } } };
  emit();
}

export function panelSetData(panel, data) {
  const panels = structuredClone(state.panels);
  panels[panel].data = data;
  state = { ...state, panels };
  emit();
}

export function panelSetPromptIndex(panel, idx) {
  const panels = structuredClone(state.panels);
  panels[panel].promptIndex = Math.max(0, Number(idx) || 0);
  state = { ...state, panels };
  emit();
}

export function togglePromptExpanded(panel) {
  const prev = state.ui.promptExpanded[panel];
  state = { ...state, ui: { ...state.ui, promptExpanded: { ...state.ui.promptExpanded, [panel]: !prev } } };
  emit();
}

export function setExpansion(panel, modelName, sectionKey, isOpen) {
  const tree = state.expansion[panel];
  if (!tree.has(modelName)) tree.set(modelName, new Map());
  tree.get(modelName).set(sectionKey, !!isOpen);
  state = { ...state, expansion: { ...state.expansion } };
  emit();
}

export function setLastActivePanel(panel) {
  state = { ...state, ui: { ...state.ui, lastActivePanel: panel } };
  emit();
}

export function panelSetGridSort(panel, sortId) {
  const panels = structuredClone(state.panels);
  if (!panels[panel]) return;
  panels[panel].gridSort = sortId || 'default';
  state = { ...state, panels };
  emit();
}

export function panelSetTableSort(panel, columnKey) {
  const panels = structuredClone(state.panels);
  const cur = panels[panel]?.tableSort || { by: null, dir: 'asc' };
  if (!columnKey) {
    panels[panel].tableSort = { by: null, dir: 'asc' };
  } else if (cur.by === columnKey) {
    panels[panel].tableSort = { by: columnKey, dir: cur.dir === 'asc' ? 'desc' : 'asc' };
  } else {
    const defaultDir = String(columnKey).toLowerCase() === 'score' ? 'desc' : 'asc';
    panels[panel].tableSort = { by: columnKey, dir: defaultDir };
  }
  state = { ...state, panels };
  emit();
}

// ---- NEW: D5 per-panel state ----
export function panelSetD5Loading(panel, flag) {
  const panels = structuredClone(state.panels);
  panels[panel].d5 = panels[panel].d5 || { data: null, loading: false, error: null };
  panels[panel].d5.loading = !!flag;
  state = { ...state, panels };
  emit();
}

export function panelSetD5(panel, data /* array or null */) {
  const panels = structuredClone(state.panels);
  panels[panel].d5 = { data, loading: false, error: null };
  state = { ...state, panels };
  emit();
}

export function panelSetD5Error(panel, msg) {
  const panels = structuredClone(state.panels);
  panels[panel].d5 = { data: null, loading: false, error: msg || 'Error' };
  state = { ...state, panels };
  emit();
}

// Selections (unchanged) …
const availableColors = [...COLOR_PALETTE];

export function resetSelectionsTo(modelNames) {
  availableColors.length = 0; availableColors.push(...COLOR_PALETTE);
  state.selections.clear();
  for (const name of (modelNames || [])) {
    assignColorFor(name);
  }
  state = { ...state, selections: new Map(state.selections) };
  emit();
}

export function assignColorFor(modelName) {
  if (state.selections.has(modelName)) return state.selections.get(modelName).color;
  const color = availableColors.shift() || COLOR_PALETTE[state.selections.size % COLOR_PALETTE.length];
  state.selections.set(modelName, { color });
  state = { ...state, selections: new Map(state.selections) };
  emit();
  return color;
}

export function releaseColorFor(modelName) {
  const entry = state.selections.get(modelName);
  if (!entry) return;
  const color = entry.color;
  if (!availableColors.includes(color) && COLOR_PALETTE.includes(color)) availableColors.push(color);
  state.selections.delete(modelName);
  state = { ...state, selections: new Map(state.selections) };
  emit();
}

export function isSelected(modelName) {
  return state.selections.has(modelName);
}
export function getSelectionColor(modelName) {
  return state.selections.get(modelName)?.color || null;
}
export function getSelectedModelNames() {
  return Array.from(state.selections.keys());
}
