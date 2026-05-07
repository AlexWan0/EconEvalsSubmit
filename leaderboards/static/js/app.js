import { getManifest, getGroups, getLeaderboard, getD5 } from './api.js';
import { populateSelect } from './ui/components/Select.js';
import { Panel } from './ui/components/Panel.js';
import { applyPanelLayout } from './ui/layout.js';
import { setLoading, clearLoading } from './ui/dom.js';
import {
  getState,
  subscribe,
  initLoaded,
  panelSetMode,
  panelSetData,
  panelSetLoading,
  panelSetPromptIndex,
  panelSetGridSort,
  panelSetTableSort,
  panelSetD5Loading,
  panelSetD5,
  panelSetD5Error,
  setLastActivePanel,
  assignColorFor,
  releaseColorFor,
  isSelected,
  resetSelectionsTo,
  togglePromptExpanded,
} from './state/index.js';
import { readURL, setURL, installPopstate, buildURLWithPatch, getAuthToken } from './router.js';
import { ALL_ID, GROUP_PREFIX } from './utils/constants.js';
import { installKeyboard } from './interactions/keyboard.js';

// Element refs
const leftSelect = document.getElementById('left-select');
const rightSelect = document.getElementById('right-select');
const leftContainer = document.getElementById('left-table-container');
const rightContainer = document.getElementById('right-table-container');
const gridContainer = document.querySelector('main .grid');

// ---------- Helpers ----------
function setURLFromState(mode = 'push') {
  const st = getState();
  const leftVal = st.panels.left.mode === 'grid' ? leftSelect.value || ALL_ID : st.panels.left.id;
  const rightVal = st.panels.right.mode === 'grid' ? rightSelect.value || ALL_ID : st.panels.right.id;
  const selected = Array.from(st.selections.keys());
  setURL({ left: leftVal || '', right: rightVal || '', selected }, mode);
}

function setPageTitle(st) {
  const targetPanel = st.panels.right;

  document.title = targetPanel.mode === 'single' ? `${targetPanel.id}` : `Grid (n=${targetPanel.gridIds.length})`
}

function updateLayout() {
  const st = getState();
  applyPanelLayout(
    gridContainer,
    st.panels.left.mode,
    st.panels.right.mode,
    st.ui.lastActivePanel
  );
}

function rerenderPanel(which) {
  const st = getState();
  Panel({
    root: which === 'left' ? leftContainer : rightContainer,
    panelKey: which,
    state: st,
    manifestMap: st.manifestMap,
    onOpenId: (evt, id) => onTileOpen(which, evt, id),
    onToggleModel: onToggleModel,
    onAdvancePrompt: (delta) => onAdvancePrompt(which, delta),
    onChangeSort: (sortId) => { panelSetGridSort(which, sortId); rerenderPanel(which); },
    onToggleExpand: () => { togglePromptExpanded(which); rerenderPanel(which); },
    onHeaderSort: (col) => { panelSetTableSort(which, col); rerenderPanel(which); },
  });
  setPageTitle(st);
}

function rerenderAll() {
  rerenderPanel('left');
  rerenderPanel('right');
}

// Compute grid ids for "All" or a "Group" value
function computeGridIdsFromValue(value) {
  const st = getState();
  if (!value) return [];
  if (value === ALL_ID) {
    return st.manifest.map((m) => m.id);
  }
  if (value.startsWith(GROUP_PREFIX)) {
    const groupId = value.slice(GROUP_PREFIX.length);
    const group = st.groups.find((g) => g.id === groupId);
    return (group?.members || []).filter((id) => st.manifestMap.has(id));
  }
  return [];
}

// Keep selects in sync with state (single subscription)
subscribe((st) => {
  if (st.panels.left.mode === 'single' && st.panels.left.id) {
    leftSelect.value = st.panels.left.id;
  }
  if (st.panels.right.mode === 'single' && st.panels.right.id) {
    rightSelect.value = st.panels.right.id;
  }
});

// ---------- Controller actions ----------
async function onTileOpen(which, evt, id) {
  const openInNew =
    evt.metaKey || evt.ctrlKey || evt.shiftKey || evt.button === 1;
  if (openInNew) {
    // Build URL snapshot after applying the intended id, preserving token
    const st = getState();
    const leftVal =
      which === 'left'
        ? id
        : (st.panels.left.mode === 'grid' ? leftSelect.value || ALL_ID : st.panels.left.id) || '';
    const rightVal =
      which === 'right'
        ? id
        : (st.panels.right.mode === 'grid' ? rightSelect.value || ALL_ID : st.panels.right.id) || '';
    const selected = Array.from(st.selections.keys());
    const url = buildURLWithPatch({ left: leftVal, right: rightVal, selected });
    window.open(url, '_blank', 'noopener');
    return;
  }

  // Switch to single & load
  await switchToSingleAndLoad(which, id);
}

async function switchToSingleAndLoad(which, id) {
  panelSetMode(which, 'single', id);
  await loadDataForPanel(which, id);
  setURLFromState('push');
  rerenderPanel(which);
  updateLayout();
}

async function loadDataForPanel(which, id) {
  const container = which === 'left' ? leftContainer : rightContainer;
  setLoading(container, 'Loading…');
  panelSetLoading(which, true);

  try {
    // Full leaderboard (needed for prompts/responses)
    const data = await getLeaderboard(id);
    panelSetData(which, data);
  } catch (e) {
    console.error('Failed to load leaderboard', id, e);
  } finally {
    clearLoading(container);
    panelSetLoading(which, false);
  }

  // Try to load D5 (optional)
  try {
    panelSetD5Loading(which, true);
    const d5 = await getD5(id);
    if (d5 && Array.isArray(d5) && d5.length) {
      panelSetD5(which, d5);
    } else {
      panelSetD5(which, null); // no data for this leaderboard
    }
  } catch (e) {
    // If 404 was thrown we’d already have handled in api.js, but catch-all just in case
    console.warn('D5 load issue for', id, e?.message || e);
    panelSetD5Error(which, 'Failed to load comparison clusters');
  } finally {
    panelSetD5Loading(which, false);
  }
}

function onToggleModel(modelName) {
  if (!modelName) return;
  if (isSelected(modelName)) releaseColorFor(modelName);
  else assignColorFor(modelName);
  setURLFromState('push');
  rerenderAll();
}

function onAdvancePrompt(which, delta) {
  const st = getState();
  const p = st.panels[which];
  const count = Array.isArray(p?.data?.prompts) ? p.data.prompts.length : 0;
  if (!count) return;
  const idx = p.promptIndex || 0;
  const next = Math.min(Math.max(idx + delta, 0), count - 1);
  if (next !== idx) {
    panelSetPromptIndex(which, next);
    rerenderPanel(which);
  }
}

// ---------- Init ----------
async function init() {
  // Update the header "Home" link to retain p_token (if present)
  const homeLink = document.querySelector('header a[href="/"]');
  const pt = getAuthToken();
  if (homeLink && pt) {
    homeLink.setAttribute('href', `/?p_token=${encodeURIComponent(pt)}`);
  }

  setLoading(leftContainer, 'Loading leaderboards…');
  setLoading(rightContainer, 'Loading leaderboards…');

  let manifest = [];
  let groups = [];
  try {
    manifest = await getManifest();
    groups = await getGroups();
    initLoaded(manifest, groups);
  } catch (err) {
    leftContainer.textContent = 'Failed to fetch data';
    rightContainer.textContent = 'Failed to fetch data';
    return;
  } finally {
    clearLoading(leftContainer);
    clearLoading(rightContainer);
  }

  if (!Array.isArray(manifest) || manifest.length === 0) {
    const msg = 'No leaderboards available. Add entries to data/manifest.json';
    leftContainer.textContent = msg;
    rightContainer.textContent = msg;
    return;
  }

  // Populate dropdowns
  populateSelect(leftSelect, manifest, groups);
  populateSelect(rightSelect, manifest, groups);

  // Defaults (if no URL overrides): left = first LB, right = a group
  leftSelect.value = manifest[0].id;
  rightSelect.value = '__group__:DWA leaderboards';

  // Apply URL state if present
  const urlState = readURL();
  const hasAny =
    urlState.left || urlState.right || (urlState.selected && urlState.selected.length);
  if (hasAny) {
    if (urlState.left) leftSelect.value = urlState.left;
    if (urlState.right) rightSelect.value = urlState.right;

    resetSelectionsTo(urlState.selected || []);

    if (urlState.left) {
      if (urlState.left === ALL_ID || urlState.left.startsWith(GROUP_PREFIX)) {
        const ids = computeGridIdsFromValue(urlState.left);
        panelSetMode('left', 'grid', ids);
      } else {
        panelSetMode('left', 'single', urlState.left);
        await loadDataForPanel('left', urlState.left);
      }
    } else {
      panelSetMode('left', 'single', leftSelect.value);
      await loadDataForPanel('left', leftSelect.value);
    }

    if (urlState.right) {
      if (urlState.right === ALL_ID || urlState.right.startsWith(GROUP_PREFIX)) {
        const ids = computeGridIdsFromValue(urlState.right);
        panelSetMode('right', 'grid', ids);
      } else {
        panelSetMode('right', 'single', urlState.right);
        await loadDataForPanel('right', urlState.right);
      }
    } else {
      const ids = computeGridIdsFromValue(rightSelect.value);
      panelSetMode('right', 'grid', ids);
    }

    setURLFromState('replace');
  } else {
    panelSetMode('left', 'single', leftSelect.value);
    await loadDataForPanel('left', leftSelect.value);

    const rightIds = computeGridIdsFromValue(rightSelect.value);
    panelSetMode('right', 'grid', rightIds);

    setURLFromState('replace');
  }

  rerenderAll();
  updateLayout();

  installPopstate(async () => {
    const { left, right, selected } = readURL();

    resetSelectionsTo(selected || []);

    if (left) {
      if (left === ALL_ID || left.startsWith(GROUP_PREFIX)) {
        panelSetMode('left', 'grid', computeGridIdsFromValue(left));
      } else {
        panelSetMode('left', 'single', left);
        await loadDataForPanel('left', left);
      }
    }
    if (right) {
      if (right === ALL_ID || right.startsWith(GROUP_PREFIX)) {
        panelSetMode('right', 'grid', computeGridIdsFromValue(right));
      } else {
        panelSetMode('right', 'single', right);
        await loadDataForPanel('right', right);
      }
    }

    rerenderAll();
    updateLayout();
  });

  [leftContainer, rightContainer].forEach((el, i) => {
    const which = i === 0 ? 'left' : 'right';
    el.addEventListener('mouseenter', () => setLastActivePanel(which));
    el.addEventListener('focusin', () => setLastActivePanel(which));
  });

  installKeyboard({
    getState,
    rerenderPanel: (which) => rerenderPanel(which),
  });

  leftSelect.addEventListener('change', () => onSelectChange('left', leftSelect.value));
  rightSelect.addEventListener('change', () => onSelectChange('right', rightSelect.value));

  window.addEventListener('resize', () => updateLayout());
}

async function onSelectChange(which, value) {
  if (value === ALL_ID) {
    const allIds = computeGridIdsFromValue(value);
    panelSetMode(which, 'grid', allIds);
    rerenderPanel(which);
  } else if (value.startsWith(GROUP_PREFIX)) {
    const ids = computeGridIdsFromValue(value);
    panelSetMode(which, 'grid', ids);
    rerenderPanel(which);
  } else {
    panelSetMode(which, 'single', value);
    await loadDataForPanel(which, value);
    rerenderPanel(which);
  }

  setURLFromState('push');
  updateLayout();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
