import { el } from '../dom.js';
import { MiniTable } from './MiniTable.js';
import { getLeaderboard } from '../../api.js';
import { TILE_W } from '../../utils/constants.js';
import { getSortMode } from '../../utils/sort_modes.js';

/**
 * Grid of mini-leaderboards.
 * Props:
 * - ids: string[]
 * - manifestMap: Map<string, ManifestItem>
 * - sortMode: string
 * - selectedModels: string[]
 * - scoreCtx?: any
 * - onOpenId(evt, id): function
 * - onRowToggle(modelName): function
 */
export function Grid({ ids, manifestMap, sortMode = 'default', selectedModels = [], scoreCtx = null, onOpenId, onRowToggle }) {
  const container = el('div');

  if (!ids || ids.length === 0) {
    container.appendChild(
      el('div', { class: 'px-3 py-2 text-xs text-gray-500' }, 'No leaderboards to display for this selection.')
    );
    return container;
  }

  const loadingMsg = el(
    'div',
    { class: 'px-3 pt-3 text-xs italic text-gray-500' },
    `Loading ${ids.length} leaderboards…`
  );
  container.appendChild(loadingMsg);

  const grid = el('div', { class: 'flex flex-wrap gap-3 items-start content-start min-w-0 m-3' });
  container.appendChild(grid);

  const mode = getSortMode(sortMode);
  const tileRecords = []; // { id, index, el, score }

  const tasks = ids.map(async (id, index) => {
    const entry = manifestMap.get(id);
    if (!entry) return;

    const tile = el('div', {
      class: 'bg-white border border-gray-200 rounded-lg shadow-sm flex flex-col overflow-hidden shrink-0 grow-0 min-w-0'
    });
    tile.style.width = TILE_W;
    // tile.style.height = TILE_H;

    const headTitle = (entry.label || entry.id).replace('@', ' @ ').replaceAll('_', ' ');
    const head = el('div', {
      class: 'px-2 py-1 border-b border-gray-100 bg-gray-50 cursor-pointer hover:bg-gray-100',
      attrs: { title: headTitle }
    });
    head.appendChild(
      el('div', { class: 'text-[10px] font-semibold text-gray-700 header-clamp w-full min-w-0' }, headTitle)
    );
    tile.appendChild(head);

    const body = el('div', { class: 'flex-1 overflow-hidden p-1 min-w-0' },
      el('div', { class: 'text-[10px] text-gray-400 italic' }, 'Loading…')
    );
    tile.appendChild(body);
    grid.appendChild(tile);

    // Load lightweight data (rows only) and replace body
    let lbData = null;
    try {
      lbData = await getLeaderboard(id, { rowsOnly: true });
      body.innerHTML = '';
      body.appendChild(MiniTable({
        rows: lbData?.rows || [],
        onRowClick: (modelName) => onRowToggle?.(modelName),
      }));
    } catch (err) {
      body.innerHTML = '';
      body.appendChild(el('div', { class: 'text-[10px] text-red-600' }, 'Failed to load.'));
    }

    // Compute score for sorting (with context)
    let score = null;
    try {
      if (lbData && typeof mode.score === 'function') {
        score = mode.score(lbData, selectedModels, scoreCtx);
      }
    } catch (_) { score = null; }

    head.addEventListener('click', (evt) => onOpenId?.(evt, entry.id));

    tileRecords.push({ id, index, el: tile, score });
  });

  Promise.all(tasks).then(() => {
    // Resort only if mode is not default and we computed some scores
    if (mode.id !== 'default') {
      const desc = mode.descending !== false;
      tileRecords.sort((a, b) => {
        const as = a.score; const bs = b.score;
        const aValid = as != null && as !== false && !Number.isNaN(as) && as !== Number.POSITIVE_INFINITY && as !== Number.NEGATIVE_INFINITY;
        const bValid = bs != null && bs !== false && !Number.isNaN(bs) && bs !== Number.POSITIVE_INFINITY && bs !== Number.NEGATIVE_INFINITY;
        if (aValid && bValid) return desc ? (bs - as) : (as - bs);
        if (aValid) return -1;
        if (bValid) return 1;
        return a.index - b.index;
      });

      // Re-append in new order
      grid.innerHTML = '';
      for (const rec of tileRecords) grid.appendChild(rec.el);
    }

    loadingMsg.remove();
  }).catch((err) => {
    loadingMsg.className = 'px-2 py-1 text-xs text-red-600';
    loadingMsg.textContent = `Failed to load one or more leaderboards: ${err?.message}`;
  });

  return container;
}
