import { el, clear } from '../dom.js';
import { Table } from './Table.js';
import { Grid } from './Grid.js';
import { PromptPager } from './PromptPager.js';
import { Responses } from './Responses.js';
import { SortBar } from './SortBar.js';
import { D5Summary } from './D5Summary.js';
import { collectResponsesForPromptIndex } from '../../state/selectors.js';
import { getSelectedModelNames } from '../../state/index.js';

export function Panel({
  root,
  panelKey,
  state,
  manifestMap,
  onOpenId,
  onToggleModel,
  onAdvancePrompt,
  onChangeSort,
  onToggleExpand,
  onHeaderSort,
}) {
  clear(root);

  const p = state.panels[panelKey];

  if (p.mode === 'grid') {
    const otherKey = panelKey === 'left' ? 'right' : 'left';
    const otherPanel = state.panels[otherKey];
    const scoreCtx = {
      otherPanel: {
        isSingle: otherPanel?.mode === 'single',
        leaderboard: otherPanel?.mode === 'single' ? otherPanel?.data : null,
      },
    };

    const sortUi = SortBar({
      value: p.gridSort || 'default',
      onChange: (id) => onChangeSort?.(id),
    });
    root.appendChild(sortUi);

    const grid = Grid({
      ids: p.gridIds || [],
      manifestMap,
      sortMode: p.gridSort || 'default',
      selectedModels: getSelectedModelNames(),
      scoreCtx,
      onOpenId: (evt, id) => onOpenId?.(evt, id),
      onRowToggle: (modelName) => onToggleModel?.(modelName),
    });
    root.appendChild(grid);
    return;
  }

  // Single mode
  const data = p.data;
  if (!data) {
    root.appendChild(el('div', { class: 'px-3 py-2 text-xs text-gray-500' }, 'Loading…'));
    return;
  }

  const tableBox = el('div', { class: 'px-3 pb-3 pt-5' });
  root.appendChild(tableBox);

  const rows = data?.rows || [];
  const table = Table({
    rows,
    sort: p.tableSort,
    onRowClick: (modelName) => onToggleModel?.(modelName),
    onHeaderClick: (col) => onHeaderSort?.(col),
  });
  tableBox.appendChild(table);

  const pager = PromptPager({
    panelKey,
    lb: data,
    state,
    onAdvance: (delta) => onAdvancePrompt?.(delta),
    onToggleExpand: () => onToggleExpand?.(),
    responsesNodeFactory: () =>
      Responses({
        panelKey,
        items: collectResponsesForPromptIndex(state, panelKey),
        expansion: state.expansion[panelKey],
      }),
  });
  root.appendChild(pager);

  // D5 Block
  const d5 = p.d5 || { data: null, loading: false, error: null };
  if (d5.loading) {
    root.appendChild(el('div', { class: 'px-3 py-2 text-[11px] italic text-gray-500' }, 'Loading comparison clusters…'));
  } else if (Array.isArray(d5.data) && d5.data.length > 0) {
    root.appendChild(D5Summary({ items: d5.data }));
  } // else: no D5 available -> show nothing
}
