import { inferColumns } from '../../state/selectors.js';
import { el } from '../dom.js';
import { getSelectionColor } from '../../state/index.js';

function rowClass(idx, modelName) {
  const zebra = idx % 2 === 0 ? 'bg-white' : 'bg-gray-50';
  const color = getSelectionColor(modelName);
  const bg = color ? color : zebra;
  return `cursor-pointer ${bg}`;
}

function toNumber(v) {
  if (v == null) return NaN;
  const n = typeof v === 'number' ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : NaN;
}

function sortRows(rows, sort) {
  const by = sort?.by;
  if (!by) return rows;
  const dir = sort?.dir === 'asc' ? 1 : -1;
  const isScore = String(by).toLowerCase() === 'score';

  const arr = rows.slice();
  arr.sort((a, b) => {
    const av = a?.[by];
    const bv = b?.[by];
    if (isScore) {
      const an = toNumber(av);
      const bn = toNumber(bv);
      if (Number.isFinite(an) && Number.isFinite(bn)) return (an - bn) * dir;
      if (Number.isFinite(an)) return -1;
      if (Number.isFinite(bn)) return 1;
      return 0;
    }
    // String-ish compare fallback
    const as = av == null ? '' : String(av);
    const bs = bv == null ? '' : String(bv);
    return as.localeCompare(bs, undefined, { numeric: true, sensitivity: 'base' }) * dir;
  });
  return arr;
}

/**
 * Table
 * Props:
 *  - rows: array of row objects
 *  - sort: { by: string|null, dir: 'asc'|'desc' }
 *  - onRowClick(modelName)
 *  - onHeaderClick(columnKey)
 */
export function Table({ rows, sort, onRowClick, onHeaderClick }) {
  const columns = inferColumns(rows);
  const table = el('table', { class: 'w-full table-fixed text-[0.75rem]' });

  // colgroup sizing
  const modelIdx = columns.findIndex(c => c.toLowerCase() === 'model');
  const scoreIdx = columns.findIndex(c => c.toLowerCase() === 'score');
  const ciIdx = columns.findIndex(c => c.toLowerCase() === 'CI (%)');

  const colgroup = el('colgroup');
  for (let i = 0; i < columns.length; i++) {
    const col = el('col');
    if (i === scoreIdx) {
      col.style.width = '3rem'; col.style.maxWidth = '3rem';
    } else if (i !== modelIdx) {
      col.style.width = '4rem'; col.style.maxWidth = '4rem';
    }
    colgroup.appendChild(col);
  }
  table.appendChild(colgroup);

  const thead = el('thead', { class: 'sticky top-0 bg-gray-100 z-10' });
  const headerRow = el('tr');

  columns.forEach(colName => {
    const isScore = colName.toLowerCase() === 'score';
    const isActive = sort?.by === colName;
    const indicator = isActive ? (sort?.dir === 'asc' ? ' ▲' : ' ▼') : '';
    const th = el('th', {
      class: `px-2 py-2 font-medium text-gray-700 truncate ${isScore ? 'text-right' : 'text-left'} select-none`,
      attrs: { scope: 'col', title: colName }
    });

    const btn = el(
      'button',
      { class: 'inline-flex items-center gap-1 hover:underline', attrs: { type: 'button' } },
      isScore ? `${colName}${indicator}` : `${colName}${indicator}`
    );
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      onHeaderClick?.(colName);
    });

    th.appendChild(btn);
    headerRow.appendChild(th);
  });

  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = el('tbody');
  const sorted = sortRows(rows || [], sort);

  sorted.forEach((row, idx) => {
    const modelName = String(row['model'] ?? '');
    const tr = el('tr', { class: rowClass(idx, modelName), attrs: { 'data-model': modelName, 'title': modelName } });
    tr.addEventListener('click', () => onRowClick?.(modelName));

    columns.forEach(colName => {
      const isScore = colName.toLowerCase() === 'score';
      const td = el('td', { class: 'px-2 py-2' });
      const valStr = row[colName] === undefined ? '' : String(row[colName]);
      const inner = el('div', { class: isScore ? 'truncate text-right tabular-nums' : 'truncate', attrs: { title: valStr } }, valStr);
      td.appendChild(inner);
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  return table;
}
