import { el } from '../dom.js';
import { isSelected, getSelectionColor } from '../../state/index.js';

function rowClass(idx, modelName) {
  const zebra = idx % 2 === 0 ? 'bg-white' : 'bg-gray-50';
  const color = getSelectionColor(modelName);
  return `cursor-pointer ${color ? color : zebra}`;
}

/**
 * @param {{ rows: Array<Record<string, any>>, onRowClick?: (modelName:string)=>void }} props
 */
export function MiniTable({ rows, onRowClick }) {
  const columns = ['model', 'score'];
  const table = el('table', { class: 'text-[10px] leading-tight table-fixed' });

  const tbody = el('tbody');
  rows.forEach((row, idx) => {
    const modelName = String(row['model'] ?? '');
    const tr = el('tr', { class: rowClass(idx, modelName), attrs: { 'data-model': modelName, 'title': modelName } });

    tr.addEventListener('click', () => {
      if (!modelName) return;
      onRowClick?.(modelName);
    });

    columns.forEach(col => {
      const td = el('td');
      if (col === 'model') {
        td.className = 'px-0.5 py-0.5 text-gray-900 truncate overflow-hidden text-ellipse align-top w-[7.5rem] max-w-[7.5rem]';
      } else if (col === 'score') {
        td.className = 'px-0.5 py-0.5 text-gray-900 text-left tabular-nums truncate overflow-hidden text-ellipsis align-top w-[2.5rem] max-w-[2.5rem]';
      } else {
        td.className = 'px-1 py-0.5 text-gray-900 truncate';
      }
      const val = row[col];
      td.textContent = val === undefined ? '' : String(val);
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  return table;
}
