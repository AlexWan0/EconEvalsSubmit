import { el } from '../dom.js';
import { listSortModes } from '../../utils/sort_modes.js';

/**
 * Simple sort bar for grid mode.
 * Props:
 *  - value: current sort mode id (string)
 *  - onChange(id)
 */
export function SortBar({ value = 'default', onChange }) {
  const wrap = el('div', { class: 'px-3 pt-3 flex items-center gap-2 text-xs' });
  const label = el('label', { class: 'text-gray-700 font-medium' }, 'Sort');
  const select = el('select', {
    class: 'border border-gray-300 rounded p-1 text-xs focus:ring-2 focus:ring-blue-500',
  });

  for (const m of listSortModes()) {
    const opt = el('option', { attrs: { value: m.id } }, m.label);
    if (m.id === value) opt.selected = true;
    select.appendChild(opt);
  }

  select.addEventListener('change', () => onChange?.(select.value));

  wrap.append(label, select);
  return wrap;
}
