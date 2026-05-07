import { el } from '../dom.js';
import { ALL_ID, GROUP_PREFIX } from '../../utils/constants.js';

export function populateSelect(elSelect, items, groups) {
  elSelect.innerHTML = '';

  const allOpt = document.createElement('option');
  allOpt.value = ALL_ID;
  allOpt.textContent = 'All leaderboards (grid)';
  elSelect.appendChild(allOpt);

  if (Array.isArray(groups) && groups.length) {
    const og = document.createElement('optgroup');
    og.label = 'Groups';
    groups.forEach((g) => {
      const opt = document.createElement('option');
      opt.value = GROUP_PREFIX + g.id;
      opt.textContent = g.label || g.id;
      og.appendChild(opt);
    });
    elSelect.appendChild(og);
  }

  const og2 = document.createElement('optgroup');
  og2.label = 'Leaderboards';
  items.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item.id;
    opt.textContent = item.label || item.id;
    og2.appendChild(opt);
  });
  elSelect.appendChild(og2);
}
