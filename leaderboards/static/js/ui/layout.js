import { DESKTOP_MIN_W } from '../utils/constants.js';

export function applyPanelLayout(gridContainer, leftMode, rightMode, preferred = null) {
  if (!gridContainer) return;
  if (window.innerWidth < DESKTOP_MIN_W) {
    gridContainer.style.removeProperty('grid-template-columns');
    return;
  }
  const leftIsGrid = leftMode === 'grid';
  const rightIsGrid = rightMode === 'grid';

  const WIDE = '2.5fr';
  const NARROW = '1fr';

  if (leftIsGrid && !rightIsGrid) gridContainer.style.gridTemplateColumns = `${WIDE} ${NARROW}`;
  else if (rightIsGrid && !leftIsGrid) gridContainer.style.gridTemplateColumns = `${NARROW} ${WIDE}`;
  else if (leftIsGrid && rightIsGrid) {
    if (preferred === 'left') gridContainer.style.gridTemplateColumns = `${WIDE} ${NARROW}`;
    else if (preferred === 'right') gridContainer.style.gridTemplateColumns = `${NARROW} ${WIDE}`;
    else gridContainer.style.gridTemplateColumns = `${WIDE} ${NARROW}`;
  } else {
    gridContainer.style.gridTemplateColumns = '1fr 1fr';
  }
}
