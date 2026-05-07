// Tiny DOM helpers

export function el(tag, opts = {}, ...children) {
  const node = document.createElement(tag);
  if (opts.class) node.className = opts.class;
  if (opts.attrs) for (const [k, v] of Object.entries(opts.attrs)) {
    if (v === undefined || v === null) continue;
    if (k === 'dataset') { for (const [dk, dv] of Object.entries(v)) node.dataset[dk] = dv; }
    else if (k in node) node[k] = v;
    else node.setAttribute(k, v);
  }
  for (const c of children.flat().filter(Boolean)) {
    if (typeof c === 'string') node.appendChild(document.createTextNode(c));
    else node.appendChild(c);
  }
  return node;
}

export function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

export function setLoading(container, text = 'Loading…') {
  if (!container) return;
  let elNode = container.querySelector('[data-loading]');
  if (!elNode) {
    elNode = el('div', { attrs: { 'data-loading': '', role: 'status', ariaLive: 'polite' }, class: 'px-3 py-2 text-xs italic text-gray-500' });
    container.prepend(elNode);
  }
  elNode.textContent = text;
}
export function clearLoading(container) {
  const elNode = container?.querySelector?.('[data-loading]');
  if (elNode) elNode.remove();
}
