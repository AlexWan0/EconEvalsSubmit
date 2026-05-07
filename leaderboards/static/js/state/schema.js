// Lightweight "type" hints via JSDoc

/**
 * @typedef {{ id: string, label?: string, file?: string }} ManifestItem
 * @typedef {{ id: string, label?: string, members: string[] }} Group
 * @typedef {Record<string, string|number|null|undefined>} Row
 * @typedef {{
 *   rows: Row[],
 *   prompts?: string[],
 *   response_data?: Record<string, Array<Record<string, any>>>
 * }} Leaderboard
 */

/**
 * @typedef {'single'|'grid'} PanelMode
 * @typedef {{
 *   mode: PanelMode,
 *   id?: string,
 *   gridIds?: string[],
 *   data?: Leaderboard,
 *   promptIndex: number
 * }} PanelState
 */

/**
 * @typedef {{
 *   manifest: ManifestItem[],
 *   manifestMap: Map<string, ManifestItem>,
 *   groups: Group[],
 *   panels: { left: PanelState, right: PanelState },
 *   selections: Map<string, { color: string }>,
 *   expansion: { left: Map<string, Map<string, boolean>>, right: Map<string, Map<string, boolean>> },
 *   ui: {
 *     lastActivePanel: 'left'|'right',
 *     promptExpanded: { left: boolean, right: boolean },
 *     loading: { left: boolean, right: boolean },
 *     errors:  { left?: string, right?: string, global?: string }
 *   }
 * }} AppState
 */

/**
 * @typedef {'single'|'grid'} PanelMode
 * @typedef {{
 *   mode: PanelMode,
 *   id?: string,
 *   gridIds?: string[],
 *   gridSort?: string,        // NEW: id of sort mode for grid
 *   data?: Leaderboard,
 *   promptIndex: number
 * }} PanelState
 */
