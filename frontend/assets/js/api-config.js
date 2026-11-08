/* Goodminton — backend location. This is the only file to edit per environment.
 *
 *   Production   baseUrl: 'https://maincloud.spacetimedb.com'
 *   Local        baseUrl: 'http://localhost:3000'   (after `spacetime start`)
 *
 * Never put a SpacetimeDB token here: this file ships to every visitor, and all the
 * tables the site reads are public. See API.md for the contract.
 */
window.GM_API_CONFIG = {
  baseUrl: 'https://maincloud.spacetimedb.com',
  database: 'badminton'
};
