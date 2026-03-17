# Configuration protection de branche main

Ces règles sont à configurer manuellement sur GitHub :
Settings → Branches → Add branch protection rule → branch name: `main`

## Règles à activer

| Règle | Paramètre |
|---|---|
| Require a pull request before merging | ✅ activé |
| Required approvals | 1 |
| Dismiss stale pull request approvals | ✅ activé |
| Require status checks to pass | ✅ activé |
| Required checks | `lint`, `tests` |
| Require branches to be up to date | ✅ activé |
| Do not allow bypassing the above settings | ✅ activé |
| Restrict who can push to matching branches | ✅ activé |
