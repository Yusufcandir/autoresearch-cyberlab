# Security policy

- Report vulnerabilities privately to the project lead (do not open a public issue).
- No secrets in Git. CI runs secret scanning (gitleaks) on every push and PR.
- Untrusted code, datasets with executable content and hostile fixtures never run on the
  shared GPU workstation. Without an isolated host they are `BLOCKED: ISOLATION_REQUIRED`.
- LLM agents never hold admin/root privileges and cannot grant themselves permissions.
- See `docs/threat_model.md` for the full threat model (Step 03).
