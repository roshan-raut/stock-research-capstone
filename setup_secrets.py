"""
One-time setup script: creates the Databricks secret scopes and stores the
Lakebase connection URL + Massive API key. Run this from a Databricks
notebook (%sh python setup_secrets.py) or locally with the Databricks CLI
configured - never commit the resulting secret values anywhere.

Usage:
    python setup_secrets.py
"""

import getpass

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace

w = WorkspaceClient()

for scope in ("database", "massive"):
    try:
        w.secrets.create_scope(scope=scope)
    except Exception as exc:
        print(f"Scope '{scope}' may already exist ({exc}); continuing.")

w.secrets.put_secret(
    scope="database",
    key="lakebase-url",
    string_value=getpass.getpass("Paste your Lakebase connection URL: "),
)

w.secrets.put_secret(
    scope="massive",
    key="api-key",
    string_value=getpass.getpass("Paste your Massive Stocks API key: "),
)

for scope in ("database", "massive"):
    w.secrets.put_acl(scope=scope, principal="users", permission=workspace.AclPermission.READ)

print("Secrets stored. Both scopes are readable by all workspace users (READ only).")
