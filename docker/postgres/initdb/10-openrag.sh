#!/usr/bin/env bash
# Runs once, on an empty data directory (docker-entrypoint-initdb.d). Applies OpenRAG's migrations
# and creates the two roles the server connects as:
#   openrag_app    NOSUPERUSER, NOBYPASSRLS: every request query goes through row-level security
#   openrag_worker NOSUPERUSER, BYPASSRLS:   background and operator work
# The owner role (POSTGRES_USER) is a superuser and bypasses RLS, so the API must not use it for
# request traffic. Grants match scripts/test-pg-local.sh.
set -euo pipefail
: "${OPENRAG_APP_DB_PASSWORD:?set OPENRAG_APP_DB_PASSWORD}"
: "${OPENRAG_WORKER_DB_PASSWORD:?set OPENRAG_WORKER_DB_PASSWORD}"

psql_db=(psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB")

for f in /openrag/migrations/[0-9][0-9][0-9]_*.sql; do
    echo "openrag: applying $(basename "$f")"
    "${psql_db[@]}" --single-transaction -q -f "$f"
done

"${psql_db[@]}" -q \
    -v app_pw="$OPENRAG_APP_DB_PASSWORD" \
    -v worker_pw="$OPENRAG_WORKER_DB_PASSWORD" <<'SQL'
CREATE ROLE openrag_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS PASSWORD :'app_pw';
CREATE ROLE openrag_worker LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE BYPASSRLS PASSWORD :'worker_pw';
GRANT USAGE ON SCHEMA iam, vectors, corpus, pii, memory, wiki, indexing, gateway, plugins, eval
    TO openrag_app, openrag_worker;

-- Application (request) role: least privilege, RLS enforced
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA corpus, wiki, memory TO openrag_app;
GRANT SELECT, INSERT ON pii.audit_logs, pii.guard_events, eval.search_feedback, indexing.outbox TO openrag_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA corpus, wiki, memory, indexing TO openrag_app;
GRANT SELECT ON iam.orgs, iam.departments, iam.groups, iam.group_members TO openrag_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA iam, vectors TO openrag_app;

-- Worker role: operational and admin access, BYPASSRLS
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES
    IN SCHEMA iam, vectors, corpus, pii, memory, wiki, indexing, gateway, plugins, eval TO openrag_worker;
GRANT USAGE, SELECT ON ALL SEQUENCES
    IN SCHEMA iam, vectors, corpus, pii, memory, wiki, indexing, gateway, plugins, eval TO openrag_worker;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA iam, vectors TO openrag_worker;
SQL
echo "openrag: roles openrag_app, openrag_worker ready"
