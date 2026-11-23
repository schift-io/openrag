#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_BASE="${OPENRAG_PG_TMPDIR:-$ROOT_DIR/.tmp-pg}"
PG_DATA="$TMP_BASE/data"
PG_LOG="$TMP_BASE/postgres.log"
DB_NAME="${OPENRAG_TEST_DB:-openrag_test}"
SUPERUSER="postgres"
APP_USER="openrag_app"
WORKER_USER="openrag_worker"

cleanup() {
    if [ -d "$PG_DATA" ] && pg_ctl -D "$PG_DATA" status >/dev/null 2>&1; then
        pg_ctl -D "$PG_DATA" -m immediate stop >/dev/null 2>&1 || true
    fi
    rm -rf "$TMP_BASE"
}
trap cleanup EXIT

rm -rf "$TMP_BASE"
mkdir -p "$PG_DATA"

PG_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"

initdb -D "$PG_DATA" -U "$SUPERUSER" --auth=trust --encoding=UTF8 --locale=C >/dev/null
# -k '' disables Unix domain sockets so long workspace paths never exceed macOS 103-byte sun_path
pg_ctl -D "$PG_DATA" -l "$PG_LOG" -o "-p $PG_PORT -h 127.0.0.1 -k ''" start >/dev/null

createdb -h 127.0.0.1 -p "$PG_PORT" -U "$SUPERUSER" "$DB_NAME"

for f in "$ROOT_DIR"/migrations/[0-9][0-9][0-9]_*.sql; do
    echo "[test-pg-local] applying $(basename "$f")"
    psql --single-transaction -v ON_ERROR_STOP=1 -h 127.0.0.1 -p "$PG_PORT" -U "$SUPERUSER" -d "$DB_NAME" -f "$f" >/dev/null
done

psql -v ON_ERROR_STOP=1 -h 127.0.0.1 -p "$PG_PORT" -U "$SUPERUSER" -d "$DB_NAME" >/dev/null <<SQL
CREATE ROLE $APP_USER LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
CREATE ROLE $WORKER_USER LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE BYPASSRLS;
GRANT USAGE ON SCHEMA iam, vectors, corpus, pii, memory, wiki, indexing, gateway, plugins, eval TO $APP_USER, $WORKER_USER;

-- Least-privilege grants for application query role ($APP_USER)
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA corpus, wiki, memory TO $APP_USER;
GRANT SELECT, INSERT ON pii.audit_logs, pii.guard_events, eval.search_feedback, indexing.outbox TO $APP_USER;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA corpus, wiki, memory, indexing TO $APP_USER;
GRANT SELECT ON iam.orgs, iam.departments, iam.groups, iam.group_members TO $APP_USER;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA iam, vectors TO $APP_USER;

-- Full operational/admin grants for background worker role ($WORKER_USER, BYPASSRLS)
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA iam, vectors, corpus, pii, memory, wiki, indexing, gateway, plugins, eval TO $WORKER_USER;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA iam, vectors, corpus, pii, memory, wiki, indexing, gateway, plugins, eval TO $WORKER_USER;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA iam, vectors TO $WORKER_USER;
SQL

export DATABASE_URL="postgres://$APP_USER@127.0.0.1:$PG_PORT/$DB_NAME"
export ADMIN_DATABASE_URL="postgres://$SUPERUSER@127.0.0.1:$PG_PORT/$DB_NAME"
export WORKER_DATABASE_URL="postgres://$WORKER_USER@127.0.0.1:$PG_PORT/$DB_NAME"

echo "[test-pg-local] PG running at 127.0.0.1:$PG_PORT ($DB_NAME)"
psql -v ON_ERROR_STOP=1 "$DATABASE_URL" -c "SELECT current_user, usesuper, usebypassrls FROM pg_user WHERE usename = current_user;"
psql -v ON_ERROR_STOP=1 "$DATABASE_URL" -c "SELECT count(*) AS schema_count FROM information_schema.schemata WHERE schema_name IN ('iam','vectors','corpus','pii','memory','wiki','indexing','gateway','plugins','eval');"

if [ "$#" -gt 0 ]; then
    "$@"
else
    cd "$ROOT_DIR"
    cargo test --workspace -- --include-ignored
fi
