#!/bin/zsh
# Read-only helper: run one SELECT through the linked Supabase project, print JSON rows.
# Usage: ./dbq.sh "select ..."
# The query runs inside a READ ONLY transaction, so Postgres refuses any write (error 25006).
set -e
here=${0:A:h}
cd "$here/../../Supabase"
out=$(supabase db query --linked "set transaction read only; $1" 2>&1) || { print -u2 -- "$out"; exit 1; }
print -r -- "$out" | python3 -c "
import sys, json
t = sys.stdin.read()
if '{' not in t:
    sys.exit('no JSON in supabase output:\n' + t)
d = json.loads(t[t.index('{'):])
if 'rows' not in d:
    sys.exit('query failed: ' + json.dumps(d))
json.dump(d['rows'], sys.stdout, default=str)
"
