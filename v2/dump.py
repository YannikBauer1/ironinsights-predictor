"""Dump a flat, read-only snapshot of every pro result to data/results.csv, and the athletes marked
inactive (retired) to data/inactive_athletes.json.

Only SELECTs go to the database (through ./dbq.sh, the linked Supabase project).
usage: dump.py            both
       dump.py inactive   only the inactive list (leaves the results snapshot as it is)
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "data" / "results.csv"
INACTIVE = HERE / "data" / "inactive_athletes.json"
INACTIVE_SQL = "select a.person_id, p.name_short as name from athlete a join person p on p.id = a.person_id where not a.active"

SQL = """
select r.id result_id, r.athlete_id, a.person_id, p.name person_slug, p.name_short person_name, p.sex,
       p.birthdate, p.country_code person_country,
       r.place, r.judging_1, r.judging_2, r.judging_3, r.judging_4, r.total, r.routine,
       r.predicted_place, r.predicted_win_percentage, r.predicted_confidence, r.created_at result_created_at,
       d.id division_id, d.description division_desc, d.qualified_top,
       ct.name cat_type, ca.name age, cl.name level, ce.name enh, cw.name weight_class,
       e.id event_id, e.start_date, e.end_date, e.year, e.country_code event_country,
       co.id competition_id, co.name competition, co.name_short competition_short,
       qr.name ruleset, t.tier legacy_tier
from result r
join athlete a on a.id = r.athlete_id
join person p on p.id = a.person_id
join division d on d.id = r.division_id
join event e on e.id = d.event_id
join competition co on co.id = e.competition_id
join category c on c.id = d.category_id
left join category_type ct on ct.id = c.category_type_id
left join category_age ca on ca.id = c.category_age_id
left join category_level cl on cl.id = c.category_level_id
left join category_enhancement ce on ce.id = c.category_enhancement_id
left join category_weight cw on cw.id = c.category_weight_id
left join qualification_ruleset qr on qr.id = e.ruleset_id
left join tier t on t.id = e.tier_id
where e.year between {y0} and {y1}
order by e.start_date, d.id, r.place nulls last
"""

YEAR_CHUNKS = [(1900, 1989), (1990, 1999), (2000, 2005), (2006, 2010), (2011, 2013), (2014, 2015),
               (2016, 2017), (2018, 2019), (2020, 2021), (2022, 2022), (2023, 2023), (2024, 2024),
               (2025, 2025), (2026, 2030)]


def fetch(y0, y1):
    out = subprocess.run(["./dbq.sh", SQL.format(y0=y0, y1=y1)], cwd=HERE, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"query for {y0}-{y1} failed:\n{out.stderr}")
    return json.loads(out.stdout)


def dump_inactive():
    """athlete.active is today's flag: fine for today's power ranking, never for backtests."""
    out = subprocess.run(["./dbq.sh", INACTIVE_SQL], cwd=HERE, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"inactive query failed:\n{out.stderr}")
    rows = json.loads(out.stdout)
    INACTIVE.write_text(json.dumps(rows, indent=1))
    print("inactive athletes", len(rows), "->", INACTIVE, file=sys.stderr)


def main():
    rows = []
    for y0, y1 in YEAR_CHUNKS:
        chunk = fetch(y0, y1)
        print(y0, y1, len(chunk), file=sys.stderr)
        rows.extend(chunk)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("rows", len(rows), "->", OUT, file=sys.stderr)


if __name__ == "__main__":
    if sys.argv[1:] != ["inactive"]:
        main()
    dump_inactive()
