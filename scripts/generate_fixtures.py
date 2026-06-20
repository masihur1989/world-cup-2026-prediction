import pandas as pd
from itertools import combinations
import datetime

GROUPS = {
    "A": ["United States", "Brazil",    "Morocco",  "Serbia"],
    "B": ["Mexico",        "Netherlands","Japan",    "Cameroon"],
    "C": ["Canada",        "Germany",   "Ecuador",  "Ivory Coast"],
    "D": ["Spain",         "Argentina", "Saudi Arabia", "Albania"],
    "E": ["France",        "Colombia",  "South Korea",  "Tunisia"],
    "F": ["England",       "Uruguay",   "Iran",     "DR Congo"],
    "G": ["Portugal",      "Belgium",   "Australia","Senegal"],
    "H": ["Croatia",       "Venezuela", "Egypt",    "Iraq"],
    "I": ["Austria",       "Paraguay",  "Jordan",   "Mali"],
    "J": ["Denmark",       "Turkey",    "Nigeria",  "Panama"],
    "K": ["Switzerland",   "Scotland",  "Uzbekistan","Costa Rica"],
    "L": ["Czechia",       "Slovakia",  "New Zealand","Honduras"],
}

VENUES = {
    "A": "MetLife Stadium, East Rutherford",
    "B": "Estadio Azteca, Mexico City",
    "C": "AT&T Stadium, Arlington",
    "D": "Rose Bowl, Pasadena",
    "E": "Levi's Stadium, Santa Clara",
    "F": "SoFi Stadium, Inglewood",
    "G": "BMO Stadium, Los Angeles",
    "H": "Empower Field, Denver",
    "I": "Lincoln Financial Field, Philadelphia",
    "J": "BC Place, Vancouver",
    "K": "Commonwealth Stadium, Edmonton",
    "L": "NRG Stadium, Houston",
}

base_dates = {
    "A": [datetime.date(2026, 6, 11), datetime.date(2026, 6, 15), datetime.date(2026, 6, 20),
          datetime.date(2026, 6, 19), datetime.date(2026, 6, 24), datetime.date(2026, 6, 25)],
    "B": [datetime.date(2026, 6, 11), datetime.date(2026, 6, 15), datetime.date(2026, 6, 20),
          datetime.date(2026, 6, 19), datetime.date(2026, 6, 24), datetime.date(2026, 6, 25)],
    "C": [datetime.date(2026, 6, 12), datetime.date(2026, 6, 16), datetime.date(2026, 6, 21),
          datetime.date(2026, 6, 20), datetime.date(2026, 6, 24), datetime.date(2026, 6, 25)],
    "D": [datetime.date(2026, 6, 12), datetime.date(2026, 6, 16), datetime.date(2026, 6, 21),
          datetime.date(2026, 6, 20), datetime.date(2026, 6, 25), datetime.date(2026, 6, 25)],
    "E": [datetime.date(2026, 6, 13), datetime.date(2026, 6, 17), datetime.date(2026, 6, 22),
          datetime.date(2026, 6, 21), datetime.date(2026, 6, 25), datetime.date(2026, 6, 25)],
    "F": [datetime.date(2026, 6, 13), datetime.date(2026, 6, 17), datetime.date(2026, 6, 22),
          datetime.date(2026, 6, 21), datetime.date(2026, 6, 25), datetime.date(2026, 6, 25)],
    "G": [datetime.date(2026, 6, 14), datetime.date(2026, 6, 18), datetime.date(2026, 6, 23),
          datetime.date(2026, 6, 22), datetime.date(2026, 6, 25), datetime.date(2026, 6, 25)],
    "H": [datetime.date(2026, 6, 14), datetime.date(2026, 6, 18), datetime.date(2026, 6, 23),
          datetime.date(2026, 6, 22), datetime.date(2026, 6, 25), datetime.date(2026, 6, 25)],
    "I": [datetime.date(2026, 6, 14), datetime.date(2026, 6, 18), datetime.date(2026, 6, 23),
          datetime.date(2026, 6, 22), datetime.date(2026, 6, 25), datetime.date(2026, 6, 25)],
    "J": [datetime.date(2026, 6, 15), datetime.date(2026, 6, 19), datetime.date(2026, 6, 23),
          datetime.date(2026, 6, 22), datetime.date(2026, 6, 25), datetime.date(2026, 6, 25)],
    "K": [datetime.date(2026, 6, 15), datetime.date(2026, 6, 19), datetime.date(2026, 6, 24),
          datetime.date(2026, 6, 23), datetime.date(2026, 6, 25), datetime.date(2026, 6, 25)],
    "L": [datetime.date(2026, 6, 16), datetime.date(2026, 6, 20), datetime.date(2026, 6, 24),
          datetime.date(2026, 6, 23), datetime.date(2026, 6, 25), datetime.date(2026, 6, 25)],
}

rows = []
for group, teams in GROUPS.items():
    pairs = list(combinations(teams, 2))  # 6 matchups
    dates = base_dates[group]
    venue = VENUES[group]
    for i, (t_a, t_b) in enumerate(pairs):
        rows.append({
            "team_a": t_a,
            "team_b": t_b,
            "stage": f"Group {group}",
            "date": dates[i],
            "venue": venue,
        })

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df.to_csv("data/raw/wc2026_fixtures.csv", index=False)
print(f"Wrote {len(df)} fixtures.")
