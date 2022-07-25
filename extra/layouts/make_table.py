import csv
import json

COMPILERS=["CLANG", "GCC"]
FUNCTIONS=["MORTONBITSHIFT", "MORTONPDEP", "PITCHEDNAIVE", "PITCHEDPRECALC", "PITCHEDFAST"]

COMBINATIONS = [(c + "_" + f) for c in COMPILERS for f in FUNCTIONS]

with open("output.csv", "w") as f:
    w = csv.DictWriter(f, ["DIMS"] + COMBINATIONS)
    w.writeheader()

    for N in range(1, 11):
        row = {"DIMS": N}

        for D in COMBINATIONS:
            with open("out/json/" + D + "_" + str(N) + ".json", "r") as g:
                d = json.loads(g.read())

            rt = d["CodeRegions"][0]["SummaryView"]["BlockRThroughput"]

            row[D] = rt

        w.writerow(row)
