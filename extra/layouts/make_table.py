import csv
import json

COMPILERS=["CLANG", "GCC"]
ARCHS=["znver2", "haswell"]
FUNCTIONS=["MORTONBITSHIFT", "MORTONPDEP", "PITCHEDNAIVE", "PITCHEDPRECALC", "PITCHEDFAST"]

COMBINATIONS = [(c + "_" + a + "_" + f) for c in COMPILERS for a in ARCHS for f in FUNCTIONS]

for c in COMPILERS:
    for a in ARCHS:
        with open("output_" + c + "_" + a + ".csv", "w") as f:
            w = csv.DictWriter(f, ["DIMS"] + FUNCTIONS)
            w.writeheader()

            for N in range(1, 11):
                row = {"DIMS": N}

                for D in FUNCTIONS:
                    with open("out/json/" + c + "_" + a + "_" + D + "_" + str(N) + ".json", "r") as g:
                        d = json.loads(g.read())

                    rt = d["CodeRegions"][0]["SummaryView"]["BlockRThroughput"]

                    row[D] = rt

                w.writerow(row)
