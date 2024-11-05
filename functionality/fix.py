import sys
import pathlib
import json

base = pathlib.Path(sys.argv[1])

for i in range(100):
    tokens_file = (base / f"{i}.tokens").open()
    tokens = [int(l.strip("\n")) for l in  tokens_file.readlines()]
    tokens_file.close()

    logits_file = (base / f"{i}.logits").open()
    logits = list(map(json.loads, logits_file.readlines()))

    for t, l in zip(tokens, logits):
        l["max"] = t

    logits_file.close()
    logits_file = (base / f"{i}.logits").open("wt")
    for l in logits:
        logits_file.write(f"{json.dumps(l)}\n")
    
    logits_file.close()
    
