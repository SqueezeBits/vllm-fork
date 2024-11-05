import sys
import pathlib
import json

a = pathlib.Path(sys.argv[1])
b = pathlib.Path(sys.argv[2])
assert a.exists()
assert b.exists()

def mse(foo, bar):
    assert len(foo) == len(bar)
    return sum((f - b)**2 for f, b in zip(foo, bar)) / len(foo)

err = 0.0
for i in range(100):
    logits = f"{i}.logits"
    tokens = f"{i}.tokens"

    a_logit_file = (a / logits).open() 
    b_logit_file = (b / logits).open() 
    a_token_file = (a / tokens).open() 
    b_token_file = (b / tokens).open() 

    a_tokens = a_token_file.readlines()
    b_tokens = b_token_file.readlines()
    a_logits = [json.loads(l)["logits"] for l in a_logit_file.readlines()[:len(a_tokens)]]
    b_logits = [json.loads(l)["logits"] for l in b_logit_file.readlines()[:len(b_tokens)]]
 
    assert len(a_tokens) == len(b_tokens)
    assert all(at == bt for at, bt in zip(a_tokens, b_tokens))

    err += (sum(mse(a, b) for a, b in zip(a_logits, b_logits)) / len(a_logits))


print(err / 100)


    
    


