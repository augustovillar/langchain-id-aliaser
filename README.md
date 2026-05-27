# langchain-id-aliaser

Reversible **UUID / ID aliasing** for LangChain & LangGraph. Replace long
identifiers with short, stable aliases before they reach the LLM — and restore
the originals on the way back, including in the model's outgoing tool calls.

```
f47ac10b-58cc-4372-a567-0e02b2c3d479   →   usr_a9Fk2
        (~15-19 tokens)                        (~2 tokens)
```

## Why

- **Cheaper.** A UUID tokenizes into ~15-19 tokens. In agent loops the same ids
  echo through tool calls, results, and state — the cost multiplies. Aliases are
  1-2 tokens.
- **More accurate.** LLMs frequently mangle or hallucinate long UUIDs. Short,
  stable aliases are far harder to corrupt, so tool calls round-trip correctly.

## Install

```bash
pip install langchain-id-aliaser            # core only
pip install "langchain-id-aliaser[langchain]"  # + adapters
```

## How it works

One dependency-free engine (`IdAliaser`) detects ids, aliases them, and restores
them. Three thin adapters share that engine so behavior is identical everywhere:

| Use case        | Adapter                                  |
| --------------- | ---------------------------------------- |
| Single call     | `wrap_model(model)`                      |
| Agent           | `AliasMiddleware`                        |
| Raw `StateGraph`| `alias_before_model` / `restore_after_model`, or `make_model_node` |

### Core

```python
from langchain_id_aliaser import IdAliaser

a = IdAliaser()                       # deterministic-hash mode, UUIDs on by default
a.register(value="cust_8f3k2", type="cust")   # alias any value, with a type prefix
a.register(pattern=r"order_\d+", type="ord")  # ...or a custom pattern

aliased = a.aliasify("user f47ac10b-58cc-4372-a567-0e02b2c3d479 placed order_991")
restored = a.restore(aliased)         # exact inverse; unknown aliases pass through
```

### Single call

```python
from langchain_id_aliaser import wrap_model

model = wrap_model(chat_model)        # alias in, restore out — incl. tool calls
model.invoke(messages)
```

### Agent middleware

```python
from langchain_id_aliaser import AliasMiddleware

agent = create_agent(model, tools, middleware=[AliasMiddleware()])
```

### Raw graph

```python
from langchain_id_aliaser import make_model_node

graph.add_node("model", make_model_node(chat_model))
```

## Options

- `mode="hash"` (default): deterministic alias from the id — same id → same
  alias, no stored state, stable across turns. `mode="ordinal"`: `id1, id2 …`
  (shorter, but the mapping must be reused to stay consistent).
- `strict=True`: raise `UnknownAliasError` when the model emits an alias-shaped
  token that isn't in the mapping. Default is forgiving (pass through).

## License

MIT
