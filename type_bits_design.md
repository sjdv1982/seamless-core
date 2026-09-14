Canonical JSON null (`b"null\n"`, checksum `38e0b9de…`) is accepted in every celltype and converts to itself. Resolving it as `bytes` returns `b""`; every other celltype returns `None`. Empty byte values canonicalize to that same null checksum.

# TypeBits — a bit-flag description of "not-quite Seamless types"

Status: design note. Supersedes `BufferInfo` (see [`buffer_info.py`](buffer_info.py)).
Grounded in the celltype lattice of [`conversion.py`](conversion.py).

## 1. Purpose & scope

`TypeBits` is a compact bit-flag classification of a **checksum** (equivalently,
of its buffer's content). It is "not quite" a Seamless type because it answers
two questions that a single celltype name does not:

1. **Deserializability** — *may this checksum be deserialized into a value using
   celltype X?* This is upward-closed in the subtype lattice: a buffer that
   deserializes as `python` also deserializes as `ipython`, `text`, `bytes`.
2. **Expression capability** — *can this expression be evaluated on the value?*
   Not every celltype supports integer-indexing/slicing, and not every celltype
   supports string/attribute access. `TypeBits` records which **kinds of access**
   the root value admits, so a path step can be statically rejected (`[3]` on an
   `int`, `.foo` on a list) before any buffer is fetched.

These two are orthogonal; they share one small integer (§4.1).

`TypeBits` is **self-contained**: there is no wrapping struct of side fields. The
non-boolean metadata that `BufferInfo` carried as plain fields (`length`,
`dtype`, `shape`, `members`) is folded into the word in **reduced** form —
small enums that retain exactly the resolution the queries in §§5–7 need
(`Length`, `DType`, `Rank`; `members` is dropped — §4.3). And it is **total**: a
`TypeBits` is never partially filled; every field always holds a definite value
(§8).

What `TypeBits` deliberately does **not** carry:

- **Cached conversion results** (`str2text`, `text2str`, `bytes2binary`,
  `binary2json`, `json2binary` in the old `BufferInfo`). Each is just a cheap
  **Expression** with an empty path —
  `Expression(checksum, path="", input_celltype=source, celltype=target)` —
  whose result checksum is cached in `seamless.db` under
  `database_key = (input_checksum, "", source, target)`, exactly like a
  Transformation. They are recovered from the Expression cache and must not be
  duplicated here. `TypeBits` answers *whether* a conversion is feasible; the
  Expression cache holds *what* the resulting checksum is.
- **Unbounded metadata**. The exact `length`, the full numpy `dtype` string, the
  literal `shape` tuple, and the deep-cell member *count* are not stored. Their
  query-relevant residue survives as the `Length`/`DType`/`Rank` enums (§4.3);
  the precise values are read from the buffer/value on the rare path that needs
  them.

## 2. The subtype lattice and the deserializability closure

`conversion.py` already encodes the checksum-preserving subtype upcasts in
`conversion_trivial` (each pair is `(sub, super)`):

```
text  ⊂ bytes
ipython ⊂ text        python ⊂ ipython        python ⊂ text
yaml ⊂ text           plain ⊂ yaml
plain ⊂ bytes         plain ⊂ mixed
binary ⊂ mixed
str, int, float, bool ⊂ plain
```

Deserializability uses **two more edges** than the trivial-conversion relation:

```
deserializability_edges = conversion_trivial  ∪  {(binary, bytes), (mixed, bytes)}
```

`(binary→bytes)` and `(mixed→bytes)` live in `conversion_reformat` — they *may
change the checksum*, so `conversion.py` rightly refuses to call them trivial.
But every binary/mixed buffer **is** a valid bytes buffer, so for the purpose of
*deserializability* the edges hold. Precisely:

> **deserializability-closure ⊇ trivial-conversion-closure**, the surplus being
> exactly the `binary→bytes` and `mixed→bytes` reformat edges.

`bytes` is therefore the top of the deserializability lattice and is always set.
`checksum` is a special case: its *value* is a single 64-hex-digit string. The
bits still **screen** it — a `checksum` buffer is `UTF8 ∧ Length == EQ64` (§9) —
but the *sufficient* validation (are all 64 characters hex digits?) lives
elsewhere, as in `BufferInfo` today. (Deep cells are separate celltypes, not a
form of `checksum`; §9.)

## 3. Economy: store predicates, not supertypes

Legal deserializability masks are exactly the upward-closed sets of the lattice,
so supertype bits are never stored — they are *derived* by closing upward from a
small set of **discriminating predicates** (the join-irreducibles). Closing from
a single generator recovers a 4–6 element up-set:

| generator | deserializes as … |
|---|---|
| `python` | python, ipython, text, bytes |
| `int`    | int, plain, yaml, mixed, text, bytes |
| `binary` | binary, mixed, bytes |
| `seamless_mixed` | mixed, bytes |

A raw "most-specific celltype" antichain is rejected because scalars explode it
(`"42"` is most-specifically `{int, float, str}` — three incomparable
generators). Orthogonal predicates (`Kind = JSON_NUMBER` + `NUMERIC_SCALAR`)
encode the same fact in fixed width. This is the same choice `BufferInfo` already
makes with `is_json` / `json_type` / `is_json_numeric_scalar`; `TypeBits` just
makes it principled and complete.

The *minimum* of that up-set — the most specific celltype — is the buffer's
**most-informative celltype** (MIC), which §8 uses to materialize a value when
a peek is not available.

## 4. Logical predicates

Semantically, `TypeBits` is the predicates below. Physically they pack into one
small integer (§4.1), where the mutually-exclusive ones collapse into a single
enum field instead of one bit each.

- **Container kind** — a buffer is at most one of JSON / NUMPY / SEAMLESS_MIXED,
  and a JSON buffer has exactly one value kind, so all of this is **one enum**
  (`Kind`, §4.1), not separate flags: `RAW_BYTES` (non-UTF-8 bytes) or `RAW_TEXT`
  (non-JSON UTF-8 text), `NUMPY` (`.npy` magic), `SEAMLESS_MIXED` (mixed magic), or
  `JSON` *with* its value kind `object` / `array` / `string` / `number`. `UTF8`,
  `JSON`, `NUMPY`, `SEAMLESS_MIXED`, and `json_type` are derived accessors over
  `Kind` (§4.1).
- **Length** — the buffer length, **bucketed** into an enum (`Length`, §4.3) with
  breakpoints at 64 (a checksum value's hex length) and 1000 (the numeric-scalar
  gate). The only length facts the queries use are "is it the >1000 over-long
  case?" and "is it checksum-shaped (==64)?"; the bucket keeps exactly those.
- **UTF8** — `buffer.decode()` succeeds. It informed only `RAW` (text vs pure
  bytes) — `JSON ⟹ UTF8`, and `NUMPY`/`SEAMLESS_MIXED` are binary — so rather than
  a flag it is **folded into `Kind`**: `RAW` splits into `RAW_TEXT` (UTF-8) and
  `RAW_BYTES` (not), and `UTF8 = kind ≥ RAW_TEXT` is derived (§4.1).
- **NUMERIC_SCALAR** — value converts to `int`/`float` (meaningful for
  `Kind ∈ {number, string}`).
- **DType** — for `Kind = NUMPY`, the dtype *class* as an enum (`DType`, §4.3):
  `NUMERIC`, `NONNUMERIC` (bytes `S` / unicode `U` / object `O`), or `STRUCTURED`
  (a record dtype → field access by name). `NA` otherwise. This replaces both the
  old full-`dtype` field **and** the first draft's `NUMPY_STRUCTURED` flag
  (`STRUCTURED` is now a `DType` value).
- **Rank** — for `Kind = NUMPY`, the ndim **bucket** (`Rank`, §4.3): `SCALAR`
  (0-d), `D1`, `D2`, `D3PLUS`. It distinguishes the scalar case (needed by
  `binary→int/float` and by capability) and bounds the depth of a chained
  positional path (§7.2). Replaces the literal `shape` tuple.
- **NUMPY_BYTES** — single (`Rank = SCALAR`) dtype-`S` array, the one case where
  binary↔bytes *changes* the checksum (`Kind = NUMPY`). It refines
  `DType = NONNUMERIC`. It does **not** change `deserializable_as(bytes)` (still
  universally True); it only records that *value-via-`bytes`* (the `.npy` buffer)
  differs from *value-via-`binary`*`.tobytes()`.
- **SEMANTIC** — the checksum is a *semantic* (AST-derived) identity of code, not
  a literal text buffer (§4.2). This is provenance, not buffer content (§8.5).

**Expression capability needs no flag.** A *true* mixed value (JSON combined with
Numpy) always has a **container** root — a dict or a list/array, never a scalar
(a bare scalar is `plain`, a bare array is `binary`). So the mixed magic splits
into two `Kind` values, `MIXED_OBJECT` and `MIXED_ARRAY`, and the root structure
rides along inside `Kind` at no extra cost (§4.1). Every `SEQ`/`MAP` capability
then follows from `Kind` (plus `Rank`/`DType` for arrays) alone (§7) — Region 5
vanishes entirely.

**`bool` / `null` are not stored at all.** Exactly three buffers serialize to
them — `true`, `false`, `null` — so their checksums are three module constants
(`CHECKSUM_TRUE`, `CHECKSUM_FALSE`, `CHECKSUM_NULL`); membership is a direct
checksum comparison, which is why `Kind` needs no `bool`/`null` value. Their full
`TypeBits` is likewise three pre-tabulated constants, so they too are total (§8).

Dropped from the first draft: the text-language **validity** flags
(`IS_YAML`/`IS_IPYTHON`/`IS_PYTHON`) — out of scope, validity stays in
`text_validation_celltype_cache`; **`JSON_NUMERIC_ARRAY`** — saved almost
nothing; **`J_BOOL`/`J_NULL`** — the three constants above; **`SEQ`/`MAP`** (and
the `MIXED_KEYED` residue) — fully derived, the mixed-root case now living in
`Kind` (§7); **`NUMPY_STRUCTURED`** — folded into `DType.STRUCTURED` (§4.1); the
**`UTF8`** flag — folded into `Kind` by splitting `RAW` into `RAW_TEXT`/`RAW_BYTES`
(§4.1); **`members` / `DEEP`** — dropped: the member count is read by no `TypeBits`
query, and "deepness" is not a fact about the bytes — a deep cell is a *distinct
celltype* (`deepcell`/`deepfolder`/…) serialized as plain JSON, so its buffer is
already a `JSON_OBJECT`/`JSON_ARRAY` `Kind`; the deep-vs-plain identity is that
declared celltype, a higher layer, out of scope here (§9).

### 4.1 Storage scheme (enum-packed)

```python
from enum import IntEnum, IntFlag

class Kind(IntEnum):           # bits 0–3, range 0–8  (4 bits)
    RAW_BYTES    = 0           # non-UTF-8 bytes (pure binary)
    NUMPY        = 1           # .npy magic
    MIXED_OBJECT = 2           # mixed magic, dict root
    MIXED_ARRAY  = 3           # mixed magic, list/array root
    RAW_TEXT     = 4           # non-JSON UTF-8 text     ← UTF8 begins here
    JSON_OBJECT  = 5
    JSON_ARRAY   = 6
    JSON_STRING  = 7
    JSON_NUMBER  = 8

class Length(IntEnum):         # bits 4–5, range 0–3  (2 bits)
    SHORT  = 0                 # < 64 bytes
    EQ64   = 1                 # exactly 64 (checksum-digest length)
    MEDIUM = 2                 # 65 .. 1000
    LONG   = 3                 # > 1000  (numeric-scalar conversion forbidden)

class DType(IntEnum):          # bits 6–7, range 0–3  (2 bits); NA unless Kind == NUMPY
    NA          = 0
    NUMERIC     = 1            # bool / int / uint / float / complex
    NONNUMERIC  = 2            # bytes 'S', unicode 'U', object 'O'
    STRUCTURED  = 3            # record dtype with named fields

class Rank(IntEnum):           # bits 8–9, range 0–3  (2 bits); SCALAR unless Kind == NUMPY
    SCALAR = 0                 # 0-d
    D1     = 1
    D2     = 2
    D3PLUS = 3                 # 3 or more dimensions (bucket saturates)

class Flag(IntFlag):           # bits 10–12
    NUMERIC_SCALAR = 1 << 0
    NUMPY_BYTES    = 1 << 1
    SEMANTIC       = 1 << 2
```

Packed into **13 bits — a 2-byte word** (3 bits spare for future predicates):

```
bit 0–3  : Kind             (0–8)
bit 4–5  : Length           (0–3)
bit 6–7  : DType            (0–3)
bit 8–9  : Rank             (0–3)
bit 10   : NUMERIC_SCALAR
bit 11   : NUMPY_BYTES
bit 12   : SEMANTIC
```

Derived accessors (no stored bit):

```python
UTF8           = kind >= Kind.RAW_TEXT          # RAW_TEXT and every JSON_* are UTF-8
JSON           = kind >= Kind.JSON_OBJECT
RAW            = kind in (Kind.RAW_BYTES, Kind.RAW_TEXT)
json_type      = ("object", "array", "string", "number")[kind - Kind.JSON_OBJECT] if JSON else None
NUMPY          = kind == Kind.NUMPY
SEAMLESS_MIXED = kind in (Kind.MIXED_OBJECT, Kind.MIXED_ARRAY)
is_bool        = checksum in (CHECKSUM_TRUE, CHECKSUM_FALSE)
is_null        = checksum == CHECKSUM_NULL
```

Both `UTF8` and `JSON` are clean `≥` thresholds: the binary-ish kinds
(`RAW_BYTES`, `NUMPY`, `MIXED_*`) sit below `RAW_TEXT`, then the UTF-8 kinds, then
the JSON kinds, so "is UTF-8" and "is JSON" are single comparisons. The old `UTF8`
flag is gone — it was `Kind`-derived for every kind *except* `RAW`, and splitting
`RAW` into `RAW_TEXT`/`RAW_BYTES` supplies that one missing bit inside `Kind` (the
4th `Kind` bit is paid for by the dropped flag — still 13 bits total).

Determining `MIXED_OBJECT` vs `MIXED_ARRAY` is a one-token peek at the mixed
skeleton (the byte after the magic is `{` or `[`), done while classifying `Kind`
— so the capability is populated for free, not deferred. The numpy `DType`,
`Rank`, and `NUMPY_BYTES` come from the `.npy` *header* the same way (§8.3): a
peek, never a load. `Length`, `DType`, and `Rank` are the reduced forms of the
old `length`/`dtype`/`shape` fields (§4.3); `members` is dropped. Nothing lives
outside this word.

### 4.2 Semantic checksums

Code cells carry two identities: the checksum of the literal source buffer, and a
**semantic checksum** computed from the parsed AST, so that comment / whitespace /
formatting changes don't bust the transformation cache. `SEMANTIC` marks a
checksum as the latter. Consequences here:

- A semantic checksum identifies a canonical AST-derived form, so
  `deserializable_as` for it is restricted to the code celltypes (`python`,
  `ipython`, and `text`/`bytes` as their serialization).
- The literal↔semantic relationship is, like every cheap conversion in this
  design, an **Expression** (empty path) cached in `seamless.db` — not a stored
  field. `SEMANTIC` records only *which side* a checksum is on.
- It is orthogonal to `Kind` (it co-occurs with `Kind = RAW_TEXT`, UTF-8 code), so
  it is a flag, not a `Kind` value. If a deployment ever stores the marshalled AST
  *as the buffer*, that would instead warrant a new `Kind` value; layering
  `SEMANTIC` on text is the lighter option and is recommended.
- It is **provenance, not content**: no buffer inspection reveals it, so it is
  supplied by the construction path (§8.5), defaulting to `False`.

### 4.3 Reduced metadata enums

The four `BufferInfo` side fields are kept only to the resolution the queries
need, and folded into the word.

**`Length`.** The methods consult length for exactly two facts: the numeric
`> 1000` gate (§6) and "is this checksum-shaped?". 64 is the sweet spot — the
mandatory length of a `checksum`-celltype *value* (a 64-hex-digit string) and of
each leaf of a deep cell. So the bucket boundaries sit at 64 and 1000:
`SHORT (<64)`, `EQ64 (==64, the only exact bucket)`, `MEDIUM (65–1000)`,
`LONG (>1000)`. `LONG` is the over-long case that forbids numeric coercion;
`EQ64` is the buffer-length signature of checksum-shaped content. Everything
between is one undifferentiated bucket, because nothing else reads the length.

**`DType`.** Reduced to the three classes the queries distinguish —
`NUMERIC` / `NONNUMERIC` / `STRUCTURED` (plus `NA` off the numpy branch). This
absorbs `NUMPY_STRUCTURED` (now `DType.STRUCTURED`). The one precision cost of
merging `S`/`U`/`O` into `NONNUMERIC` is `binary→plain`: an object-dtype array is
infeasible (`json_encode` fails) while an `S`/`U` array is feasible-but-changing,
and the merged enum can no longer tell them apart — so that one cell becomes `?`
instead of a split `✗`/`≠` (§6.1). This is an accepted trade: `binary→plain` is a
rare conversion. `NUMPY_BYTES` still pins the scalar-`S` sub-case that the
binary↔bytes reformat hinges on.

**`Rank`.** Reduced to `SCALAR / D1 / D2 / D3PLUS` — "0-d, 1, 2, 3+ dimensions".
Beyond distinguishing the scalar case (for `binary→int/float` and capability), the
dimension count is what lets a *chained* positional path be validated statically:
a rank-`r` array admits `r` positional-item steps before bottoming out at a scalar
(§7.2). `D3PLUS` saturates: at three or more dims the exact rank is no longer
tracked, so a path deeper than three is answered `None` rather than rejected.

**`members`** is **dropped**. No `TypeBits` query reads the member count, and
"deepness" is not a fact about the bytes: a deep cell is a *distinct celltype*
(`deepcell`/`deepfolder`/…) that serializes as plain JSON, byte-identical to an
ordinary dict/list of 64-hex strings. Its buffer's `Kind` is therefore already
`JSON_OBJECT`/`JSON_ARRAY`, which is all capability needs (§7); which plain-JSON
buffers are *deep* is the cell's declared celltype — a higher layer, out of scope
here (§9).

## 5. Method 1 — `deserializable_as(celltype) -> bool` (strict, Role 1)

"Strict" = *`parse_buffer(buffer, celltype)` would succeed.* Matches
`verify_buffer_info` / `validate_buffer_info`:

Canonical null is accepted before these predicates, for every celltype. The table
below describes non-null inputs. Function pins/results apply their stricter null
boundary separately; HashType describes storage compatibility.

| celltype | predicate |
|---|---|
| `bytes` | always True (lattice top) |
| `text` | `UTF8` |
| `yaml` / `ipython` / `python` | `UTF8` (may-be; validity checked at parse time) † |
| `plain` | `Kind ≥ JSON_OBJECT` |
| `str` | `Kind ∈ {JSON_STRING, JSON_NUMBER}`, or a bool/null constant |
| `int` / `float` | `NUMERIC_SCALAR` |
| `bool` | `checksum ∈ {CHECKSUM_TRUE, CHECKSUM_FALSE}` |
| `binary` | `Kind == NUMPY` |
| `mixed` | `¬RAW` (i.e. `Kind ∉ {RAW_BYTES, RAW_TEXT}`) |
| `checksum` | `Kind == RAW_TEXT ∧ Length == EQ64` (may-be; all-hex test is value-level) ‡ |

† `yaml`/`ipython`/`python` **validity** is no longer classified (the Region-4
flags were dropped, §4), so these report only the necessary `UTF8` condition;
the exact check stays in `text_validation_celltype_cache`. A `SEMANTIC` code
checksum is deserializable as its code celltype.

‡ A `checksum` buffer is a bare 64-byte non-JSON UTF-8 digest
(`Kind = RAW_TEXT ∧ Length = EQ64`); the bits give that necessary screen, the exact
all-hex test stays value-level. Deep cells are separate celltypes (§9), not a
form of `checksum`.

`mixed = ¬RAW` falls straight out: `NUMPY`, `SEAMLESS_MIXED`, and every `JSON_*`
kind are mixed-deserializable; only `RAW_BYTES` (pure bytes) and `RAW_TEXT`
(non-JSON text) are not.

## 6. Method 2 — `convertible_to(source, target, *, preserving=True)`

The conversion-aware query. It assumes a fixed `source` celltype (the buffer may
be deserializable as several) and reports feasibility of `source → target`,
mirroring `conversion.py`'s categories:

- `preserving=True` — restrict to checksum-preserving conversions
  (`conversion_trivial`, plus `conversion_reinterpret` that the bits confirm).
- `preserving=False` — also allow checksum-changing-but-feasible
  (`conversion_reformat`, `conversion_possible`, value conversions).

Return contract, inherited from `convert_from_buffer_info` but **without** the
cached-checksum branch (that now comes from the Expression cache, §1):

| result | meaning |
|---|---|
| `True`  | feasible and (if `preserving`) checksum-preserving |
| `-1`    | surely feasible, but resulting checksum unknown from bits |
| `None`  | may or may not be feasible (needs the value) |
| `False` | surely infeasible (`conversion_forbidden`, or bits contradict it) |

The actual resulting checksum, when it changes, is **not** returned here — it is
`Expression(checksum, "", source, target).result` from `seamless.db`.

**Totality changes the `?`/`None` budget.** Because `TypeBits` is total (§8),
every `?`/`None` that `BufferInfo` returned because a *field was missing*
disappears: `UTF8`, `Kind`, `DType`, `Rank` are always known. What stays `?` is
only genuine *value-level* uncertainty — text-language validity, integer-vs-float
canonicality, and the merged-dtype `binary→plain` ambiguity (§4.3).

### 6.1 The full table

The source-validity gate is applied first: if `deserializable_as(source)` is
provably `False`, raise `SeamlessConversionError` (as `convert_from_buffer_info`
does today). Otherwise each ordered pair classifies into one of four outcomes,
which the two `preserving` modes read off as follows:

| glyph | meaning | `preserving=False` | `preserving=True` |
|---|---|---|---|
| `=` | feasible, checksum **preserved** (result == input checksum) | `True` | `True` |
| `≠` | feasible, checksum **changes** (result via Expression cache) | `-1` | `False` |
| `?` | **maybe** — needs the value | `None` | `None` |
| `✗` | provably **infeasible** | `False` | `False` |

`preserving=True` is the stricter query: every definite `≠` becomes `False`,
while `=`/`?`/`✗` are unchanged. A `?` under `preserving=True` means "might be
preserving."

**Trivial** — all `=` (checksum-preserving by definition):
`text→bytes`, `ipython→text`, `python→text`, `python→ipython`, `yaml→text`,
`plain→yaml`, `plain→bytes`, `binary→mixed`, `plain→mixed`, `str→plain`,
`int→plain`, `float→plain`, `bool→plain`.

**Forbidden** — all `✗`:
`python→{yaml,int,float,bool}`, `ipython→{yaml,int,float,bool}`,
`yaml→{python,ipython}`, `int→{python,ipython}`, `float→{python,ipython}`,
`bool→{python,ipython}`.

**Reinterpret** (target-validated; preserving whenever feasible). Totality
decides all of these except the value-level validity triple:

| pair | result |
|---|---|
| `bytes→text` | `=` if `UTF8` else `✗` |
| `bytes→plain` | `=` if `JSON` else `✗` (`RAW`/`NUMPY`/`SEAMLESS_MIXED` are not JSON) |
| `text→yaml` / `text→ipython` / `text→python` | `?` — validity is value-level |
| `mixed→binary` | `=` if `NUMPY` else `✗` |
| `mixed→plain` | `=` if `JSON` else `✗` |

`text→{yaml,ipython,python}` stay `?`: code/YAML **validity** is out of scope
(Region 4 dropped, §4), so the bits can neither confirm nor refute them — the
same `?` `BufferInfo` returns. They resolve at parse time via
`text_validation_celltype_cache`. The other four rows no longer return the
field-absent `?` `BufferInfo` did, because `UTF8`/`JSON`/`NUMPY` are always known.

**Reformat** (always feasible given a valid source — never `?` or `✗`):

| pair | `=` when | otherwise |
|---|---|---|
| `bytes→binary` | `NUMPY` | `≠` (wrap as `S`-array) |
| `bytes→mixed` | `NUMPY ∨ SEAMLESS_MIXED` | `≠` |
| `binary→bytes` | `¬NUMPY_BYTES` | `≠` (`value.tobytes()`) |
| `mixed→bytes` | `¬NUMPY ∨ (NUMPY ∧ ¬NUMPY_BYTES)` | `≠` |
| `plain→text` | `Kind ∉ {JSON_STRING}` (object/array/number, and bool/null constants) | `≠` (strip quotes) |
| `text→plain` | `JSON` | `≠` (wrap text as JSON string) |
| `text→str` | — | always `≠` (add quotes) |
| `str→text` | — | always `≠` (remove quotes) |
| `yaml→plain` | — | always `≠` (re-dump as canonical JSON) |
| `ipython→python` | — | always `≠` (expand magics) |
| `int/float/bool→str` | — | always `≠` |
| `int↔float`, `int↔bool`, `float↔bool` (6 pairs) | — | always `≠` † |

`NUMPY_BYTES` already means "scalar dtype-`S` array" (`DType = NONNUMERIC`,
`Rank = SCALAR`), so `binary→bytes`/`mixed→bytes` read it directly — the old
`shape`/`dtype` lookups collapse into the one flag.

**Possible** (may change *and* not guaranteed). For numeric targets,
`Length == LONG` ⟹ `✗` first.

| pair(s) | rule |
|---|---|
| `plain/mixed→str` | `Kind = JSON_STRING` → `=`; `Kind = JSON_NUMBER` (and bool/null constants) → `≠`; `Kind ∈ {JSON_OBJECT, JSON_ARRAY}` → `✗` |
| `plain/mixed→float` | `Kind ∈ {JSON_OBJECT, JSON_ARRAY}` or `Length == LONG` → `✗`; `NUMERIC_SCALAR` → `≠`; else `✗` |
| `plain/mixed→int` | as `→float`, but a feasible numeric is `?` (integrality of the value is value-level), not `≠` |
| `str→int/float` | `Length == LONG` → `✗`; `NUMERIC_SCALAR` → `≠` † (`→int` stays `?` on integrality); else `✗` |
| `binary→int/float` | `DType.NUMERIC ∧ Rank.SCALAR` → `≠`; otherwise `✗` |

`bool`/`null` *targets* are decided by direct comparison against the three
constant checksums (§4), plus numeric coercion (`number→bool` via
`NUMERIC_SCALAR`); they need no per-pair bit logic. The reduced `Kind` no longer
distinguishes integer-canonical from float-canonical JSON numbers, so the old
`json_type == target → =` shortcut is gone — these pairs are `≠`/`?`, never the
questionable preserving `True` (see the † note).

`plain/mixed→str` is a gain over `convert_from_buffer_info`, which returns `None`
for it (the `str` target skips its `int/float/bool` shortcut), whereas `Kind`
decides it here.

**Values, non-checksum** (need the value or its cached Expression):

| pair | rule |
|---|---|
| `binary→plain` | `DType ∈ {NUMERIC, STRUCTURED}` → `≠` (`json_encode`); `DType = NONNUMERIC` → `?` (object would be `✗`, bytes-`S`/`U` `≠` — the merge can't tell, §4.3) |
| `plain→binary` | `Kind = JSON_OBJECT` → `✗`; `Kind = JSON_NUMBER` (scalar → 0-d array) → `≠`; `Kind = JSON_ARRAY` → `?` (homogeneous-numeric no longer tracked — `JSON_NUMERIC_ARRAY` dropped); else `?` |

**Checksum** — *not* a blanket `?`. A `checksum` value is a single bare 64-hex-digit
string, so its buffer is `Kind == RAW_TEXT ∧ Length == EQ64` (a digest with
`a`–`f` is not JSON), which the bits screen directly (§9):

- `(*, checksum)` → `✗` when the buffer fails that screen (numpy/mixed, wrong-length
  text, a quoted/scalar JSON string or number); `?` once it passes (the exact
  all-hex test is value-level).
- `(checksum, *)` → the source is a 64-hex string `conversion.py` may promote to
  any celltype; feasibility follows the text/str rules, but the result checksum is
  value-level — recovered as an Expression in `seamless.db` (`-1`/`?`).

Deep cells are **separate** celltypes, not a dict/list form of `checksum`; their
plain-JSON buffers are classified by `Kind` like any other and are out of scope
for this base-celltype table (§9).

**Equivalent / chain** (structural redirects, resolved before classification):

- `conversion_equivalent[(A,B)]` → evaluate the table at the canonical pair.
- `conversion_chain[(A,C)] = B` → multi-hop `A→B→C`; `convertible_to` raises
  ("handled upstream"), exactly as `convert_from_buffer_info` does today.

> **† Divergence from current `convert_from_buffer_info`.** Two families return
> `True` (preserving) there but `≠`/`?` here:
> (a) numeric reformats `int↔float` (and `number↔bool` via the constants);
> (b) `str→int/float`. In both, the source buffer is reserialized — `3`→`3.0`,
> `true`→`1`, the quoted `"3"`→`3` — so the checksum changes; `True` was only
> valid when the source buffer already equalled the target's canonical buffer.
> The reduced `Kind` (no int-vs-float split) deliberately drops the bit that
> would confirm that match, so the questionable `True` is no longer reachable —
> the reduction *enforces* the caution the old code only documented.

## 7. Method 3 — expression capability (Role 4)

A path step from [`expression_class.py`](expression_class.py) is either an
**item** (`.name` / `["key"]`, built by `Expression.item`) or a **slice**
(`[a:b:c]`, `Expression.slice`). An item with an identifier key is
mapping/attribute access (**MAP**); an item with an integer key, and every slice,
is positional (**SEQ**).

Capability is **relative to the expression's `source_celltype`** (the `celltype`
field of the `Expression`), not to the value's intrinsic type — the path operates
on the buffer *read as that celltype*. So it is a function of `(ti, source)`, gated
by `deserializable_as(source)` (§5):

```python
FLAT_SEQ = {"text", "str", "python", "ipython", "yaml"}   # UTF-8 character sequences

def capabilities(ti, source) -> set[str]:    # ⊆ {"SEQ","MAP"}; assumes deserializable_as(source)
    if source == "bytes":      return {"SEQ"}      # byte sequence — available for ANY checksum
    if source in FLAT_SEQ:     return {"SEQ"}      # character sequence
    if source == "binary":
        caps = {"SEQ"} if ti.rank != Rank.SCALAR else set()
        if ti.dtype == DType.STRUCTURED: caps.add("MAP")   # field access by name
        return caps
    if source in ("plain", "mixed"):
        k = ti.kind
        if k in (Kind.JSON_OBJECT, Kind.MIXED_OBJECT): return {"MAP"}
        if k in (Kind.JSON_ARRAY,  Kind.MIXED_ARRAY):  return {"SEQ"}
        if k == Kind.JSON_STRING:                       return {"SEQ"}   # characters
        return set()                                    # JSON_NUMBER → scalar
    return set()                                  # int / float / bool / checksum → scalar
```

The **structural** sources (`binary`/`plain`/`mixed`) read the access kind off
`Kind`/`Rank`/`DType` — Region 5 stays empty, the mixed root riding in `Kind` (§4).
The **flat** sources are the practical point: **`bytes` slices any checksum** (the
buffer is a byte sequence) and `text`/`str`/code slice any UTF-8 buffer, both
independent of `Kind`. So a `JSON_NUMBER` — a scalar, no SEQ when read as `plain` —
is *still* sliceable by an expression whose `source_celltype` is `bytes`. The
intrinsic, MIC-relative capability (`capabilities(ti, MIC(ti))`) describes the
value's *own* structure; an expression may opt into a flatter view where slicing
always works.

### 7.1 Item-type accessors

What a positional step *yields* is likewise `(ti, source)`-relative, so an
expression can be typed — and statically accepted or rejected — one step at a time.
Each is `True` / `False` / `None` (`None` = the bits can't say; the child must be
materialized):

| accessor | `True` | `None` | else `False` |
|---|---|---|---|
| `has_slicing(ti, source)` | `"SEQ" in capabilities(ti, source)` | — (structural) | otherwise |
| `has_numeric_items(ti, source)` | `source == "bytes"` (`b[i]` is an int); or `source == "binary" ∧ DType.NUMERIC ∧ Rank ≥ D1` | `source ∈ {plain, mixed} ∧ Kind ∈ {JSON_ARRAY, MIXED_ARRAY}` (element type untracked) | flat-text / `JSON_STRING` / structured / scalar |
| `has_string_items(ti, source)` | `source ∈ FLAT_SEQ` (characters); or `source ∈ {plain, mixed} ∧ JSON_STRING` | `source == "binary" ∧ DType.NONNUMERIC` (`S`/`U` strings, `O` not); JSON/mixed arrays | `bytes` (ints) / numeric / structured / scalar |

`has_slicing` is total. The item-type pair is `None` exactly where the dropped
JSON-array element typing or the merged non-numeric dtype (§4.3) hides the element
type — resolvable only by materializing the child.

### 7.2 Rank bounds the chain depth (`source == "binary"`)

For a `binary` source, `Rank` lets a *multi-step* positional path be checked
without fetching the value. A positional **item** step lowers an array's rank by
one; a **slice** preserves it. So a path with `k` positional-item steps is valid
iff `k ≤ rank`:

| `Rank` | positional-item steps statically allowed |
|---|---|
| `SCALAR` | 0 — any `[i]`/`[:]` on the root is rejected |
| `D1` | 1 — `a[i]` ok, `a[i][j]` rejected |
| `D2` | 2 |
| `D3PLUS` | ≥3 — confirmable up to 3; a 4th positional item is `None` (the bucket saturated) |

Slices never exhaust the budget (they preserve rank). The flat sources have trivial
depth: `bytes[i]`/`text[i]` yields a scalar (one positional step, then stop), though
slices chain freely. For JSON/mixed *nesting* (a list of lists, a dict of dicts) the
bits carry no depth — only `Kind`, which types the **first** step; each deeper step
needs the child's `TypeBits`, i.e. a materialization. Numpy is the one container
whose entire chain depth is static, and that is exactly the `1/2/3+` resolution
`Rank` keeps.

### 7.3 Why the earlier "irreducible bit" claim fails

My first draft kept a `MIXED_KEYED` flag, justified thus: two `SEAMLESS_MIXED`
checksums, one rooted `{...}` and one `[...]`, are identical in every other field
yet differ in MAP vs SEQ. That holds only under an *impoverished* `Kind` that
lumps both mixed roots into one value. Since a true mixed root is **always** a
dict or a list/array (§4), splitting the magic into `MIXED_OBJECT`/`MIXED_ARRAY`
moves that one bit into `Kind` for free (`Kind` had spare range), and the two
checksums now differ in `Kind`. The information is real, but it is a `Kind`
distinction, not a separate flag — so Region 5 is empty. Deep cells need no bit
either: a deep cell is a distinct celltype that serializes as a plain JSON
dict/list, so its buffer's `Kind` is already `JSON_OBJECT`/`JSON_ARRAY`
(→ MAP/SEQ) and its deep-vs-plain identity (a celltype fact, §9) is irrelevant to
capability.

## 8. Totality — no partial definition, and the two fill procedures

### 8.1 The completeness invariant

`BufferInfo` permitted every field to be `None`: a value could be
half-classified, and queries returned `None`/`?` whenever a field they needed
happened to be absent. `TypeBits` **forbids** this. A `TypeBits` is either
*absent* (not yet computed for a checksum) or **total**: every field — `Kind`,
`Length`, `DType`, `Rank`, and all four flags — holds a definite value.
"Not applicable" is itself a definite value (`DType.NA`, `Rank.SCALAR` for
non-arrays, the flags `False`), never a stand-in for "unknown". It is computed
**once, when the checksum is first computed** (§8.4) — where the buffer is always
in hand and a typed value usually is. There is no setter that fills one field and
leaves another open; a `TypeBits` is produced whole, each field drawn from its
cheapest authoritative source (§8.3–8.4) and stored only once *confirmed*, never as
a guess (§8.6).

The payoff is §6's collapsed `?` budget: the only residual `?`/`None` are
genuinely value-level, never field-absence.

### 8.2 The most-informative celltype (MIC)

The celltypes a buffer deserializes as form an up-set in the lattice (§2); its
**minimum** — the most specific celltype — carries the most structure. The MIC is
a **total function of `Kind`**: every one of the nine `Kind`s has exactly one MIC,
and a buffer has exactly one `Kind`, so look up the row and stop — never a priority
scan, never a tie.

Each `Kind` needs only **one** MIC. The two scalar cases that looked like
antichains collapse: `JSON_NUMBER`'s MIC is `float` — choosing `int` instead
changes no stored bit (both set `NUMERIC_SCALAR`, and int-vs-float canonicality is
not tracked, §6.1); and a `JSON_STRING`'s `NUMERIC_SCALAR` is a `float(value)`
*probe on the `str`*, not a second deserialization. So "multiple MICs" is never
actually needed.

| `Kind` | MIC | reads off the value |
|---|---|---|
| `RAW_BYTES` | `bytes` | — (all fields already peeked) |
| `RAW_TEXT` | `text` | — |
| `NUMPY` | `binary` | `dtype`→`DType`, `shape`→`Rank`, `NUMPY_BYTES` |
| `MIXED_OBJECT`/`MIXED_ARRAY` | `mixed` | root already in `Kind` |
| `JSON_OBJECT`/`JSON_ARRAY` | `plain` | container ⟹ `NUMERIC_SCALAR = False` |
| `JSON_STRING` | `str` | `NUMERIC_SCALAR ← float(value)` probe |
| `JSON_NUMBER` | `float` | `NUMERIC_SCALAR = True` |

The three `bool`/`null` constants are recognised by checksum and carry
pre-tabulated `TypeBits` (§4) — they need no MIC.

### 8.3 Procedure A — the cheap peek (`O(1)`; never `json.loads`)

A peek inspects the raw bytes **without deserializing**, and in particular **never
calls `json.loads`** — on a gigabyte buffer that allocates a gigabyte-plus parse
tree. It reads only a bounded amount, and is *authoritative* for the fields it can
settle that way:

0. If the checksum equals one of the three bool/null constants → its pre-tabulated
   `TypeBits` (§4). An `O(1)` 32-byte compare.
1. `len(buffer)` → `Length`. `O(1)`.
2. The first ≤14 **magic** bytes:
   - `\x93NUMPY` → `Kind = NUMPY`. The `.npy` header is a short ASCII dict at a
     fixed offset (6 magic + 2 version + 2/4 length bytes), read **without touching
     the array payload** — still `O(1)`: the `descr` dtype letter gives `DType`
     (`b/i/u/f/c`→`NUMERIC`, `S/U/O`→`NONNUMERIC`, a `descr` field list→
     `STRUCTURED`), `len(shape)` gives `Rank` (capped at `D3PLUS`), and an `S`-dtype
     with `shape == ()` sets `NUMPY_BYTES`.
   - `\x94SEAMLESS-MIXED` → first skeleton token → `MIXED_OBJECT` / `MIXED_ARRAY`;
     `DType = NA`, `Rank = SCALAR`.
3. Otherwise a bounded **head/tail token** read (skip leading/trailing whitespace):
   a JSON document must open with one of `{ [ " - 0-9` and close with the matching
   `} ] "` / digit. A first byte outside that set, or head/tail that don't pair, is
   **decisive — not JSON → `RAW`** (`RAW_TEXT` if the bytes are UTF-8, else
   `RAW_BYTES` — the one `O(n)` bit, settled in §8.4). A pairing instead yields only
   a *candidate* `JSON_OBJECT`/`ARRAY`/`STRING`/`NUMBER`: the interior may still be
   malformed, so the candidate is **not stored** until confirmed (§8.4/§8.6).
4. `SEMANTIC ←` provenance (§8.5); a peeked literal buffer is always `False`.

So the cheap peek is authoritative, at `O(1)`, for `Length`, the two magic kinds
with **all** their numpy sub-fields, and the **negative** "not JSON → `RAW`"
verdict. It leaves exactly two things open: for a `RAW` buffer, the
`RAW_TEXT`-vs-`RAW_BYTES` split (an `O(n)` UTF-8 validation — formerly the `UTF8`
flag); and for a JSON candidate, its *confirmation* and `NUMERIC_SCALAR`. Both are
settled for free when a value is in hand (§8.4), and only otherwise — as a last
resort — by an `O(n)` pass.

### 8.4 Procedure B — read off the value in hand; the fill priority

`TypeBits` totality is computed **once, when the checksum is first computed**. At
that moment the buffer is present by construction, and — because a checksum is
normally taken right after serializing a value, or right before/after deserializing
one — **a typed value is usually present too**. Reading the fields off that value
is the cheapest authoritative route for exactly what the cheap peek left open: the
JSON family and the `RAW_TEXT`-vs-`RAW_BYTES` split.

- A value whose celltype is the MIC (or any `plain`/`text`/`str`/number value)
  settles `Kind`/`json_type`/`UTF8`/`NUMERIC_SCALAR` from `type(value)` — **no
  parse**, so §8.3's would-be `json.loads` is skipped entirely.
- The value need **not** be the MIC. A *less*-informative value still settles every
  field at or above its own celltype; only strictly-lower fields fall back to the
  peek — and for the numpy sub-fields that fallback is the `O(1)` header anyway, so
  even holding the buffer merely as `bytes` loses nothing.
- **Materializing** into the MIC (an actual `parse_buffer`) is the **last resort** —
  reached only when no value is in hand *and* a JSON candidate survived §8.3. Each
  `Kind` has a single MIC (§8.2), so this is **one** deserialization; a
  `JSON_STRING`'s `NUMERIC_SCALAR` rides on that same `str` via `float(value)`.

Putting §8.3 and this together, a total `TypeBits` is filled by a per-field race to
the cheapest authoritative source, run once:

1. `Length`, the magic kinds, the numpy sub-fields, and the `not-JSON → RAW`
   verdict — from the `O(1)` peek, always.
2. The JSON family and the `RAW` text/bytes split — from an in-hand value if present
   (free); else from the head/tail filter, which usually **discards** JSON (→ `RAW`).
3. **Only** a surviving JSON candidate with no value in hand forces the `O(n)` last
   resort: one UTF-8 validation and/or one streaming parse to confirm `Kind` and read
   `NUMERIC_SCALAR`. Rare path, not the default.

`SEMANTIC` is filled by neither route — it is provenance (§8.5).

### 8.5 `SEMANTIC` is provenance, not content

No buffer inspection can reveal whether a checksum is a literal source buffer or
the AST-canonical form of code — they are *different* checksums. So `SEMANTIC` is
supplied by the construction path: peek/materialization of an ordinary buffer
always sets it `False`; the AST-canonicalizer, when it emits the semantic
checksum, constructs *that* checksum's `TypeBits` with `SEMANTIC = True` and fills
the remaining fields by peeking the canonical text buffer (`Kind = RAW_TEXT`).
Either way the field is definite — totality holds.

### 8.6 Agreement

The peek and the value are two sources for the same total `TypeBits`; where both
can settle a field they must agree — the natural correctness test (a cheap audit:
re-peek a stored `TypeBits`'s buffer and compare). Two consequences sharpen "no
partial definition":

- A field is always filled by the *cheapest source that can settle it* — never left
  open, but also never stored as a **guess**. The peek's JSON *candidate* (§8.3) is
  promoted to a stored `Kind` only once a value or an `O(n)` parse confirms it; an
  unconfirmed candidate is not yet a `TypeBits`.
- A `TypeBits` whose `Kind` disagrees with a re-peek of its own buffer is, by
  definition, ill-formed.

## 9. `checksum` celltype

A `checksum` *value* is a single 64-hex-digit string
([`checksum_class.py`](../checksum_class.py) `.hex()` — a 32-byte digest). Its
buffer is therefore a bare, non-JSON UTF-8 string of exactly that length:
`Kind == RAW_TEXT ∧ Length == EQ64` (a digest containing `a`–`f` does not parse
as JSON). This is exactly why §4.3 spends an exact bucket on 64. So `checksum` is
a real may-be predicate, not a blanket out-of-scope `?`:

- `deserializable_as(checksum)` → may-be when `Kind == RAW_TEXT ∧ Length == EQ64`;
  the residual "are all 64 characters hex digits?" test is value-level, in the same
  spirit as text-language validity (§5). Otherwise `✗`.
- `convertible_to(·, checksum)` / `convertible_to(checksum, ·)` are screened by the
  same predicate (§6.1): `✗` when a buffer cannot be a 64-hex digest, `?` once it
  passes, with the result checksum recovered as an Expression (§1).

What stays out of scope is only the *sufficient* hex validation — never the
necessary screen the bits now provide. This matches today's `validate_buffer_info`,
which already requires `is_utf8` for `checksum` and rejects dict/list/number
buffers; `TypeBits` adds the `Kind == RAW_TEXT ∧ Length == EQ64` half `BufferInfo`
left implicit.

**Deep cells are not `checksum`.** A deep cell (`deepcell`, `deepfolder`, `folder`,
`module`) is a *distinct, explicit* celltype in Seamless 1.x — not a "dict/list
form" of `checksum`, and not the legacy "hash pattern" attached to an ordinary
cell. It serializes as plain JSON (it maps to `plain` in
[`buffer_class.py`](../buffer_class.py)), so its buffer is an ordinary
`JSON_OBJECT`/`JSON_ARRAY` `Kind` — classified like any other plain buffer, with
its 64-hex leaves landing in `EQ64`. Which plain-JSON buffers are *deep* is the
cell's declared celltype, a layer above this base-celltype classification, and out
of scope here just as the deep celltypes are absent from `conversion.py`'s
13-celltype lattice.

## 10. Migration from `BufferInfo`

| `BufferInfo` field | becomes |
|---|---|
| `is_utf8` | **derived** `Kind ≥ RAW_TEXT` — no stored flag; the `RAW_BYTES`/`RAW_TEXT` split + JSON kinds supply it (§4.1) |
| `is_json` | `Kind ≥ JSON_OBJECT` |
| `json_type` | `Kind` (`JSON_OBJECT`..`JSON_NUMBER`, 5–8); `bool`/`null` via the three checksum constants (§4) |
| `is_json_numeric_scalar` | `Flag.NUMERIC_SCALAR` |
| `is_json_numeric_array` | **dropped** (saved little) |
| `is_numpy` | `Kind == NUMPY` |
| `is_seamless_mixed` | `Kind ∈ {MIXED_OBJECT, MIXED_ARRAY}` |
| mixed root (dict vs list) | folded into `Kind` (`MIXED_OBJECT`/`MIXED_ARRAY`) — no flag (§7) |
| `dtype` | **`DType` enum** (`NA`/`NUMERIC`/`NONNUMERIC`/`STRUCTURED`); absorbs the old `NUMPY_STRUCTURED` flag (§4.3) |
| `shape` | **`Rank` enum** (`SCALAR`/`D1`/`D2`/`D3PLUS`) (§4.3) |
| `length` | **`Length` enum** (`SHORT`/`EQ64`/`MEDIUM`/`LONG`); the `>1000` numeric gate is now `Length == LONG` (§4.3) |
| `members` | **dropped** (count read by no query; deep cells are distinct celltypes serialized as plain JSON, §4.3) |
| `NUMPY_BYTES` | retained as `Flag.NUMPY_BYTES` (scalar dtype-`S`; refines `DType = NONNUMERIC`) |
| `str2text`, `text2str`, `bytes2binary`, `binary2json`, `json2binary` | **removed** — empty-path Expressions in `seamless.db`, keyed `(checksum, "", source, target)` |
| yaml/ipython/python validity (`text_validation_celltype_cache`) | **not ported** — stays where it is (out of scope) |
| *(new)* `SEMANTIC` | AST-identity of code (§4.2), provenance (§8.5) |
| *(new invariant)* any-field-`None` | **forbidden** — `TypeBits` is total (§8); the `None`-from-missing-field branches collapse |

`validate_buffer_info` / `verify_buffer_info` / `convert_from_buffer_info`
collapse into queries over `TypeBits` (§§5–7) plus Expression-cache lookups —
one source of truth instead of scattered booleans plus cached fields.

## 11. Sketch

```python
from enum import IntEnum, IntFlag

class Kind(IntEnum):
    RAW_BYTES = 0; NUMPY = 1; MIXED_OBJECT = 2; MIXED_ARRAY = 3; RAW_TEXT = 4
    JSON_OBJECT = 5; JSON_ARRAY = 6; JSON_STRING = 7; JSON_NUMBER = 8

class Length(IntEnum):
    SHORT = 0; EQ64 = 1; MEDIUM = 2; LONG = 3

class DType(IntEnum):
    NA = 0; NUMERIC = 1; NONNUMERIC = 2; STRUCTURED = 3

class Rank(IntEnum):
    SCALAR = 0; D1 = 1; D2 = 2; D3PLUS = 3

class Flag(IntFlag):
    NUMERIC_SCALAR = 1 << 0; NUMPY_BYTES = 1 << 1; SEMANTIC = 1 << 2

def UTF8(k):  return k >= Kind.RAW_TEXT         # derived; RAW_TEXT and every JSON_*

# the three singleton buffers — bool/null need no bit, but get a total TypeBits
CHECKSUM_TRUE  = checksum_of(b"true\n")
CHECKSUM_FALSE = checksum_of(b"false\n")
CHECKSUM_NULL  = checksum_of(b"null\n")


def deserializable_as(ti, checksum, celltype: str) -> bool:
    if checksum == CHECKSUM_NULL: return True  # storage, not function boundary
    k = ti.kind
    if celltype == "bytes":  return True
    if celltype == "text":   return UTF8(k)
    if celltype in ("yaml", "ipython", "python"):
        return UTF8(k)                          # may-be; validity checked at parse time
    if celltype == "plain":  return k >= Kind.JSON_OBJECT
    if celltype == "str":
        return k in (Kind.JSON_STRING, Kind.JSON_NUMBER) or \
               checksum in (CHECKSUM_TRUE, CHECKSUM_FALSE, CHECKSUM_NULL)
    if celltype in ("int", "float"):
        return bool(ti.flag & Flag.NUMERIC_SCALAR)
    if celltype == "bool":
        return checksum in (CHECKSUM_TRUE, CHECKSUM_FALSE)
    if celltype == "binary": return k == Kind.NUMPY
    if celltype == "mixed":  return k not in (Kind.RAW_BYTES, Kind.RAW_TEXT)
    if celltype == "checksum":                  # single-checksum screen (§9)
        return k == Kind.RAW_TEXT and ti.length == Length.EQ64
    return False


FLAT_SEQ = {"text", "str", "python", "ipython", "yaml"}

def capabilities(ti, source) -> set[str]:       # §7 — relative to source_celltype
    if source == "bytes":  return {"SEQ"}
    if source in FLAT_SEQ: return {"SEQ"}
    if source == "binary":
        caps = {"SEQ"} if ti.rank != Rank.SCALAR else set()
        if ti.dtype == DType.STRUCTURED: caps.add("MAP")
        return caps
    if source in ("plain", "mixed"):
        k = ti.kind
        if k in (Kind.JSON_OBJECT, Kind.MIXED_OBJECT): return {"MAP"}
        if k in (Kind.JSON_ARRAY,  Kind.MIXED_ARRAY):  return {"SEQ"}
        if k == Kind.JSON_STRING:                       return {"SEQ"}
        return set()
    return set()
```

## 12. Enumeration of the valid `TypeBits`

The raw field cross-product (`9 × 4 × 4 × 4 × 2 × 2 × 2`) is mostly unreachable;
the **cross-field invariants** below define which words are valid. They are the
well-formedness predicate an implementation should assert (it is exactly "this
`TypeBits` equals the peek of *some* buffer").

**Invariants** (a `TypeBits` is valid iff all hold):

- `DType ≠ NA  ⟺  Kind == NUMPY`. (Non-numpy ⟹ `DType = NA`.)
- `Rank ≠ SCALAR  ⟹  Kind == NUMPY`. (Non-numpy ⟹ `Rank = SCALAR`.)
- `NUMPY_BYTES  ⟹  Kind == NUMPY ∧ DType == NONNUMERIC ∧ Rank == SCALAR`.
- `Kind == JSON_NUMBER  ⟹  NUMERIC_SCALAR`; and `NUMERIC_SCALAR  ⟹  Kind ∈ {JSON_NUMBER, JSON_STRING}`.
- `SEMANTIC  ⟹  Kind == RAW_TEXT`.
- `UTF8` is derived (`Kind ≥ RAW_TEXT`), never independent.
- `Length` is free per the §12.2 reachability rule.

### 12.1 Structural witness classes

Holding `Length` aside (§12.2), the invariants leave **26** structural classes.
Each row is one distinct `TypeBits`, with a buffer (or a snippet that builds one);
`np`/`Buffer` are `numpy` / `seamless.Buffer`. `NUM`/`NB`/`SEM` =
`NUMERIC_SCALAR`/`NUMPY_BYTES`/`SEMANTIC` (blank = `False`).

| # | Kind | DType | Rank | NUM | NB | SEM | example buffer |
|---|---|---|---|---|---|---|---|
| 1 | `RAW_BYTES` | NA | SCALAR | | | | `b"\x00\xff\xfe"` (not UTF-8) |
| 2 | `RAW_TEXT` | NA | SCALAR | | | | `b"hello, world"` (UTF-8, not JSON) |
| 3 | `RAW_TEXT` | NA | SCALAR | | | ✓ | semantic checksum of `ast.parse("x = 1")` — `SEMANTIC` set by the producer (§4.2) |
| 4 | `NUMPY` | NUMERIC | SCALAR | | | | `Buffer(np.array(3.0), "binary")` (0-d) |
| 5 | `NUMPY` | NUMERIC | D1 | | | | `Buffer(np.array([1,2,3]), "binary")` |
| 6 | `NUMPY` | NUMERIC | D2 | | | | `Buffer(np.zeros((2,3)), "binary")` |
| 7 | `NUMPY` | NUMERIC | D3PLUS | | | | `Buffer(np.zeros((2,3,4)), "binary")` |
| 8 | `NUMPY` | NONNUMERIC | SCALAR | | ✓ | | `Buffer(np.array(b"abc"), "binary")` (0-d `S` → binary↔bytes special case) |
| 9 | `NUMPY` | NONNUMERIC | SCALAR | | | | `Buffer(np.array("abc"), "binary")` (0-d `<U3`); or `np.array(None, dtype=object)` |
| 10 | `NUMPY` | NONNUMERIC | D1 | | | | `Buffer(np.array([b"a", b"bb"]), "binary")` (`S2`, non-scalar ⟹ `NB=False`) |
| 11 | `NUMPY` | NONNUMERIC | D2 | | | | `Buffer(np.array([["a"],["b"]]), "binary")` (`U`) |
| 12 | `NUMPY` | NONNUMERIC | D3PLUS | | | | `Buffer(np.array([b"x"]*8).reshape(2,2,2), "binary")` |
| 13 | `NUMPY` | STRUCTURED | SCALAR | | | | `Buffer(np.array((1,2.0), dtype=[("a","<i4"),("b","<f8")]), "binary")` |
| 14 | `NUMPY` | STRUCTURED | D1 | | | | same dtype, `shape (2,)` |
| 15 | `NUMPY` | STRUCTURED | D2 | | | | same dtype, `shape (2,2)` |
| 16 | `NUMPY` | STRUCTURED | D3PLUS | | | | same dtype, `shape (2,2,2)` |
| 17 | `MIXED_OBJECT` | NA | SCALAR | | | | `Buffer({"a": np.arange(3)}, "mixed")` (dict root) |
| 18 | `MIXED_ARRAY` | NA | SCALAR | | | | `Buffer([np.arange(3), {"x": 1}], "mixed")` (list root) |
| 19 | `JSON_OBJECT` | NA | SCALAR | | | | `b'{"a": 1}'` |
| 20 | `JSON_ARRAY` | NA | SCALAR | | | | `b'[1, 2, 3]'` |
| 21 | `JSON_STRING` | NA | SCALAR | | | | `b'"hello"'` (non-numeric) |
| 22 | `JSON_STRING` | NA | SCALAR | ✓ | | | `b'"42"'` (numeric string) |
| 23 | `JSON_NUMBER` | NA | SCALAR | ✓ | | | `b'3.14'` |
| 24 | *(const)* | — | — | — | — | — | `b"true\n"` → `CHECKSUM_TRUE`, pre-tabulated (§4) |
| 25 | *(const)* | — | — | — | — | — | `b"false\n"` → `CHECKSUM_FALSE` |
| 26 | *(const)* | — | — | — | — | — | `b"null\n"` → `CHECKSUM_NULL` |

Rows that the invariants make **unreachable** (worth stating, since they are the
common bugs): `DType`/`Rank ≠ NA`/`SCALAR` with a non-`NUMPY` `Kind`; `NUMPY_BYTES`
with anything but a scalar `S` array (rows 5–7, 9–16 all have `NB=False`);
`NUMERIC_SCALAR` on `JSON_OBJECT`/`ARRAY`/`RAW`/`NUMPY`; `SEMANTIC` off `RAW_TEXT`.

### 12.2 The `Length` axis

Each structural class is multiplied by the `Length` buckets its buffer can take:

| Kind | reachable `Length` | why |
|---|---|---|
| `RAW_BYTES`, `RAW_TEXT` | `SHORT` / `EQ64` / `MEDIUM` / `LONG` | any size |
| `NUMPY` | `MEDIUM` / `LONG` | the `.npy` header alone pads past 64 bytes, so never `SHORT`/`EQ64` |
| `MIXED_*` | `SHORT` (tiny) / `MEDIUM` / `LONG` | magic + skeleton; `EQ64` only by coincidence |
| `JSON_OBJECT`/`ARRAY`/`STRING` | `SHORT` … `LONG` | any size ≥ 2 |
| `JSON_NUMBER` | `SHORT` … `LONG` | `LONG` (>1000 digits) is valid but **forbids** numeric conversion (§6) |
| `bool`/`null` consts | fixed (`SHORT`) | the three buffers are 5/6/5 bytes |

Two `Length` witnesses the design leans on:

- **`RAW_TEXT ∧ EQ64`** — a 64-byte UTF-8 non-JSON string: precisely a `checksum`
  digest (`Checksum.hex()`), the §9 screen.
- **`JSON_NUMBER ∧ LONG`** — e.g. `b"1" + b"0"*2000`: a valid `JSON_NUMBER` whose
  `Length == LONG` flips every `→int/float` conversion to `✗` (§6.1).

Summing the structural classes over their reachable `Length` buckets gives on the
order of **70** distinct valid `TypeBits` words — out of the 4608 the raw bit
ranges could spell, the rest excluded by §12's invariants.

## 13. Deferred / open items

- **Producer of `SEMANTIC`** — out of scope (§4.2); it is provenance, set by the
  AST-canonicalizer, not derived from bytes (§8.5).
- **Per-Kind fill** is fully specified (§8.3 peek, §8.4 materialization); the
  `MIXED_OBJECT`/`MIXED_ARRAY` split and the numpy `DType`/`Rank`/`NUMPY_BYTES`
  are all peeked from the magic/header — no deferred producer.
- **Exact `yaml`/`ipython`/`python` validity** stays in
  `text_validation_celltype_cache` (Region 4 dropped); `deserializable_as`
  reports only the necessary `UTF8` "may-be" condition.
- **The merged-`DType` blind spot** — `binary→plain` for a non-scalar
  `NONNUMERIC` array is `?` (object would be `✗`, bytes-`S`/`U` `≠`). **Resolved:
  accepted**, as `binary→plain` is rare. The 3 spare bits of the word leave room
  for a fourth `DType` value (or a second refinement flag alongside `NUMPY_BYTES`)
  should that ever stop being true — no escape hatch is needed now.
