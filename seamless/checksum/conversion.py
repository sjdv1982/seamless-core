"""Checksum-level celltype conversion rules.

The table below is executed by :mod:`seamless.checksum.convert` for an empty
Expression path. A rule either retains the checksum, reinterprets the same
buffer, creates a replacement buffer, or requires an ordinary value
conversion. Paths and deep hash patterns remain Expression-level work.

The relevant storage hierarchy is ``bytes`` / ``mixed`` / ``binary`` and
``plain``, plus UTF-8 ``text`` (and its code/YAML subtypes). Scalar celltypes
are interpretations of JSON-compatible buffers. In particular, ``str``
buffers are not restricted to quoted JSON strings: the reference parser gives
the scalar interpretation for the requested celltype.

Subclass-to-superclass conversions are normally trivial; superclass-to-
subclass conversions are reinterpretations and therefore validate the target
interpretation. The binary/bytes boundary and text/JSON boundary have the
explicit reformat rules in ``conversion_reformat``.

``text -> plain`` first tries :mod:`orjson` and otherwise stores the text as a
plain string. Integer and float parsing has a 1000-byte input limit. Null and
boolean canonical buffers are checksum-based virtual values, so relevant rules
can succeed or fail without fetching their buffer.
"""

from .celltypes import celltypes


class SeamlessConversionError(ValueError):
    """Seamless celltype-to-celltype conversion error"""

    def __str__(self):
        args = [str(arg) for arg in self.args]
        return "\n".join(args)

    def __repr__(self):
        args = [str(arg) for arg in self.args]
        return "\n".join(args)


conversion_trivial = set(
    [
        # conversions that do not change checksum and are guaranteed to work (if the input is valid)
        ("text", "bytes"),
        ("ipython", "text"),
        ("python", "text"),
        ("python", "ipython"),
        ("yaml", "text"),
        ("plain", "yaml"),
        ("plain", "bytes"),
        ("binary", "mixed"),
        ("plain", "mixed"),
        ("str", "plain"),
        ("int", "plain"),
        ("float", "plain"),
        ("bool", "plain"),
        # Scalar widening / stringification changes interpretation, not bytes.
        ("int", "float"),
        ("float", "int"),
        ("int", "str"),
        ("float", "str"),
        ("bool", "str"),
    ]
)

conversion_reinterpret = set()
# conversions that do not change checksum, but are not guaranteed to work (raise exception).
for source, target in conversion_trivial:
    reverse = (target, source)
    if reverse not in conversion_trivial:
        conversion_reinterpret.add(reverse)
conversion_reinterpret.difference_update(
    set(
        [
            ("ipython", "python"),
            ("yaml", "plain"),
        ]
    )
)

conversion_reformat = set(
    [
        # conversions that are guaranteed to work (if the input is valid), but may change checksum
        # special cases:
        (
            "bytes",
            "binary",
        ),  # for numpy buffer format (magic numpy string), trivial.
        # Else, create np.dtype(S) array from bytes buffer.
        #
        ("bytes", "mixed"),  # as above, but:
        #  - Seamless-mixed buffer format (MAGIC_SEAMLESS_MIXED string) is also trivial.
        #  - If buffer is text, value stays the same (text-to-str)
        #
        ("binary", "bytes"),  # for np.dtype(S..), get value.tobytes(); else trivial
        #
        (
            "mixed",
            "bytes",
        ),  # ("binary", "bytes") if binary (magic numpy string); else trivial
        #
        (
            "plain",
            "text",
        ),  # if value is a string, value stays the same; else checksum stays the same
        #
        (
            "text",
            "plain",
        ),  # if orjson accepts it, keep; else serialize the text as a plain str.
        #
        ("text", "str"),  # value stays the same
        ("str", "text"),  # value stays the same
        ("yaml", "plain"),  # run YAML parser
        ("ipython", "python"),  # convert to Python code
    ]
)

conversion_possible = set(
    [  # conversions that (may) change checksum and are not guaranteed to work (raise exception)
        # simple value conversions:
        ("binary", "int"),
        ("binary", "float"),
        ("binary", "bool"),
        ("mixed", "str"),
        ("mixed", "int"),
        ("mixed", "float"),
        ("mixed", "bool"),
    ]
)

###

conversion_equivalent = {  # equivalent conversions
    # special cases
    #
    # 1. text_subtype-to-str. Do not promote to text-to-plain!
    ("yaml", "str"): ("text", "str"),
    ("text", "mixed"): ("text", "str"),  # NOT text-to-plain.
    # But mixed => text does go mixed => plain => text
    #
    ("yaml", "mixed"): ("text", "str"),  # NOT yaml-to-plain
    ("python", "str"): ("text", "str"),
    ("ipython", "str"): ("text", "str"),
    ("python", "mixed"): ("text", "str"),
    ("ipython", "mixed"): ("text", "str"),
    ("python", "plain"): ("text", "str"),
    ("ipython", "plain"): ("text", "str"),
    #
    # 2. str-to-text_subtype. Plain-to-text or str-to-text are the same here.
    ("str", "yaml"): ("str", "text"),
    ("str", "python"): ("str", "text"),
    ("str", "ipython"): ("str", "text"),
    # apply specific converter to generalized outputs
    ("python", "bytes"): ("python", "text"),
    ("ipython", "bytes"): ("ipython", "text"),
    ("str", "mixed"): ("str", "plain"),
    ("int", "mixed"): ("int", "plain"),
    ("float", "mixed"): ("float", "plain"),
    ("bool", "mixed"): ("bool", "plain"),
    # apply generic converter to specified inputs
    ("str", "binary"): ("plain", "binary"),
    ("float", "binary"): ("plain", "binary"),
    ("int", "binary"): ("plain", "binary"),
    ("bool", "binary"): ("plain", "binary"),
    ("python", "binary"): ("text", "binary"),
    ("ipython", "binary"): ("text", "binary"),
    ("yaml", "bytes"): ("text", "bytes"),
    ("str", "bytes"): ("plain", "bytes"),
    ("int", "bytes"): ("plain", "bytes"),
    ("float", "bytes"): ("plain", "bytes"),
    ("bool", "bytes"): ("plain", "bytes"),
    ("int", "text"): ("plain", "text"),
    ("float", "text"): ("plain", "text"),
    ("bool", "text"): ("plain", "text"),
}

conversion_chain = {  # (A,C): B means convert A => B => C
    ("mixed", "text"): "plain",  # special case
    ("mixed", "yaml"): "text",
    ("mixed", "ipython"): "text",
    ("mixed", "python"): "text",
    ("binary", "text"): "plain",  # special case
    ("binary", "yaml"): "text",
    ("binary", "ipython"): "text",
    ("binary", "python"): "text",
    ("bytes", "str"): "plain",
    ("bytes", "float"): "plain",
    ("bytes", "int"): "plain",
    ("bytes", "bool"): "plain",
    ("bytes", "yaml"): "text",
    ("bytes", "ipython"): "text",
    ("bytes", "python"): "text",
    ("binary", "str"): "bytes",  # binary => bytes (special case) => plain => str
    ("plain", "python"): "text",
    ("plain", "ipython"): "text",
    ("text", "binary"): "mixed",
    ("text", "float"): "plain",
    ("text", "int"): "plain",
    ("text", "bool"): "plain",
    ("yaml", "binary"): "plain",
    ("yaml", "int"): "plain",
    ("yaml", "float"): "plain",
    ("yaml", "bool"): "plain",
    ("int", "yaml"): "plain",
    ("float", "yaml"): "plain",
    ("bool", "yaml"): "plain",
}

conversion_values = set(
    [
        # These conversions must be handled elsewhere (with caching and/or values)
        # value conversions. Invalid for many values.
        ("binary", "plain"),  # use json_encode
        (
            "plain",
            "binary",
        ),  # value must be a list; parse with numpy, and dtype must not be "object"
        # or: value must be a scalar
        # Scalar conversions that must change their serialized value.
        ("bool", "int"),
        ("int", "bool"),
        ("float", "bool"),
        ("bool", "float"),
        #
        # conversions from/to checksum.
        ("checksum", "bytes"),
        ("checksum", "mixed"),
        ("checksum", "binary"),
        ("checksum", "plain"),
        ("checksum", "str"),
        ("checksum", "int"),
        ("checksum", "float"),
        ("checksum", "bool"),
        ("checksum", "text"),
        ("checksum", "yaml"),
        ("checksum", "python"),
        ("checksum", "ipython"),
        ("bytes", "checksum"),
        ("mixed", "checksum"),
        ("binary", "checksum"),
        ("plain", "checksum"),
        ("str", "checksum"),
        ("int", "checksum"),
        ("float", "checksum"),
        ("bool", "checksum"),
        ("text", "checksum"),
        ("yaml", "checksum"),
        ("python", "checksum"),
        ("ipython", "checksum"),
    ]
)

conversion_forbidden = set(
    [
        # completely forbidden conversions.
        ("python", "yaml"),
        ("python", "int"),
        ("python", "float"),
        ("python", "bool"),
        ("ipython", "yaml"),
        ("ipython", "int"),
        ("ipython", "float"),
        ("ipython", "bool"),
        ("yaml", "python"),
        ("yaml", "ipython"),
        ("int", "python"),
        ("int", "ipython"),
        ("float", "python"),
        ("float", "ipython"),
        ("bool", "python"),
        ("bool", "ipython"),
    ]
)


def check_conversions():
    """Check if all possible conversions are classified into categories"""
    categories = (
        conversion_trivial,
        conversion_reformat,
        conversion_reinterpret,
        conversion_possible,
        conversion_equivalent,
        conversion_chain,
        conversion_values,
        conversion_forbidden,
    )
    for celltype1 in celltypes:
        for celltype2 in celltypes:
            if celltype1 == celltype2:
                continue
            conv = (celltype1, celltype2)
            done = [conv]
            while 1:
                covered = 0
                for category in categories:
                    if conv in category:
                        covered += 1
                if covered == 0:
                    raise SeamlessConversionError("Missing conversion: %s" % str(conv))
                elif covered > 1:
                    raise SeamlessConversionError(
                        "Duplicate conversion: %s" % str(conv)
                    )
                if conv not in conversion_equivalent and conv not in conversion_chain:
                    break
                if conv in conversion_equivalent and conv in conversion_chain:
                    raise SeamlessConversionError(
                        "Duplicate conversion mapping: %s" % str(conv)
                    )
                if conv in conversion_equivalent:
                    conv = conversion_equivalent[conv]
                elif conv in conversion_chain:
                    conv = (conversion_chain[conv], conv[1])
                if conv in done:
                    raise SeamlessConversionError(
                        "Circular equivalence: %s" % str(conv)
                    )
                done.append(conv)


check_conversions()
