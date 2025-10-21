import pathway as pw

class InputSchema(pw.Schema):
    name: str

# Use tuples instead of dicts
table = pw.debug.table_from_rows(
    schema=InputSchema,
    rows=[
        ("Alice",),
        ("Bob",),
        ("Charlie",),
    ]
)

pw.debug.compute_and_print(table)