import pandas as pd


def print_overview(df):
    """Print the shape, plus each column's missing count, unique count, and an example value."""
    if len(df) > 0:
        example = df.iloc[0]
    else:
        example = None

    summary = pd.DataFrame({
        "missing": df.isna().sum(),
        # astype(str) so list columns (which can't be hashed) can still be counted
        "unique": df.astype(str).nunique(),
        "example": example,
    })
    _print_table(summary, f"{df.shape[0]} rows x {df.shape[1]} columns")


def print_columns(df, *columns, n=10):
    """Print the first n rows of the chosen columns.

    print_columns(clean, "age", "hemorrhage_type", "outcome", n=5)
    """
    _check_columns(df, columns)
    _print_table(df[list(columns)].head(n), f"First {min(n, len(df))} of {len(df)} rows")


def print_rows(df, start=0, stop=10, columns=None):
    """Print rows by position, optionally only some columns.

    print_rows(clean, 100, 110, columns=["patient_id", "age", "outcome"])
    """
    selected = df.iloc[start:stop]
    if columns is not None:
        _check_columns(df, columns)
        selected = selected[columns]
    _print_table(selected, f"Rows {start} to {start + len(selected) - 1} of {len(df)}")


def print_patient(df, patient_id, id_column="patient_id"):
    """Print every field for one patient, one field per line.

    print_patient(clean, "BH-10042")
    """
    matches = df[df[id_column] == patient_id]
    if matches.empty:
        print(f"No patient with {id_column} == {patient_id!r}\n")
        return
    print(f"== Patient {patient_id} ==")
    with _display_options():
        print(matches.iloc[0].to_string(), end="\n\n")


def print_where(df, query=None, columns=None, n=10, **filters):
    """Print rows matching a pandas query string and/or column=value filters.

    A filter value can be one value or a list of allowed values. For list
    columns (risk_factors, symptoms) it matches rows containing that item.

    print_where(clean, outcome="Fatal", gender="Female")
    print_where(clean, "age >= 80", symptoms="Seizures", columns=["age", "symptoms", "outcome"])
    print_where(clean, region=["Oceania", "Central Asia"], n=None)    # n=None prints every match
    """
    _check_columns(df, list(filters))
    if columns is not None:
        _check_columns(df, columns)

    # Start with every row (or only the rows matching the query),
    # then narrow it down one filter at a time.
    if query:
        selected = df.query(query)
    else:
        selected = df

    for column, value in filters.items():
        # Treat a single value as a list of one, so both cases work the same way.
        if isinstance(value, (list, tuple, set)):
            wanted = value
        else:
            wanted = [value]

        if _is_list_column(selected[column]):
            matches = selected[column].map(lambda items: _contains_any(items, wanted))
        else:
            matches = selected[column].isin(wanted)
        selected = selected[matches]

    if n is None:
        shown = selected
    else:
        shown = selected.head(n)
    if columns is not None:
        shown = shown[columns]

    # Describe the filters for the title, e.g. "age >= 80, outcome='Fatal'".
    conditions = []
    if query:
        conditions.append(query)
    for column, value in filters.items():
        conditions.append(f"{column}={value!r}")
    if conditions:
        description = ", ".join(conditions)
    else:
        description = "everything"

    title = f"{len(selected)} of {len(df)} rows match {description} (showing {len(shown)})"
    _print_table(shown, title)


def print_counts(df, column):
    """Print how often each value appears, with percentages.

    For list columns, each item is counted separately, so percentages are
    "% of patients with this item" and can add up to more than 100.
    """
    _check_columns(df, [column])
    values = df[column]
    if _is_list_column(values):
        values = values.explode()

    counts = values.value_counts(dropna=False)
    percent = (counts / len(df) * 100).round(1)
    table = pd.DataFrame({"count": counts, "percent": percent})
    _print_table(table, f"Counts for {column} ({len(df)} rows)")


def print_crosstab(df, row, column, percent=False):
    """Print a count table of one column against another.

    With percent=True each row adds up to 100, which is handy for
    "what share of each hemorrhage type was fatal?".

    print_crosstab(clean, "hemorrhage_type", "outcome", percent=True)
    print_crosstab(clean, "symptoms", "outcome")    # list columns work too
    """
    _check_columns(df, [row, column])
    pair = df[[row, column]]
    # Give each item in a list column its own row, so items are counted separately.
    for name in [row, column]:
        if _is_list_column(pair[name]):
            pair = pair.explode(name, ignore_index=True)

    if percent:
        shares = pd.crosstab(pair[row], pair[column], normalize="index")
        table = (shares * 100).round(1)
        title = f"{row} vs {column} (row %)"
    else:
        table = pd.crosstab(pair[row], pair[column], margins=True, margins_name="Total")
        title = f"{row} vs {column}"
    _print_table(table, title)


def _print_table(table, title):
    print(f"== {title} ==")
    if len(table) == 0:
        print("(no rows)\n")
        return
    with _display_options():
        print(table, end="\n\n")


def _display_options():
    """Show every column and full cell text instead of pandas' "..." truncation."""
    return pd.option_context(
        "display.max_columns", None,
        "display.max_colwidth", None,
        "display.width", 200,
    )


def _check_columns(df, columns):
    """Raise an error listing the real column names if any of `columns` don't exist."""
    missing = []
    for column in columns:
        if column not in df.columns:
            missing.append(column)
    if missing:
        raise KeyError(f"No column(s) {missing}. Available columns: {list(df.columns)}")


def _is_list_column(values):
    """True if the column holds lists, like the cleaned symptoms column."""
    for value in values:
        if isinstance(value, list):
            return True
    return False


def _contains_any(items, wanted):
    """True if the list `items` holds at least one of the `wanted` values."""
    for item in wanted:
        if item in items:
            return True
    return False
