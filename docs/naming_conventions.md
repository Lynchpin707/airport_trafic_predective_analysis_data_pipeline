# Naming Conventions

## General Guidelines

- **Language**: Use English for all identifiers.
- **Case Format**: Use `snake_case` — all lowercase with underscores (`_`) separating words.
- **Avoid Reserved Words**: Do **not** use SQL or python reserved keywords for naming tables, columns, or procedures.

## Functions naming convention
- **Format**: `<verb>_<object>_[<goal>]`
- **Rule**: a function name starts with a verb describing what the fonction does followed by the object on which the function acts with the possibility of adding more details about its goal.
- **Example**: `prepare_data_for_prophet()`, `make_forecast()`

## Error handeling
- **Format**: `Error processing {airport} : "The original error message"`
- **Example**: `Error processing casablanca: time data "2013_s1" doesn't match format "%Y_%m", at position 0.`
  
## Specific naming conventions
### Period naming convention
- **Format**: `<year>_<semester>`
- **Rule**: a period starts with the year followed by the semester (s1 or s2).
- **Example**: `2024_s2`