from prettytable import PrettyTable

def generate_table(fields_names: list, rows: zip) -> PrettyTable:
    table = PrettyTable()
    table.field_names = fields_names
    table.float_format = ".2"

    for sample, row in enumerate(rows, start=1):
        if len(fields_names) != len(row):
            row = (sample,) + row
        table.add_row(row)

    return table