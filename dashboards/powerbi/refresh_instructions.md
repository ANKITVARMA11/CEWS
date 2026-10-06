# Loading and refreshing in Power BI Desktop

## 1. Export

```
cews export-powerbi
```

Writes to `data\exports\powerbi\` (or use `--out FOLDER`). Run it after `cews insights` so the
tables reflect the latest scores. The export only reads the database and never changes it.

**`refresh_metadata.json` is written last.** If it is missing, an export was interrupted and the
folder should not be trusted: run the export again.

## 2. Load (once)

1. Power BI Desktop > **Home > Transform data > Manage Parameters > New Parameter**.
   Name `ExportFolder`, type Text, value `C:\projects\CEWS\data\exports\powerbi` (your path,
   no trailing backslash).
2. **Home > New Source > Blank Query**, open **Advanced Editor**, paste the following, and name
   the query `fn_LoadTable`. It reads each table's column types from `refresh_metadata.json`, so
   types stay correct if columns are ever added:

```m
(tableName as text) as table =>
let
    Meta = Json.Document(File.Contents(ExportFolder & "\refresh_metadata.json")),
    Columns = Record.Field(Meta[tables], tableName)[columns],
    Csv = Csv.Document(
        File.Contents(ExportFolder & "\" & tableName & ".csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
    ),
    Promoted = Table.PromoteHeaders(Csv, [PromoteAllScalars = true]),
    TypeFor = (kind as text) as type =>
        if kind = "int" or kind = "flag" then Int64.Type
        else if kind = "float" then type number
        else if kind = "date" then type date
        else if kind = "datetime" then type datetimezone
        else type text,
    Typed = Table.TransformColumnTypes(
        Promoted,
        List.Transform(Columns, each {[name], TypeFor([type])}),
        "en-US"
    )
in
    Typed
```

3. For each of the 13 tables, **New Source > Blank Query**, Advanced Editor, and enter
   `= fn_LoadTable("fact_scores")` (with that table's name). Name each query after its table.
4. **Close & Apply**, then follow `data_model.md` to mark `dim_date` as the date table and check
   the relationships.

`"en-US"` matters: the files always use a full stop as the decimal separator, whatever your
computer's regional settings.

## 3. Refresh

Run `cews export-powerbi`, then **Home > Refresh** in Power BI Desktop.

Refreshing a published report automatically from the Power BI service would need a data gateway
to reach the folder; that is not set up here.

## 4. Theme

**View > Themes > Browse for themes** and choose `powerbi_theme.json`. It is colour-blind safe;
keep red for things that need attention.

## Troubleshooting

- *"File not found"*: check the `ExportFolder` parameter; run `cews export-powerbi` first.
- *Numbers look wrong or as text*: the culture argument in `fn_LoadTable` must stay `"en-US"`.
- *Accented characters garbled*: keep `Encoding = 65001` (UTF-8).
- *A table is empty*: that is normal before its pipeline step has run (headers are always present).
  For example `fact_forecasts` is empty until `cews forecast`.
