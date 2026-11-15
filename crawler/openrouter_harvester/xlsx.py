from __future__ import annotations

import math
import re
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


INVALID_XML = re.compile(
    "[\x00-\x08\x0B\x0C\x0E-\x1F\uD800-\uDFFF\uFFFE\uFFFF]"
)


def _xml_text(value: Any) -> str:
    return escape(INVALID_XML.sub("", str(value)))


def _col_name(index: int) -> str:
    result = ""
    value = index
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _cell_ref(row: int, col: int) -> str:
    return f"{_col_name(col)}{row}"


def _is_date_text(value: Any) -> bool:
    return isinstance(value, str) and bool(
        re.match(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?", value)
    )


def _style_for(header: str, value: Any) -> int:
    lowered = header.lower()
    if value is None:
        return 0
    if _is_date_text(value) or lowered.endswith("_at") or lowered in {
        "date",
        "snapshot_date",
    }:
        return 7
    if any(token in lowered for token in ("share", "change", "growth_percent")):
        return 6
    if isinstance(value, int) and not isinstance(value, bool):
        return 4
    if isinstance(value, float):
        return 5
    return 0


def _excel_serial(value: str) -> float | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    epoch = datetime(1899, 12, 30, tzinfo=UTC)
    return (parsed - epoch).total_seconds() / 86400


def _cell_xml(row: int, col: int, value: Any, style: int = 0) -> str:
    ref = _cell_ref(row, col)
    style_attr = f' s="{style}"' if style else ""
    if value is None:
        return f'<c r="{ref}"{style_attr}/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"{style_attr}><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            value = None
        else:
            return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    if style == 7 and isinstance(value, str):
        serial = _excel_serial(value)
        if serial is not None:
            return f'<c r="{ref}"{style_attr}><v>{serial}</v></c>'
    text = _xml_text("" if value is None else value)
    preserve = ' xml:space="preserve"' if text != text.strip() else ""
    return (
        f'<c r="{ref}" t="inlineStr"{style_attr}>'
        f"<is><t{preserve}>{text}</t></is></c>"
    )


@dataclass(frozen=True)
class SheetSpec:
    name: str
    title: str
    note: str
    rows: list[dict[str, Any]]
    columns: list[str] | None = None


class XlsxWriter:
    def __init__(self, sheets: list[SheetSpec], *, created_at: str) -> None:
        self.sheets = sheets
        self.created_at = created_at

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", self._content_types())
            archive.writestr("_rels/.rels", self._root_relationships())
            archive.writestr("docProps/core.xml", self._core_properties())
            archive.writestr("docProps/app.xml", self._app_properties())
            archive.writestr("xl/workbook.xml", self._workbook())
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                self._workbook_relationships(),
            )
            archive.writestr("xl/styles.xml", self._styles())
            for index, sheet in enumerate(self.sheets, 1):
                archive.writestr(
                    f"xl/worksheets/sheet{index}.xml",
                    self._worksheet(sheet),
                )
        temp.replace(path)

    def _columns(self, sheet: SheetSpec) -> list[str]:
        if sheet.columns:
            return sheet.columns
        return sorted({key for row in sheet.rows for key in row})

    def _worksheet(self, sheet: SheetSpec) -> str:
        columns = self._columns(sheet)
        last_col = _col_name(max(len(columns), 1))
        last_row = max(len(sheet.rows) + 4, 4)
        widths: list[float] = []
        for column in columns:
            samples = [str(row.get(column) or "") for row in sheet.rows[:200]]
            longest = max([len(column), *[min(len(value), 80) for value in samples]])
            widths.append(min(max(longest + 2, 10), 45))
        cols_xml = "".join(
            f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
            for index, width in enumerate(widths, 1)
        )
        rows_xml = [
            '<row r="1" ht="26" customHeight="1">'
            + _cell_xml(1, 1, sheet.title, 1)
            + "</row>",
            '<row r="2" ht="34" customHeight="1">'
            + _cell_xml(2, 1, sheet.note, 2)
            + "</row>",
            '<row r="3" ht="8" customHeight="1"></row>',
        ]
        header_cells = "".join(
            _cell_xml(4, index, column, 3)
            for index, column in enumerate(columns, 1)
        )
        rows_xml.append(
            f'<row r="4" ht="22" customHeight="1">{header_cells}</row>'
        )
        for row_index, row in enumerate(sheet.rows, 5):
            cells = "".join(
                _cell_xml(
                    row_index,
                    col_index,
                    row.get(column),
                    _style_for(column, row.get(column)),
                )
                for col_index, column in enumerate(columns, 1)
            )
            rows_xml.append(f'<row r="{row_index}">{cells}</row>')
        merge_xml = ""
        if len(columns) > 1:
            merge_xml = (
                '<mergeCells count="2">'
                f'<mergeCell ref="A1:{last_col}1"/>'
                f'<mergeCell ref="A2:{last_col}2"/>'
                "</mergeCells>"
            )
        auto_filter = (
            f'<autoFilter ref="A4:{last_col}{last_row}"/>' if columns else ""
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="A1:{last_col}{last_row}"/>'
            '<sheetViews><sheetView showGridLines="0" workbookViewId="0">'
            '<pane ySplit="4" topLeftCell="A5" activePane="bottomLeft" state="frozen"/>'
            '<selection pane="bottomLeft" activeCell="A5" sqref="A5"/>'
            "</sheetView></sheetViews>"
            '<sheetFormatPr defaultRowHeight="15"/>'
            f"<cols>{cols_xml}</cols>"
            f"<sheetData>{''.join(rows_xml)}</sheetData>"
            f"{merge_xml}{auto_filter}"
            "</worksheet>"
        )

    def _content_types(self) -> str:
        sheets = "".join(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for index in range(1, len(self.sheets) + 1)
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            f"{sheets}</Types>"
        )

    def _root_relationships(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            "</Relationships>"
        )

    def _workbook(self) -> str:
        sheets = "".join(
            f'<sheet name="{_xml_text(sheet.name[:31])}" sheetId="{index}" r:id="rId{index}"/>'
            for index, sheet in enumerate(self.sheets, 1)
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{sheets}</sheets>"
            '<calcPr calcId="191029" fullCalcOnLoad="1"/>'
            "</workbook>"
        )

    def _workbook_relationships(self) -> str:
        relationships = "".join(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
            for index in range(1, len(self.sheets) + 1)
        )
        styles_id = len(self.sheets) + 1
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{relationships}"
            f'<Relationship Id="rId{styles_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            "</Relationships>"
        )

    def _styles(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="4">
    <numFmt numFmtId="164" formatCode="#,##0"/>
    <numFmt numFmtId="165" formatCode="#,##0.000"/>
    <numFmt numFmtId="166" formatCode="0.0%"/>
    <numFmt numFmtId="167" formatCode="yyyy-mm-dd"/>
  </numFmts>
  <fonts count="4">
    <font><sz val="10"/><name val="Aptos"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="16"/><name val="Aptos Display"/></font>
    <font><i/><color rgb="FF475569"/><sz val="10"/><name val="Aptos"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Aptos"/></font>
  </fonts>
  <fills count="4">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF0F172A"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF0F766E"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left/><right/><top/><bottom style="thin"><color rgb="FFD1D5DB"/></bottom><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="8">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFill="1" applyFont="1"><alignment vertical="center"/></xf>
    <xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyFont="1"><alignment wrapText="1" vertical="center"/></xf>
    <xf numFmtId="0" fontId="3" fillId="3" borderId="1" xfId="0" applyFill="1" applyFont="1" applyBorder="1"><alignment vertical="center"/></xf>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
    <xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
    <xf numFmtId="166" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
    <xf numFmtId="167" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""

    def _core_properties(self) -> str:
        created = self.created_at.replace(" ", "T")
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            "<dc:creator>OpenRouter Harvester</dc:creator>"
            "<cp:lastModifiedBy>OpenRouter Harvester</cp:lastModifiedBy>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{_xml_text(created)}</dcterms:created>'
            "</cp:coreProperties>"
        )

    def _app_properties(self) -> str:
        titles = "".join(
            f"<vt:lpstr>{_xml_text(sheet.name[:31])}</vt:lpstr>"
            for sheet in self.sheets
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            "<Application>OpenRouter Harvester</Application>"
            f'<TitlesOfParts><vt:vector size="{len(self.sheets)}" baseType="lpstr">{titles}</vt:vector></TitlesOfParts>'
            "</Properties>"
        )


def build_workbook(
    path: Path,
    *,
    captured_at: str,
    status: str,
    sources: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    tables: dict[str, list[dict[str, Any]]],
) -> None:
    notes = {
        "model_rankings": "Daily view, sorted by token usage. Frontend endpoint; not a guaranteed public API.",
        "model_history": "Weekly model usage points exposed by the current rankings chart.",
        "task_model_share": "Current 30-day task shares. Repeated daily snapshots form the longitudinal record; this is not the removed legacy category cube.",
        "apps_root": "Marketplace popular, trending and category preview collections.",
        "app_global_rankings": "Top 40 global Apps for day, week and month.",
        "app_category_rankings": "Current category rankings for four groups and day/week periods.",
        "app_category_history": "Weekly category chart history. The current endpoint ignores day/week query differences.",
        "apps": "Deduplicated App entities selected from global top 40 and category top 10 unions.",
        "app_details": "App metadata and current summary parsed from React Flight payloads.",
        "app_model_rankings": "Top 20 models used by each selected App in the current monthly view.",
        "app_model_history": "Daily App-model usage history, top 20 models per App and date.",
        "models": "Model entities from the current catalog plus observed fallback identities.",
    }
    summary_rows = [
        {"key": "captured_at", "value": captured_at},
        {"key": "status", "value": status},
        *[
            {"key": f"rows:{name}", "value": len(rows)}
            for name, rows in sorted(tables.items())
        ],
    ]
    readme_rows = [
        {
            "topic": "Architecture",
            "detail": "Raw snapshots -> contract validation -> normalized JSON/SQLite -> Excel.",
        },
        {
            "topic": "Routine cost",
            "detail": "The normal cron path is deterministic and consumes no LLM tokens.",
        },
        {
            "topic": "Drift handling",
            "detail": "A new failure signature may invoke Codex once; repeated signatures are cooled down.",
        },
        {
            "topic": "Semantic safety",
            "detail": "Removed concepts are not silently remapped. They are classified for human review.",
        },
        {
            "topic": "Legacy categories",
            "detail": "The former roleplay/marketing/etc. category cube is no longer present in the current site shape. Task shares are stored separately.",
        },
    ]
    sheet_specs = [
        SheetSpec(
            "Read Me",
            "OpenRouter Public Data",
            "Methodology and caveats.",
            readme_rows,
            ["topic", "detail"],
        ),
        SheetSpec(
            "Run Summary",
            "Run Summary",
            "Counts and contract status.",
            summary_rows,
            ["key", "value"],
        ),
        SheetSpec(
            "Sources",
            "Source Provenance",
            "URLs, hashes and raw snapshot paths.",
            sources,
            ["name", "url", "sha256", "bytes", "content_type", "raw_path"],
        ),
        SheetSpec(
            "Contract Issues",
            "Contract Issues",
            "Errors block publication; warnings publish as degraded.",
            issues,
            ["severity", "source", "code", "message", "observed", "expected"],
        ),
    ]
    column_orders = {
        "model_rankings": [
            "snapshot_date", "view", "rank", "model_id", "model_name", "author",
            "token_usage", "share", "change", "request_count", "variant", "model_url",
        ],
        "model_history": [
            "date", "grain", "rank", "model_id", "model_name", "author",
            "token_usage", "total_usage", "share", "model_url",
        ],
        "task_model_share": [
            "snapshot_date", "metric", "window_days", "macro_category", "task", "rank",
            "model_id", "model_name", "author", "share_within_task", "delta_pp",
            "task_share_total", "model_url",
        ],
        "apps_root": [
            "section", "rank", "app_id", "app_name", "app_slug", "categories",
            "token_usage", "request_count", "growth_percent", "description",
            "origin_url", "main_url", "app_url", "created_at",
        ],
        "app_global_rankings": [
            "snapshot_date", "period", "rank", "app_id", "app_name", "app_slug",
            "categories", "token_usage", "request_count", "description",
            "origin_url", "main_url", "app_url", "created_at",
        ],
        "app_category_rankings": [
            "snapshot_date", "group", "period", "rank", "app_id", "app_name",
            "app_slug", "categories", "token_usage", "request_count", "description",
            "origin_url", "main_url", "app_url", "created_at",
        ],
        "app_category_history": [
            "date", "grain", "group", "rank", "app_id", "app_name",
            "token_usage", "total_usage", "share",
        ],
        "apps": [
            "app_id", "app_name", "app_slug", "categories", "description",
            "origin_url", "main_url", "app_url", "created_at", "selection_reasons",
        ],
        "app_details": [
            "app_id", "app_name", "app_slug", "categories", "total_tokens",
            "global_rank", "models_used", "category_rankings", "selection_reasons",
            "description", "origin_url", "main_url", "app_url", "page_url",
            "fetch_status", "fetch_error", "created_at",
        ],
        "app_model_rankings": [
            "snapshot_date", "app_id", "app_name", "rank", "model_id", "model_name",
            "author", "token_usage", "model_url",
        ],
        "app_model_history": [
            "date", "app_id", "app_name", "rank", "model_id", "model_name", "author",
            "token_usage", "total_usage", "share", "model_url",
        ],
        "models": [
            "model_id", "slug", "model_name", "author", "context_length",
            "input_modalities", "output_modalities", "supports_reasoning",
            "created_at", "updated_at", "model_url",
        ],
    }
    mapping = [
        ("Model Rankings", "model_rankings"),
        ("Model History", "model_history"),
        ("Task Model Share", "task_model_share"),
        ("Apps Root", "apps_root"),
        ("Apps Global", "app_global_rankings"),
        ("Apps Category", "app_category_rankings"),
        ("App Cat History", "app_category_history"),
        ("Apps", "apps"),
        ("App Details", "app_details"),
        ("App Model Rank", "app_model_rankings"),
        ("App Model History", "app_model_history"),
        ("Models", "models"),
    ]
    for sheet_name, table_name in mapping:
        sheet_specs.append(
            SheetSpec(
                sheet_name,
                sheet_name,
                notes[table_name],
                tables.get(table_name, []),
                column_orders[table_name],
            )
        )
    XlsxWriter(sheet_specs, created_at=captured_at).write(path)
