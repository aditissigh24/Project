"""
CLI entrypoint for the Multi-Agent Long Document Intelligence system.

Commands:
  python -m main ingest <pdf_path>               Run full ingestion pipeline
  python -m main query  <doc_id> "<question>"    Run query pipeline against processed doc
  python -m main status <doc_id>                 Show document processing status
  python -m main reset  <doc_id>                 Delete all Mongo data for a document (dev use)
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from db.mongo_client import ensure_indexes
import db.repositories as repo

app = typer.Typer(help="Multi-Agent Long Document Intelligence CLI", no_args_is_help=True)
console = Console()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Only show our loggers at INFO
for mod in ("agent", "db"):
    logging.getLogger(mod).setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    pdf_path: str = typer.Argument(..., help="Path to the scanned PDF file"),
    doc_id: str = typer.Option(None, "--doc-id", help="Explicit document ID (auto-generated if omitted)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed logs"),
):
    """
    Run the full ingestion pipeline on a PDF:
    OCR → Cleaning → Segmentation → Local Analysis → Aggregation → Consistency Check
    """
    if verbose:
        logging.getLogger("agent").setLevel(logging.DEBUG)

    pdf = Path(pdf_path)
    if not pdf.exists():
        console.print(f"[red]Error: File not found: {pdf_path}[/red]")
        raise typer.Exit(1)

    if doc_id is None:
        doc_id = str(uuid.uuid4())

    run_id = str(uuid.uuid4())

    console.print(Panel.fit(
        f"[bold cyan]Document ID:[/bold cyan] {doc_id}\n"
        f"[bold cyan]PDF:[/bold cyan] {pdf_path}\n"
        f"[bold cyan]Run ID:[/bold cyan] {run_id}",
        title="[bold]Starting Ingestion Pipeline[/bold]",
    ))

    ensure_indexes()
    repo.upsert_run(doc_id, run_id, {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
    })

    initial_state = {
        "doc_id": doc_id,
        "pdf_path": str(pdf.resolve()),
        "run_id": run_id,
        "page_count": 0,
        "pages_ocr_used": 0,
        "cleaned": False,
        "segment_ids": [],
        "analyzed_segment_ids": [],
        "section_ids": [],
        "chapter_ids": [],
        "has_master_summary": False,
        "contradiction_count": 0,
        "errors": [],
    }

    async def _run():
        from agent.graph import get_ingestion_graph
        graph = get_ingestion_graph()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Running ingestion pipeline…", total=None)

            final_state = await graph.ainvoke(initial_state)

            progress.update(task, description="[green]Pipeline complete!")

        return final_state

    final_state = asyncio.run(_run())

    # Summary table
    table = Table(title="Ingestion Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Document ID", doc_id)
    table.add_row("Pages processed", str(final_state.get("page_count", 0)))
    table.add_row("Pages via OCR", str(final_state.get("pages_ocr_used", 0)))
    table.add_row("Segments", str(len(final_state.get("segment_ids", []))))
    table.add_row("Sections", str(len(final_state.get("section_ids", []))))
    table.add_row("Chapters", str(len(final_state.get("chapter_ids", []))))
    table.add_row("Contradictions found", str(final_state.get("contradiction_count", 0)))
    errors = final_state.get("errors", [])
    table.add_row("Errors", str(len(errors)) + (" ⚠" if errors else ""))
    console.print(table)

    if errors and verbose:
        console.print("\n[yellow]Errors:[/yellow]")
        for err in errors[:10]:
            console.print(f"  • {err}")

    console.print(f"\n[green]✓ Document ready. Use:[/green]")
    console.print(f'  python -m main query {doc_id} "Your question here"')


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

@app.command()
def query(
    doc_id: str = typer.Argument(..., help="Document ID returned by ingest"),
    question: str = typer.Argument(..., help="Natural language query"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Query a processed document using the multi-agent reasoning pipeline.
    """
    if verbose:
        logging.getLogger("agent").setLevel(logging.DEBUG)

    doc = repo.get_document(doc_id)
    if doc is None:
        console.print(f"[red]Document '{doc_id}' not found. Run ingest first.[/red]")
        raise typer.Exit(1)

    if doc.get("status") != "READY":
        console.print(f"[yellow]Document status: {doc.get('status')}. Wait for ingestion to complete.[/yellow]")
        raise typer.Exit(1)

    run_id = str(uuid.uuid4())
    initial_state = {
        "doc_id": doc_id,
        "query": question,
        "run_id": run_id,
        "query_type": None,
        "target_section_ids": [],
        "messages": [],
        "answer": None,
    }

    console.print(Panel.fit(
        f"[bold cyan]Query:[/bold cyan] {question}",
        title="[bold]Processing Query[/bold]",
    ))

    async def _run():
        from agent.graph import get_query_graph
        graph = get_query_graph()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Running query pipeline…", total=None)
            final_state = await graph.ainvoke(initial_state)

        return final_state

    final_state = asyncio.run(_run())

    answer = final_state.get("answer")
    if answer is None:
        console.print("[red]No answer was produced.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[dim]Query type: {answer.query_type}  |  Confidence: {answer.confidence:.0%}[/dim]\n")
    console.print(Markdown(answer.answer))

    if answer.citations:
        console.print("\n[bold]Citations:[/bold]")
        for c in answer.citations:
            parts = []
            if c.section_heading:
                parts.append(c.section_heading)
            if c.page_range:
                parts.append(f"pp. {c.page_range[0]}-{c.page_range[1]}")
            if c.segment_id:
                parts.append(f"segment: {c.segment_id}")
            console.print(f"  • {' | '.join(parts)}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@app.command()
def status(
    doc_id: str = typer.Argument(..., help="Document ID"),
):
    """
    Show processing status and latest run metrics for a document.
    """
    doc = repo.get_document(doc_id)
    if doc is None:
        console.print(f"[red]Document '{doc_id}' not found.[/red]")
        raise typer.Exit(1)

    table = Table(title=f"Document Status: {doc_id}", show_header=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Status", doc.get("status", "unknown"))
    table.add_row("Source", doc.get("source_path", ""))
    table.add_row("Page count", str(doc.get("page_count", 0)))
    table.add_row("Created", str(doc.get("created_at", "")))
    table.add_row("Updated", str(doc.get("updated_at", "")))

    run = repo.get_latest_run(doc_id)
    if run:
        table.add_row("Last run ID", run.get("run_id", ""))
        table.add_row("Run started", str(run.get("started_at", "")))
        table.add_row("Run finished", str(run.get("finished_at", "")))
        table.add_row("Segments analyzed", str(run.get("analyzed_count", 0)))
        table.add_row("Contradictions", str(run.get("contradiction_count", 0)))
        errors = run.get("errors", [])
        table.add_row("Errors", str(len(errors)))

    console.print(table)


# ---------------------------------------------------------------------------
# reset (dev utility)
# ---------------------------------------------------------------------------

@app.command()
def reset(
    doc_id: str = typer.Argument(..., help="Document ID to delete"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """
    Delete all MongoDB data for a document. For development/testing use only.
    """
    if not confirm:
        typer.confirm(f"Delete all data for document '{doc_id}'?", abort=True)

    from db.mongo_client import get_db
    db = get_db()
    collections = ["documents", "pages", "segments", "segment_analyses",
                   "section_summaries", "chapter_summaries", "document_summary",
                   "contradictions", "runs"]
    for col_name in collections:
        result = db[col_name].delete_many({"doc_id": doc_id})
        console.print(f"  Deleted {result.deleted_count} records from '{col_name}'")

    console.print(f"\n[green]✓ Cleared all data for document {doc_id}[/green]")


if __name__ == "__main__":
    app()
