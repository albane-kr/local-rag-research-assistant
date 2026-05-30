import typer
import requests
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

app = typer.Typer(help="CLI for Local RAG Research Assistant")
console = Console()

API_BASE_URL = "http://localhost:8000"


@app.command()
def ingest(file_path: str = typer.Argument(..., help="Path to file to ingest")):
    """Ingest a document file."""
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    with console.status(f"[bold green]Ingesting {path.name}..."):
        with open(path, "rb") as f:
            files = {"file": (path.name, f, "application/octet-stream")}
            try:
                response = requests.post(f"{API_BASE_URL}/ingest", files=files)
                response.raise_for_status()
            except requests.exceptions.ConnectionError:
                console.print("[red]Error: Cannot connect to API. Is it running?[/red]")
                raise typer.Exit(1)

    data = response.json()
    console.print()
    console.print(Panel(
        f"[green]✓ Ingestion successful[/green]\n\n"
        f"Resource ID: [bold]{data['resource_id']}[/bold]\n"
        f"Version: {data['version']}\n"
        f"Status: {data['status']}\n"
        f"ETL Status: {data['etl_status']}\n"
        f"Chunks: {data.get('chunk_count', 0)}",
        title="Ingest Result"
    ))


@app.command()
def resources():
    """UI-01: List all resources with their latest versions."""
    try:
        response = requests.get(f"{API_BASE_URL}/resources")
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        console.print("[red]Error: Cannot connect to API. Is it running?[/red]")
        raise typer.Exit(1)

    data = response.json()
    if not data.get("resources"):
        console.print("[yellow]No resources found.[/yellow]")
        return

    table = Table(title="Resources")
    table.add_column("Resource ID", style="cyan")
    table.add_column("Current Version", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Chunks", style="blue")

    for res in data["resources"]:
        table.add_row(
            res["resource_id"],
            str(res["latest_version"]),
            res["status"],
            str(res.get("chunk_count", 0))
        )

    console.print(table)


@app.command()
def versions(resource_id: str = typer.Argument(..., help="Resource ID")):
    """List all versions of a resource."""
    try:
        response = requests.get(f"{API_BASE_URL}/resources/{resource_id}/versions")
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        console.print("[red]Error: Cannot connect to API.[/red]")
        raise typer.Exit(1)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            console.print(f"[red]Resource not found: {resource_id}[/red]")
        else:
            console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    data = response.json()
    table = Table(title=f"Versions of {resource_id}")
    table.add_column("Version", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Created", style="blue")
    table.add_column("Chunks", style="yellow")

    for v in data["versions"]:
        table.add_row(
            str(v["version"]),
            v["status"],
            v.get("created_at", "N/A"),
            str(v.get("chunk_count", 0))
        )

    console.print(table)


@app.command()
def chat(query: str = typer.Argument(..., help="Question to ask")):
    """UI-02: Chat with the RAG system and display provenance."""
    with console.status("[bold green]Querying..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={"query": query, "top_k": 5}
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            console.print("[red]Error: Cannot connect to API.[/red]")
            raise typer.Exit(1)

    data = response.json()
    console.print()

    # Main answer
    console.print(Panel(data["answer"], title="Answer", expand=False))

    # Provenance metadata
    console.print()
    console.print("[bold]Provenance:[/bold]")
    console.print(f"  Model: {data.get('model_used', 'N/A')}")
    console.print(f"  Query Type: {data.get('query_type', 'N/A')}")
    console.print(f"  Chunks Retrieved: {data.get('chunk_count', 0)}")
    console.print(f"  Resources: {', '.join(data.get('resource_ids', []))}")
    console.print(f"  Versions: {', '.join(str(v) for v in data.get('versions', []))}")

    # Source chunks
    if data.get("chunk_count", 0) > 0:
        console.print()
        console.print("[bold]Source Chunks:[/bold]")
        for i, chunk_id in enumerate(data.get("chunk_ids", [])[:3], 1):
            console.print(f"  [{i}] {chunk_id}")
        if len(data.get("chunk_ids", [])) > 3:
            console.print(f"  ... and {len(data['chunk_ids']) - 3} more")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
