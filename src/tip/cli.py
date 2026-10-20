from __future__ import annotations

import logging

import typer

app = typer.Typer(help="Test Intelligence Platform", no_args_is_help=True)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@app.command()
def ingest(days: int = typer.Option(90, help="Keep the last N days of builds locally")) -> None:
    """Fetch Zuul builds (incremental + backfill) into data/raw/builds."""
    from tip.ingest.zuul import crawl
    n = crawl(days)
    typer.echo(f"wrote {n} build rows")


@app.command()
def stats() -> None:
    """Print a summary of the local build store."""
    from tip.warehouse import connect
    con = connect()
    print(con.sql("""
        SELECT count(*) AS builds, count(DISTINCT project) AS projects,
               count(DISTINCT job_name) AS jobs, min(start_time) AS first, max(end_time) AS last
        FROM builds
    """))
    print(con.sql("SELECT result, count(*) n FROM final_builds GROUP BY 1 ORDER BY n DESC"))


@app.command()
def dataset() -> None:
    """Build the flaky-failure triage dataset (labels + as-of features)."""
    from tip.flaky.dataset import build_dataset
    df = build_dataset()
    typer.echo(f"{len(df)} labelled failures, flaky rate {df['flaky'].mean():.3f}")


@app.command()
def train() -> None:
    """Train and evaluate the triage model on a temporal split."""
    from tip.flaky.model import train_and_evaluate
    metrics = train_and_evaluate()
    for split in ("test", "test_voting_only"):
        typer.echo(f"[{split}]")
        for name, m in metrics[split].items():
            typer.echo(f"{name:>22}: " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))


if __name__ == "__main__":
    app()
