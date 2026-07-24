"""
CLI commands for the AI Face Recognition System.

This module provides all terminal commands using Click framework.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ..application.export_logs import ExportLogsInput
from ..application.recognize import RecognizeInput
from ..application.register_user import RegisterUserInput
from ..application.train_model import TrainModelInput
from ..core.domain.exceptions import (
    CameraUnavailableError,
    FaceRecognitionError,
    InsufficientFaceDataError,
    UserAlreadyExistsError,
    UserNotFoundError,
)
from . import create_services

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(package_name="ai-face-recognition-system")
@click.option(
    "--config",
    "-c",
    type=click.Path(path_type=Path, exists=True),
    help="Path to .env configuration file",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity (use -v, -vv, or -vvv)",
)
def main(config: Path | None, verbose: int) -> None:
    """AI Face Recognition System - CLI terminal application."""
    log_level = "WARNING"
    if verbose == 1:
        log_level = "INFO"
    elif verbose >= 2:
        log_level = "DEBUG"

    from ..infrastructure.config import Config
    from ..infrastructure.logging_config import setup_logging
    app_config = Config(env_file=config)
    setup_logging(level=log_level, log_file=app_config.logs_dir / "app.log")
    logger.info(f"Starting {app_config.app_name} v{app_config.app_version}")


def _get_services(config_path: Path | None):
    """Create and return service container."""
    from ..infrastructure.config import Config
    config = Config(env_file=config_path)

    return create_services(config)


@main.command()
@click.argument("name")
@click.option(
    "--images",
    "-i",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Paths to face images",
)
@click.option(
    "--min-faces",
    default=1,
    help="Minimum number of faces required",
)
@click.pass_context
def register(
    ctx: click.Context, name: str, images: tuple[Path, ...], min_faces: int
) -> None:
    """Register a new user in the system.

    Example:
        face-recognition register "John Doe" --images ./photos/john1.jpg ./photos/john2.jpg
    """
    try:
        services = _get_services(ctx.parent.params.get("config"))
        use_case = services.register_user_use_case

        if not images:
            console.print("[red]Error:[/red] No images provided.")
            console.print("Use --images option to specify face image paths.")
            sys.exit(1)

        image_paths = list(images)
        input_data = RegisterUserInput(
            name=name, image_paths=image_paths, min_faces_required=min_faces
        )
        result = use_case.execute(input_data)

        console.print(f"[green bold]Success:[/green bold] {result.message}")
        console.print(f"[cyan]User ID:[/cyan] {result.user.id}")
        console.print(f"[cyan]Faces Registered:[/cyan] {result.faces_registered}")

    except UserAlreadyExistsError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except InsufficientFaceDataError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except FaceRecognitionError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        logger.exception("Unexpected error during registration")
        sys.exit(1)


@main.command()
@click.argument("user_name")
@click.option(
    "--count",
    "-n",
    default=5,
    help="Number of images to capture",
)
@click.option(
    "--no-preview",
    is_flag=True,
    help="Disable camera preview window",
)
@click.option(
    "--delay",
    default=0.5,
    help="Delay between captures in seconds",
)
@click.pass_context
def capture(
    ctx: click.Context,
    user_name: str,
    count: int,
    no_preview: bool,
    delay: float,
) -> None:
    """Capture face images from webcam for a user.

    Example:
        face-recognition capture "John Doe" --count 10
    """
    try:
        services = _get_services(ctx.parent.params.get("config"))
        use_case = services.capture_images_use_case

        from ..application.capture_images import CaptureImagesInput
        input_data = CaptureImagesInput(
            user_name=user_name,
            count=count,
            display_preview=not no_preview,
            delay_seconds=delay,
        )
        result = use_case.execute(input_data)

        console.print(f"[green]Success:[/green] {result.message}")
        for i, path in enumerate(result.saved_paths, 1):
            console.print(f"  [dim]{i}. {path}[/dim]")

    except UserNotFoundError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except CameraUnavailableError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except FaceRecognitionError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        logger.exception("Unexpected error during capture")
        sys.exit(1)


@main.command()
@click.pass_context
def train(ctx: click.Context) -> None:
    """Train/update the face recognition model."""
    try:
        services = _get_services(ctx.parent.params.get("config"))

        input_data = TrainModelInput()
        result = services.train_model_use_case.execute(input_data)

        console.print(f"[green]Training complete:[/green] {result.message}")
        console.print(f"[cyan]Users trained:[/cyan] {result.users_trained}")
        console.print(f"[cyan]Total encodings:[/cyan] {result.total_encodings}")

    except FaceRecognitionError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        logger.exception("Unexpected error during training")
        sys.exit(1)


@main.command()
@click.option(
    "--source",
    type=click.Choice(["camera", "image"]),
    default="camera",
    help="Input source",
)
@click.option(
    "--image-file",
    type=click.Path(exists=True, path_type=Path),
    help="Image file path (when source=image)",
)
@click.option(
    "--tolerance",
    default=0.5,
    help="Face matching tolerance (lower = stricter)",
)
@click.option(
    "--save-unknowns",
    is_flag=True,
    default=True,
    help="Save unknown face images",
)
@click.pass_context
def recognize(
    ctx: click.Context,
    source: str,
    image_file: Path | None,
    tolerance: float,
    save_unknowns: bool,
) -> None:
    """Start live face recognition from webcam or image.

    Example:
        face-recognition recognize --source camera
        face-recognition recognize --source image --image-file ./test.jpg
    """
    try:
        services = _get_services(ctx.parent.params.get("config"))
        use_case = services.recognize_use_case

        input_data = RecognizeInput(
            source=source,
            image_path=image_file,
            tolerance=tolerance,
            save_unknown=save_unknowns,
        )
        result = use_case.execute(input_data)

        console.print(f"[green]{result.message}[/green]")
        console.print(f"[cyan]Recognized:[/cyan] {len(result.recognized_users)} faces")
        console.print(f"[cyan]Unknown:[/cyan] {result.unknown_faces} faces")

        for user, confidence in result.recognized_users:
            console.print(
                f"  - [bold]{user.name}[/bold] (confidence: {confidence:.2%})"
            )

    except CameraUnavailableError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except FaceRecognitionError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        logger.exception("Unexpected error during recognition")
        sys.exit(1)


@main.command()
@click.pass_context
def unknown(ctx: click.Context) -> None:
    """Identify and permit isolating unknown faces."""
    try:
        services = _get_services(ctx.parent.params.get("config"))
        repo = services.user_repo
        users = repo.list_all()
        console.print(f"[green]Registered users:[/green] {len(users)}")
        for user in users:
            console.print(f"  - {user.name} (ID: {user.id})")

    except FaceRecognitionError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.pass_context
def list_users(ctx: click.Context) -> None:
    """List all registered users."""
    try:
        services = _get_services(ctx.parent.params.get("config"))
        use_case = services.manage_users_use_case

        result = use_case.list_users()

        if not result.users:
            console.print("[yellow]No users registered.[/yellow]")
            return

        table = Table(title="Registered Users", show_lines=True)
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("Name", style="bold")
        table.add_column("Faces", justify="right")
        table.add_column("Created", style="dim")

        for user in result.users:
            table.add_row(
                str(user.id),
                user.name,
                str(user.face_count),
                user.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)
        console.print(f"\n[dim]Total: {result.total} user(s)[/dim]")

    except FaceRecognitionError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.argument("identifier")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Skip confirmation prompt",
)
@click.pass_context
def delete_user(
    ctx: click.Context, identifier: str, force: bool
) -> None:
    """Delete a registered user and all their data.

    Example:
        face-recognition delete-user "John Doe"
        face-recognition delete-user 1
    """
    try:
        services = _get_services(ctx.parent.params.get("config"))
        use_case = services.manage_users_use_case

        user = use_case.get_user(identifier)
        if not user:
            console.print(f"[red]Error:[/red] User '{identifier}' not found.")
            sys.exit(1)

        if not force:
            confirm = click.confirm(
                f"Delete user '{user.name}' (ID: {user.id}) and all associated data?"
            )
            if not confirm:
                console.print("[yellow]Cancelled.[/yellow]")
                return

        result = use_case.delete_user(identifier)
        console.print(f"[green]{result.message}[/green]")

    except UserNotFoundError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except FaceRecognitionError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--limit",
    "-n",
    default=20,
    help="Maximum number of log entries to display",
)
@click.pass_context
def logs(ctx: click.Context, limit: int) -> None:
    """Display recent recognition logs."""
    try:
        services = _get_services(ctx.parent.params.get("config"))
        log_repo = services.log_repo

        events = log_repo.get_recent(limit=limit)

        if not events:
            console.print("[yellow]No logs found.[/yellow]")
            return

        table = Table(title="Recognition Logs", show_lines=True)
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("Type", style="bold")
        table.add_column("User ID", justify="right")
        table.add_column("Confidence")
        table.add_column("Timestamp", style="dim")

        for event in events:
            event_type_color = (
                "green" if event.event_type == "recognized" else "red"
            )
            confidence = (
                f"{event.confidence:.2%}" if event.confidence is not None else "N/A"
            )
            table.add_row(
                str(event.id),
                f"[{event_type_color}]{event.event_type}[/{event_type_color}]",
                str(event.user_id) if event.user_id else "N/A",
                confidence,
                event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            )

        console.print(table)

    except FaceRecognitionError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output file path",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["csv", "json"]),
    default="csv",
    help="Export format",
)
@click.option(
    "--limit",
    default=1000,
    help="Maximum number of entries to export",
)
@click.option(
    "--start-date",
    type=click.DateTime(),
    help="Start date (ISO format)",
)
@click.option(
    "--end-date",
    type=click.DateTime(),
    help="End date (ISO format)",
)
@click.pass_context
def export_logs(
    ctx: click.Context,
    output: Path,
    format: str,
    limit: int,
    start_date: datetime | None,
    end_date: datetime | None,
) -> None:
    """Export recognition logs to CSV or JSON.

    Example:
        face-recognition export-logs --output logs.csv --format csv
        face-recognition export-logs --output logs.json --format json --limit 500
    """
    try:
        services = _get_services(ctx.parent.params.get("config"))
        use_case = services.export_logs_use_case

        input_data = ExportLogsInput(
            output_path=output,
            format=format,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
        )
        result = use_case.execute(input_data)
        console.print(f"[green]Success:[/green] {result.message}")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except FaceRecognitionError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        logger.exception("Unexpected error during export")
        sys.exit(1)


@main.command()
@click.pass_context
def clean_unknowns(ctx: click.Context) -> None:
    """Clear unknown face detection entries."""
    try:
        services = _get_services(ctx.parent.params.get("config"))
        log_repo = services.log_repo

        count = log_repo.delete_by_event_type("unknown")
        console.print(f"[green]Removed {count} unknown face entries.[/green]")

    except FaceRecognitionError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Display system information and status."""
    try:
        services = _get_services(ctx.parent.params.get("config"))
        user_repo = services.user_repo
        face_encoding_repo = services.face_encoding_repo
        log_repo = services.log_repo

        users = user_repo.list_all()
        all_encodings = face_encoding_repo.get_all_encodings()
        recent_logs = log_repo.get_recent(limit=5)

        console.print(f"\n[bold cyan]{services.config.app_name}[/bold cyan]")
        console.print(f"Version: {services.config.app_version}")
        console.print(f"Database: {services.config.database_path}")
        console.print(f"Encodings Dir: {services.config.encodings_dir}")
        console.print(f"Logs Dir: {services.config.logs_dir}")
        console.print("\n[bold]Statistics:[/bold]")
        console.print(f"  Registered users: {len(users)}")
        console.print(f"  Total face encodings: {len(all_encodings)}")
        console.print(f"  Recent log entries: {len(recent_logs)}")
        console.print("")

    except FaceRecognitionError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        sys.exit(1)
