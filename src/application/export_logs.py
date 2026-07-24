"""Use case for exporting recognition logs."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..core.domain.face_data import RecognitionEvent
from ..core.ports.repositories import LogRepository

logger = logging.getLogger(__name__)


@dataclass
class ExportLogsInput:
    """Input data for log export."""

    output_path: Path
    format: str = "csv"
    limit: int | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None


@dataclass
class ExportLogsOutput:
    """Output data from log export."""

    output_path: Path
    event_count: int
    message: str


class ExportLogsUseCase:
    """Use case for exporting recognition logs."""

    SUPPORTED_FORMATS = {"csv", "json"}

    def __init__(self, log_repo: LogRepository) -> None:
        self.log_repo = log_repo

    def execute(self, input_data: ExportLogsInput) -> ExportLogsOutput:
        """Execute the log export workflow.

        Args:
            input_data: Export parameters.

        Returns:
            Export result output.

        Raises:
            ValueError: If format is not supported.
        """
        format_lower = input_data.format.lower()
        if format_lower not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{input_data.format}'. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        if input_data.start_date and input_data.end_date:
            events = self.log_repo.get_by_date_range(
                input_data.start_date, input_data.end_date
            )
        elif input_data.limit:
            events = self.log_repo.get_recent(limit=input_data.limit)
        else:
            events = self.log_repo.get_recent(limit=1000)

        input_data.output_path.parent.mkdir(parents=True, exist_ok=True)

        if format_lower == "csv":
            self._export_csv(events, input_data.output_path)
        else:
            self._export_json(events, input_data.output_path)

        logger.info(
            f"Exported {len(events)} log events to {input_data.output_path}"
        )
        return ExportLogsOutput(
            output_path=input_data.output_path,
            event_count=len(events),
            message=(
                f"Successfully exported {len(events)} log events "
                f"to {input_data.output_path}"
            ),
        )

    def _export_csv(self, events: list[RecognitionEvent], output_path: Path) -> None:
        """Export events to CSV format.

        Args:
            events: List of recognition events.
            output_path: Destination file path.
        """
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "event_type",
                    "user_id",
                    "confidence",
                    "image_path",
                    "timestamp",
                ],
            )
            writer.writeheader()
            for event in events:
                writer.writerow(
                    {
                        "id": event.id,
                        "event_type": event.event_type,
                        "user_id": event.user_id if event.user_id else "",
                        "confidence": (
                            f"{event.confidence:.4f}" if event.confidence else ""
                        ),
                        "image_path": str(event.image_path),
                        "timestamp": event.timestamp.isoformat(),
                    }
                )

    def _export_json(
        self, events: list[RecognitionEvent], output_path: Path
    ) -> None:
        """Export events to JSON format.

        Args:
            events: List of recognition events.
            output_path: Destination file path.
        """
        data = []
        for event in events:
            data.append(
                {
                    "id": event.id,
                    "event_type": event.event_type,
                    "user_id": event.user_id,
                    "confidence": event.confidence,
                    "image_path": str(event.image_path),
                    "timestamp": event.timestamp.isoformat(),
                }
            )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
