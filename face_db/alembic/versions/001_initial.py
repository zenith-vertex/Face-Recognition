"""Initial migration for face recognition database"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "persons",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=150), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False, server_default="unspecified"),
        sa.Column("registration_date", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_persons_id"), "persons", ["id"], unique=False)

    op.create_table(
        "face_embeddings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("person_id", sa.Integer(), nullable=False),
        sa.Column("embedder_model", sa.String(length=30), nullable=False),
        sa.Column("embedding", postgresql.VECTOR(dim=128), nullable=False),
        sa.Column("source_image_path", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["person_id"], ["persons.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_face_embeddings_id"), "face_embeddings", ["id"], unique=False)
    op.create_index(op.f("idx_face_embeddings_person_id"), "face_embeddings", ["person_id"], unique=False)

    op.execute("CREATE INDEX idx_face_embeddings_ann ON face_embeddings USING hnsw (embedding)")

    op.create_table(
        "recognition_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("matched_person_id", sa.Integer(), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("metric_used", sa.String(length=20), nullable=False),
        sa.Column("detector_used", sa.String(length=20), nullable=True),
        sa.Column("embedder_used", sa.String(length=20), nullable=True),
        sa.Column("decision", sa.String(length=10), nullable=False),
        sa.Column("source_camera", sa.String(length=100), nullable=True),
        sa.Column("frame_reference", sa.Text(), nullable=True),
        sa.Column("raw_bbox", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["matched_person_id"], ["persons.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_recognition_logs_id"), "recognition_logs", ["id"], unique=False)
    op.create_index(op.f("idx_recognition_logs_timestamp"), "recognition_logs", ["timestamp"], unique=False)
    op.create_index(op.f("idx_recognition_logs_matched_person"), "recognition_logs", ["matched_person_id"], unique=False)


def downgrade():
    op.drop_table("recognition_logs")
    op.drop_table("face_embeddings")
    op.drop_table("persons")
    op.execute("DROP EXTENSION IF EXISTS vector")