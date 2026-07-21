"""encrypt face embeddings

Revision ID: a1b2c3d4e5f6
Revises: e513729df8ea
Create Date: 2026-07-21 09:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'e513729df8ea'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index('ix_face_embeddings_embedding', table_name='face_embeddings')
    op.alter_column('face_embeddings', 'embedding', existing_type=Vector(512), type_=sa.Text(), existing_nullable=False)


def downgrade() -> None:
    op.alter_column('face_embeddings', 'embedding', existing_type=sa.Text(), type_=Vector(512), existing_nullable=False)
    op.create_index('ix_face_embeddings_embedding', 'face_embeddings', ['embedding'], unique=False, postgresql_using='hnsw', postgresql_ops={'embedding': 'vector_cosine_ops'})
