from alembic import context
from shar.config import load_settings
from shar.storage.database import engine_for
from shar.storage.models import metadata

engine = engine_for(load_settings().storage_root)
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=metadata)
    with context.begin_transaction():
        context.run_migrations()
