from typing import Annotated

from fastapi import Depends

from sem_analyzer_server.editor.stacking.schemas import StackingResponseSchema

from ...database import Database, get_database


class StackingService:
    def __init__(self, db: Annotated[Database, Depends(get_database)]):
        self.db = db

    def example_method(self, example_value: str) -> StackingResponseSchema:
        return StackingResponseSchema(value1=example_value, value2=example_value.encode("utf-8"))
