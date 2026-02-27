from pydantic import BaseModel
from typing import Union, List, TypeVar, Generic
from pydantic.generics import GenericModel
from datetime import datetime

T = TypeVar("T")


class ApiResponse(GenericModel, Generic[T]):
    status: int = 0
    message: str = ''
    data: T
    ts: float = datetime.now().timestamp()

