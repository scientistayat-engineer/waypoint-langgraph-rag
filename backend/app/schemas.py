from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    k: int | None = Field(default=None, ge=1, le=6)


class AddDocumentRequest(BaseModel):
    text: str = Field(min_length=3, max_length=20_000)
    source: str = Field(default="pasted", max_length=60)
