"""Task 14 - the Final-Editor's structured output."""
from pydantic import BaseModel, Field


class Verdict(BaseModel):
    approved: bool = Field(description="True when the draft is fully supported and is returned unchanged.")
    final_answer: str = Field(description="The answer to send: the draft itself, or a corrected version.")
    reason: str = Field(description="Why it was approved, or what was removed and why.")