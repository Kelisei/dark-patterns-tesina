from pydantic import BaseModel, Field


class TextItem(BaseModel):
    text: str = Field(..., description="Texto a analizar.")
    path: str | None = Field(None, description="Ruta asociada al texto.")
    id: str | None = Field(None, description="Identificador opcional del texto.")


class ScarcityRequest(BaseModel):
    version: str = Field(..., description="Versión del esquema.")
    texts: list[TextItem]


class ScarcityInstance(BaseModel):
    text: str
    path: str | None = None
    id: str | None = None
    has_scarcity: bool


class ScarcityResponse(BaseModel):
    version: str
    instances: list[ScarcityInstance]


class UrgencyRequest(BaseModel):
    version: str = Field(..., description="Versión del esquema.")
    texts: list[TextItem]


class UrgencyInstance(BaseModel):
    text: str
    has_urgency: bool
    id: str | None = None
    path: str | None = None


class UrgencyResponse(BaseModel):
    version: str
    urgency_instances: list[UrgencyInstance]


class ShamingToken(BaseModel):
    text: str
    id: str | None = None
    path: str | None = None


class ShamingRequestV1(BaseModel):
    Version: str = Field("0.1", description="Versión del esquema.")
    tokens: list[ShamingToken]


class ShamingRequestV2(BaseModel):
    Version: str = Field(..., description="Versión del esquema (ej. '0.2').")
    texts: list[TextItem]


class ShamingInstanceV1(BaseModel):
    text: str
    path: str | None = None
    id: str | None = None


class ShamingInstanceV2(BaseModel):
    text: str
    path: str | None = None
    id: str | None = None
    has_shaming: bool


class ShamingResponseV2(BaseModel):
    version: str
    instances: list[ShamingInstanceV2]


class GenericDetectRequest(BaseModel):
    texts: list[TextItem] | None = None
    tokens: list[TextItem] | None = None


class GenericDetectInstance(BaseModel):
    text: str
    path: str | None = None
    id: str | None = None
    detected: bool
    labels: list[str]


class GenericDetectResponse(BaseModel):
    version: str
    instances: list[GenericDetectInstance]
