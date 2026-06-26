"""
Unit tests for the media (image upload) layer.

Fully offline — every LLM call is replaced by an injected fake. Covers each
stage in isolation (validation, classification, routing, processing, storage)
plus the pipeline end to end and the LLM client's multimodal contents builder.
"""

from __future__ import annotations

import pytest

from graphrag.config.settings import settings
from graphrag.llm.gemini_client import _build_contents

from app.services.media.classifier import ImageClassifier
from app.services.media.pipeline import MediaPipeline
from app.services.media.processors import DocumentProcessor, PhotoProcessor
from app.services.media.router import route_for_category
from app.services.media.storage import LocalMediaStorage
from app.services.media.types import (
    ImageCategory,
    ImageClassification,
    MediaPart,
    MediaValidationError,
    ProcessingRoute,
    UploadedImage,
)
from app.services.media.validation import validate_image

# Minimal valid magic-byte payloads.
PNG = b"\x89PNG\r\n\x1a\n" + b"rest-of-png"
JPEG = b"\xff\xd8\xff" + b"rest-of-jpeg"

ALLOWED = "image/png,image/jpeg"
MAX = 10 * 1024 * 1024


def _img(data=PNG, mime="image/png") -> UploadedImage:
    return UploadedImage(data=data, mime_type=mime, size_bytes=len(data), filename="x.png")


def _fake_generate(payload: str):
    """An async stand-in for generate_text_async that returns a fixed string."""

    async def _gen(prompt, **kwargs):  # noqa: ANN001 — test double
        return payload

    return _gen


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_accepts_png_and_jpeg():
    for data, mime in ((PNG, "image/png"), (JPEG, "image/jpeg")):
        img = validate_image(
            data=data, mime_type=mime, filename="f", max_bytes=MAX, allowed_mime_types=ALLOWED
        )
        assert img.mime_type == mime
        assert img.size_bytes == len(data)


def test_validate_normalizes_mime_with_charset():
    img = validate_image(
        data=PNG, mime_type="image/png; charset=binary", filename=None,
        max_bytes=MAX, allowed_mime_types=ALLOWED,
    )
    assert img.mime_type == "image/png"


def test_validate_rejects_empty():
    with pytest.raises(MediaValidationError):
        validate_image(data=b"", mime_type="image/png", filename=None, max_bytes=MAX, allowed_mime_types=ALLOWED)


def test_validate_rejects_oversize():
    with pytest.raises(MediaValidationError):
        validate_image(data=PNG, mime_type="image/png", filename=None, max_bytes=4, allowed_mime_types=ALLOWED)


def test_validate_rejects_disallowed_mime():
    with pytest.raises(MediaValidationError):
        validate_image(data=b"%PDF-1.4", mime_type="application/pdf", filename=None, max_bytes=MAX, allowed_mime_types=ALLOWED)


def test_validate_rejects_spoofed_content():
    # Declared png, but bytes are not a png.
    with pytest.raises(MediaValidationError):
        validate_image(data=b"not-an-image", mime_type="image/png", filename=None, max_bytes=MAX, allowed_mime_types=ALLOWED)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category, route",
    [
        (ImageCategory.CLINICAL_PHOTO, ProcessingRoute.MULTIMODAL_LLM),
        (ImageCategory.GENERAL_PHOTO, ProcessingRoute.MULTIMODAL_LLM),
        (ImageCategory.UNKNOWN, ProcessingRoute.MULTIMODAL_LLM),
        (ImageCategory.LAB_REPORT, ProcessingRoute.DOCUMENT_EXTRACTION),
        (ImageCategory.RADIOLOGY_REPORT, ProcessingRoute.DOCUMENT_EXTRACTION),
        (ImageCategory.DOCUMENT, ProcessingRoute.DOCUMENT_EXTRACTION),
        (ImageCategory.OTHER_MEDICAL_DOCUMENT, ProcessingRoute.DOCUMENT_EXTRACTION),
    ],
)
def test_routing_reports_bypass_vision(category, route):
    assert route_for_category(category) == route


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


async def test_local_storage_writes_and_dedupes(tmp_path):
    storage = LocalMediaStorage(str(tmp_path))
    uri1 = await storage.save(data=PNG, mime_type="image/png")
    uri2 = await storage.save(data=PNG, mime_type="image/png")
    assert uri1 == uri2  # content-addressed
    assert uri1.startswith("file://")
    assert len(list(tmp_path.iterdir())) == 1


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


async def test_classifier_parses_category_and_scales_confidence():
    clf = ImageClassifier(generate=_fake_generate('{"category":"lab_report","confidence":92,"rationale":"tables"}'))
    out = await clf.classify(_img())
    assert out.category == ImageCategory.LAB_REPORT
    assert out.confidence == pytest.approx(0.92)


async def test_classifier_unknown_on_bad_json():
    clf = ImageClassifier(generate=_fake_generate("not json"))
    out = await clf.classify(_img())
    assert out.category == ImageCategory.UNKNOWN


async def test_classifier_unknown_on_unrecognized_category():
    clf = ImageClassifier(generate=_fake_generate('{"category":"banana","confidence":0.5}'))
    out = await clf.classify(_img())
    assert out.category == ImageCategory.UNKNOWN


async def test_classifier_survives_llm_error():
    async def _boom(prompt, **kwargs):  # noqa: ANN001
        raise RuntimeError("vision down")

    out = await ImageClassifier(generate=_boom).classify(_img())
    assert out.category == ImageCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------


async def test_photo_processor_sets_caption():
    proc = PhotoProcessor(generate=_fake_generate("Red raised rash on the left forearm."))
    art = await proc.process(_img(), ImageClassification(ImageCategory.CLINICAL_PHOTO, 0.9))
    assert art.route == ProcessingRoute.MULTIMODAL_LLM
    assert "rash" in (art.caption or "")
    assert art.extracted_facts == []


async def test_document_processor_extracts_facts():
    payload = '{"summary":"CBC report","facts":["Hemoglobin 9.1 g/dL (low)","WBC 11.2"]}'
    proc = DocumentProcessor(generate=_fake_generate(payload))
    art = await proc.process(_img(), ImageClassification(ImageCategory.LAB_REPORT, 0.95))
    assert art.route == ProcessingRoute.DOCUMENT_EXTRACTION
    assert art.caption == "CBC report"
    assert "Hemoglobin 9.1 g/dL (low)" in art.extracted_facts


# ---------------------------------------------------------------------------
# Pipeline (end to end, stubbed collaborators)
# ---------------------------------------------------------------------------


def _pipeline(tmp_path, classifier, processors) -> MediaPipeline:
    return MediaPipeline(
        settings=settings,
        storage=LocalMediaStorage(str(tmp_path)),
        classifier=classifier,
        processors=processors,
    )


async def test_pipeline_photo_route_forwards_image(tmp_path):
    classifier = ImageClassifier(generate=_fake_generate('{"category":"clinical_photo","confidence":90}'))
    processors = {
        ProcessingRoute.MULTIMODAL_LLM: PhotoProcessor(generate=_fake_generate("A skin lesion.")),
        ProcessingRoute.DOCUMENT_EXTRACTION: DocumentProcessor(generate=_fake_generate("{}")),
    }
    result = await _pipeline(tmp_path, classifier, processors).process(
        data=PNG, mime_type="image/png", filename="rash.png", query="is this serious?"
    )
    att = result.attachment
    assert att.artifact.category == ImageCategory.CLINICAL_PHOTO
    assert att.parts and isinstance(att.parts[0], MediaPart)  # raw image forwarded
    assert att.artifact.storage_uri and att.artifact.storage_uri.startswith("file://")
    assert result.effective_query == "is this serious?"


async def test_pipeline_document_route_does_not_forward_image(tmp_path):
    classifier = ImageClassifier(generate=_fake_generate('{"category":"lab_report","confidence":95}'))
    processors = {
        ProcessingRoute.MULTIMODAL_LLM: PhotoProcessor(generate=_fake_generate("x")),
        ProcessingRoute.DOCUMENT_EXTRACTION: DocumentProcessor(
            generate=_fake_generate('{"summary":"Lab report","facts":["Glucose 180 mg/dL"]}')
        ),
    }
    result = await _pipeline(tmp_path, classifier, processors).process(
        data=JPEG, mime_type="image/jpeg", filename="labs.jpg", query=None
    )
    att = result.attachment
    assert att.artifact.route == ProcessingRoute.DOCUMENT_EXTRACTION
    assert att.parts == []  # NO raw image to the answer LLM on the document route
    assert "Glucose 180 mg/dL" in att.artifact.extracted_facts
    # Empty query → synthesized document prompt.
    assert "document" in result.effective_query.lower()


async def test_pipeline_invalid_upload_raises(tmp_path):
    classifier = ImageClassifier(generate=_fake_generate("{}"))
    processors = {
        ProcessingRoute.MULTIMODAL_LLM: PhotoProcessor(generate=_fake_generate("x")),
        ProcessingRoute.DOCUMENT_EXTRACTION: DocumentProcessor(generate=_fake_generate("{}")),
    }
    with pytest.raises(MediaValidationError):
        await _pipeline(tmp_path, classifier, processors).process(
            data=b"nope", mime_type="image/png", filename="x", query=None
        )


# ---------------------------------------------------------------------------
# Memory safety — artifacts never carry raw bytes
# ---------------------------------------------------------------------------


def test_artifact_has_no_raw_bytes():
    from app.services.media.types import MediaArtifact

    art = MediaArtifact(
        category=ImageCategory.LAB_REPORT,
        route=ProcessingRoute.DOCUMENT_EXTRACTION,
        mime_type="image/png",
        size_bytes=123,
        caption="CBC",
        extracted_facts=["Hb 9.1"],
    )
    dumped = art.model_dump()
    assert "data" not in dumped and "bytes" not in dumped
    # The memory note is text-only and includes findings, not bytes.
    note = art.to_memory_note()
    assert "Hb 9.1" in note
    assert "CBC" in note


# ---------------------------------------------------------------------------
# LLM client multimodal contents builder
# ---------------------------------------------------------------------------


def test_build_contents_text_only_returns_prompt():
    assert _build_contents("hello", None) == "hello"
    assert _build_contents("hello", []) == "hello"


def test_build_contents_with_media_appends_prompt_last():
    contents = _build_contents("describe", [MediaPart(data=PNG, mime_type="image/png")])
    assert isinstance(contents, list)
    assert len(contents) == 2
    assert contents[-1] == "describe"  # prompt comes after the media part
