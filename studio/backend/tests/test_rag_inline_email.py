# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inline forwarded messages are body content, including in a multipart/digest."""

import io
from email import policy
from email.message import EmailMessage

import pytest
from fastapi import UploadFile

from core.rag import ingestion, parsers, store
from routes.rag import _save_upload
from storage import rag_db


def _original(kind = "plain"):
    message = EmailMessage()
    message["From"] = "Author <author@example.com>"
    message["Subject"] = "Original résumé"
    if kind == "html":
        message.set_content("<p>inlinezebramarker: reçu.</p>", subtype = "html")
    else:
        message.set_content("inlinezebramarker: reçu.")
        if kind == "alternative":
            message.add_alternative("<p>duplicatehtmlmarker</p>", subtype = "html")
    return message


def _forward(original, disposition = "inline"):
    message = EmailMessage()
    message["Subject"] = "Forwarded report"
    message.set_content("Before the original.")
    message.make_mixed()
    inline = EmailMessage()
    inline.set_content(original)
    if disposition:
        inline.add_header("Content-Disposition", disposition, filename = "original.eml")
    message.attach(inline)
    after = EmailMessage()
    after.set_content("After the original.")
    message.attach(after)
    return message


def _save(path, message):
    path.write_bytes(message.as_bytes(policy = policy.SMTP))
    return path


def _text(path):
    return "\n".join(page.text for page in parsers.parse(str(path)))


@pytest.mark.parametrize("kind", ["plain", "html", "alternative"])
@pytest.mark.parametrize("disposition", [None, "inline"])
def test_inline_forward_keeps_original_headers_and_body(tmp_path, kind, disposition):
    text = _text(_save(tmp_path / "forward.eml", _forward(_original(kind), disposition)))
    assert "Subject: Original résumé" in text
    assert "From: Author <author@example.com>" in text
    assert text.count("inlinezebramarker: reçu.") == 1
    assert text.index("Before the original.") < text.index("inlinezebramarker")
    assert text.index("inlinezebramarker") < text.index("After the original.")
    assert "duplicatehtmlmarker" not in text


def test_nested_inline_forwards_keep_the_innermost_message(tmp_path):
    nested = _forward(_original())
    nested.replace_header("Subject", "Intermediate forward")
    text = _text(_save(tmp_path / "nested.eml", _forward(nested)))
    assert text.count("Subject: Original résumé") == 1
    assert text.count("Subject: Intermediate forward") == 1
    assert text.count("inlinezebramarker") == 1


def test_digest_parts_with_the_default_rfc822_type_are_read(tmp_path):
    first, second = _original(), _original()
    second.replace_header("Subject", "Second original")
    second.set_content("seconddigestmarker")
    body = (
        b"Subject: Daily digest\r\nMIME-Version: 1.0\r\n"
        b'Content-Type: multipart/digest; boundary="digest-boundary"\r\n\r\n'
    )
    for message in (first, second):
        body += b"--digest-boundary\r\n\r\n" + message.as_bytes(policy = policy.SMTP) + b"\r\n"
    body += b"--digest-boundary--\r\n"
    path = tmp_path / "digest.eml"
    path.write_bytes(body)
    text = _text(path)
    assert "Subject: Original résumé" in text and "Subject: Second original" in text
    assert text.index("inlinezebramarker") < text.index("seconddigestmarker")


def test_forwarded_message_marked_as_an_attachment_stays_an_attachment(tmp_path):
    text = _text(_save(tmp_path / "attachment.eml", _forward(_original(), "attachment")))
    assert "Attachments: original.eml" in text
    assert "inlinezebramarker" not in text
    assert "Before the original." in text and "After the original." in text


def test_external_body_message_is_not_read_as_an_inline_forward(tmp_path):
    external = EmailMessage()
    external.set_type("message/external-body")
    external.set_payload("externalbodymarker")
    text = _text(_save(tmp_path / "external.eml", _forward(external)))
    assert "externalbodymarker" not in text


def test_uploaded_inline_forward_is_lexically_searchable(rag_home, stub_embeddings, tmp_path):
    if not rag_db.rag_available():
        pytest.skip(reason = "sqlite-vec unavailable here, so there is no lexical index to search")
    source = _save(tmp_path / "searchable.eml", _forward(_original()))
    stored, filename, _ = _save_upload(
        UploadFile(file = io.BytesIO(source.read_bytes()), filename = source.name)
    )
    scope = store.thread_scope("inline-email")
    document, job = ingestion.start_ingestion(
        scope, None, "inline-email", filename, stored, background = False
    )
    assert ingestion.get_job_status(job)["status"] == "completed"
    conn = rag_db.get_connection()
    try:
        assert store.get_document(conn, document)["status"] == "completed"
        assert store.search_lexical(conn, scope, "inlinezebramarker", 5)
    finally:
        conn.close()
