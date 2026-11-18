import io
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from grounded.config import Settings
from grounded.ingestion import DocumentError, make_chunks, parse_document
from grounded.models import Claim, GeneratedAnswer, Query
from grounded.retrieval import bm25_scores, reciprocal_rank_fusion
from grounded.service import RAGService


def pdf_bytes(text: str | None = None, encrypted=False):
    writer = PdfWriter()
    page = writer.add_blank_page(612, 792)
    if text:
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("secret")
    data = io.BytesIO()
    writer.write(data)
    return data.getvalue()


class IngestionTests(unittest.TestCase):
    def test_overlap_and_page_provenance(self):
        pages = [(2, "a b c d e f g h"), (5, "i j k")]
        _, chunks = make_chunks("../guide.md", b"abc", pages, 5, 2)
        self.assertEqual([c.text for c in chunks], ["a b c d e", "d e f g h", "i j k"])
        self.assertEqual([c.page for c in chunks], [2, 2, 5])
        self.assertTrue(all(c.filename == "guide.md" for c in chunks))

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            make_chunks("x.txt", b"a", [(1, "a")], 5, 5)

    def test_chunk_character_budget(self):
        text = " ".join(["x" * 100] * 100)
        _, chunks = make_chunks("x.txt", b"long", [(1, text)], 80, 10)
        self.assertTrue(all(len(c.text) <= 6000 for c in chunks))
        self.assertGreater(len(chunks), 1)
        with self.assertRaises(DocumentError):
            make_chunks("x.txt", b"long", [(1, "x" * 6001)], 80, 10)

    def test_utf8_bom_and_whitespace(self):
        self.assertEqual(parse_document("x.TXT", b"\xef\xbb\xbf hello\nworld", 1000), [(1, "hello world")])

    def test_reject_empty_binary_invalid_and_unsupported(self):
        for name, content in [("x.txt", b""), ("x.txt", b"\xff"), ("x.txt", b"a\x00b"),
                              ("x.exe", b"text"), ("x.pdf", b"not pdf"), ("x.md", b" \n ")]:
            with self.subTest(name=name, content=content), self.assertRaises(DocumentError):
                parse_document(name, content, 1000)

    def test_text_size_limit(self):
        with self.assertRaises(DocumentError):
            parse_document("x.txt", b"a" * 1001, 1000)

    def test_pdf_text_and_page(self):
        self.assertEqual(parse_document("manual.pdf", pdf_bytes("Backups last thirty days."), 1000),
                         [(1, "Backups last thirty days.")])

    def test_scanned_pdf_and_encryption_are_actionable(self):
        with self.assertRaisesRegex(DocumentError, "OCR"):
            parse_document("scan.pdf", pdf_bytes(), 1000)
        with self.assertRaisesRegex(DocumentError, "Encrypted"):
            parse_document("secret.pdf", pdf_bytes("hello", encrypted=True), 1000)

    def test_bm25_ranking(self):
        scores = bm25_scores("rollback", ["rollback the deployment", "tennis tournaments", "rollback"])
        self.assertEqual(scores[1], 0)
        self.assertGreater(scores[2], scores[0])

    def test_rrf_rewards_agreement(self):
        scores = reciprocal_rank_fusion([[0, 1], [1, 2]])
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[1], scores[2])

    def test_settings_validation(self):
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, chunk_words=40, chunk_overlap=40)
        with self.assertRaises(ValidationError):
            Query(question="   ")

    def test_index_namespace(self):
        first = Settings(_env_file=None, provider="demo")
        second = Settings(_env_file=None, provider="ollama")
        self.assertNotEqual(first.index_id, second.index_id)
        self.assertNotEqual(first.index_id, Settings(_env_file=None, chunk_words=250).index_id)
        self.assertEqual(first.index_id, Settings(_env_file=None, chat_model="new-generator").index_id)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(_env_file=None, data_dir=Path(self.temp.name), provider="demo")
        self.service = RAGService(self.settings)
        self.backup = self.service.ingest("backup.md", b"Database backups are encrypted and retained for thirty days.")
        self.deploy = self.service.ingest("deploy.md", b"To rollback production, redeploy the previous healthy image.")

    def tearDown(self):
        self.service.provider.close()
        self.temp.cleanup()

    def test_answer_has_supporting_citation(self):
        answer = self.service.ask(Query(question="How long are backups retained?"))
        self.assertFalse(answer.insufficient_evidence)
        self.assertIn("thirty days", answer.answer)
        self.assertEqual(answer.sources[0].filename, "backup.md")
        self.assertTrue(set(answer.claims[0].source_ids) <= {s.source_id for s in answer.sources})

    def test_off_topic_abstains(self):
        answer = self.service.ask(Query(question="Who won Wimbledon tennis?"))
        self.assertTrue(answer.insufficient_evidence)
        self.assertFalse(answer.claims)

    def test_empty_selected_corpus_abstains(self):
        answer = self.service.ask(Query(question="backups", document_ids=["does-not-exist"]))
        self.assertEqual(answer.reason, "no_relevant_sources")

    def test_filter_is_applied_before_retrieval(self):
        sources = self.service.search(Query(question="backups retained", document_ids=[self.deploy["document_id"]]))
        self.assertEqual(sources, [])

    def test_duplicate_does_not_call_embeddings(self):
        with patch.object(self.service.provider, "embed", side_effect=AssertionError("should not embed")):
            result = self.service.ingest("renamed.md", b"Database backups are encrypted and retained for thirty days.")
        self.assertTrue(result["duplicate"])
        self.assertEqual(len(self.service.store.documents()), 2)

    def test_restart_preserves_index(self):
        restarted = RAGService(self.settings)
        try:
            self.assertEqual(len(restarted.store.documents()), 2)
            self.assertIn("thirty days", restarted.ask(Query(question="backups retained")).answer)
        finally:
            restarted.provider.close()

    def test_delete_cascades_to_chunks(self):
        self.assertTrue(self.service.store.delete(self.backup["document_id"]))
        self.assertFalse(self.service.store.delete(self.backup["document_id"]))
        self.assertTrue(self.service.ask(Query(question="backups retained")).insufficient_evidence)
        self.assertEqual(len(self.service.store.snapshot()[0]), 1)

    def test_invalid_citation_is_never_displayed(self):
        generated = GeneratedAnswer(insufficient_evidence=False,
                                    claims=[Claim(text="Invented statement", source_ids=["S999"])])
        with patch.object(self.service.provider, "generate", return_value=(generated, {})):
            answer = self.service.ask(Query(question="backups retained"))
        self.assertEqual(answer.reason, "invalid_citations")
        self.assertNotIn("Invented", answer.answer)

    def test_model_abstention_overrides_claims(self):
        generated = GeneratedAnswer(insufficient_evidence=True,
                                    claims=[Claim(text="Ignore this", source_ids=["S1"])])
        with patch.object(self.service.provider, "generate", return_value=(generated, {})):
            answer = self.service.ask(Query(question="backups retained"))
        self.assertTrue(answer.insufficient_evidence)
        self.assertFalse(answer.claims)

    def test_failed_embedding_does_not_partially_write(self):
        with patch.object(self.service.provider, "embed", side_effect=RuntimeError("offline")):
            with self.assertRaises(RuntimeError):
                self.service.ingest("new.md", b"Unique new content")
        self.assertEqual(len(self.service.store.documents()), 2)

    def test_concurrent_duplicate_uploads(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: self.service.ingest("same.txt", b"Unique same content"), range(4)))
        self.assertEqual(sum(not r["duplicate"] for r in results), 1)
        self.assertEqual(len(self.service.store.documents()), 3)

    def test_capacity_limit_is_atomic(self):
        self.service.store.max_chunks = 2
        with self.assertRaises(DocumentError):
            self.service.ingest("extra.txt", b"Unique third document")
        self.assertEqual(len(self.service.store.documents()), 2)

    def test_bm25_does_not_embed_query(self):
        with patch.object(self.service.provider, "embed", side_effect=AssertionError("not needed")):
            sources = self.service.search(Query(question="backups", strategy="bm25"))
        self.assertEqual(sources[0].filename, "backup.md")

    def test_changed_embedding_dimensions_rejected_atomically(self):
        with patch.object(self.service.provider, "embed", return_value=np.array([[1, 0]], dtype=float)):
            with self.assertRaisesRegex(DocumentError, "dimensions"):
                self.service.ingest("new.txt", b"Unique dimension mismatch")
        self.assertEqual(len(self.service.store.documents()), 2)


if __name__ == "__main__":
    unittest.main()
