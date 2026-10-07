from backend.app.services.chunking import split_document


def test_short_document_remains_one_chunk():
    chunks = split_document("标题\n\n五花海适合拍摄倒影。")
    assert len(chunks) == 1
    assert chunks[0].index == 0


def test_long_document_splits_on_paragraphs_and_keeps_overlap():
    text = "\n\n".join(f"## 第{i}节\n" + ("五花海、珍珠滩和镜海的游览提示。" * 30) for i in range(1, 5))
    chunks = split_document(text, target_size=300, overlap=40, max_size=420)
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 420 for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert any(set(chunks[i].content[-40:]) & set(chunks[i + 1].content[:40]) for i in range(len(chunks) - 1))


def test_oversized_sentence_is_hard_split_without_empty_chunks():
    chunks = split_document("五花海" * 1000, target_size=100, overlap=20, max_size=140)
    assert chunks
    assert all(0 < len(chunk.content) <= 140 for chunk in chunks)

