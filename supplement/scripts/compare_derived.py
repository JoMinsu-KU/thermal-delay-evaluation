"""Compare regenerated content separately from frozen-input byte integrity."""
import json
from pathlib import Path


def assert_derived_equal(reference, regenerated):
    reference, regenerated = Path(reference), Path(regenerated)
    if reference.suffix == '.json':
        expected = json.loads(reference.read_text(encoding='utf-8'))
        actual = json.loads(regenerated.read_text(encoding='utf-8'))
        assert expected == actual, reference.name
        return 'json_content'
    assert reference.read_bytes() == regenerated.read_bytes(), reference.name
    return 'bytes'
