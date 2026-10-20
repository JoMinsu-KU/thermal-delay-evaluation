"""Regression checks for JSON line endings and genuine result changes."""
from pathlib import Path
import json
import tempfile
from compare_derived import assert_derived_equal


def main():
    with tempfile.TemporaryDirectory(prefix='derived-check-') as folder:
        p=Path(folder); ref=p/'reference.json'; generated=p/'generated.json'
        value={'count':324,'range':[0.478,2.276],'completed':True}
        text=json.dumps(value,indent=2)+'\n'
        ref.write_bytes(text.replace('\n','\r\n').encode('utf8'))
        generated.write_bytes(text.encode('utf8'))
        assert ref.read_bytes()!=generated.read_bytes()
        assert assert_derived_equal(ref,generated)=='json_content'
        generated.write_text(json.dumps({**value,'count':325}),encoding='utf8')
        try:
            assert_derived_equal(ref,generated)
        except AssertionError:
            pass
        else:
            raise AssertionError('A changed numerical result was accepted')
        ref=p/'reference.csv';generated=p/'generated.csv'
        ref.write_bytes(b'count\n324\n');generated.write_bytes(ref.read_bytes())
        assert assert_derived_equal(ref,generated)=='bytes'
        generated.write_bytes(b'count\n325\n')
        try:
            assert_derived_equal(ref,generated)
        except AssertionError:
            pass
        else:
            raise AssertionError('A changed CSV was accepted')
    print('PASS: JSON CRLF/LF equivalence; changed JSON and CSV values rejected.')


if __name__=='__main__':
    main()
