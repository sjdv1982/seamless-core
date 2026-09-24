"""Exercise GC during a writer import against cache reference acquisition."""
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize('operation', ['incref', 'incref_refholder', 'transfer_write', 'register'])
def test_writer_import_can_finalize_expression(tmp_path, operation):
    (tmp_path / 'writer_import.py').write_text(
        'import gc\n'
        'from __main__ import importing, registering\n'
        'importing.set()\n'
        'assert registering.wait(5)\n'
        'gc.collect()\n'
    )
    script = textwrap.dedent('''
        import faulthandler
        import gc
        import sys
        import threading
        from seamless import Buffer, Checksum
        from seamless.expression_class import Expression
        from seamless.caching.buffer_cache import get_buffer_cache
        from seamless.caching import buffer_writer

        faulthandler.dump_traceback_later(5, exit=True)
        gc.disable()
        cache = get_buffer_cache()
        checksum = Checksum('ab' * 32)
        expression = Expression(checksum, '', input_celltype='text', celltype='text')
        cycle = [expression]
        cycle.append(cycle)
        del cycle
        del expression
        buffer = Buffer(b'writer import deadlock')
        target = buffer.get_checksum()
        operation = sys.argv[2]
        if operation == 'register':
            cache.incref_refholder(target)
            cache.strong_cache[target].buffer = None

        importing = threading.Event()
        registering = threading.Event()
        def register(buffer):
            registering.set()
            __import__('writer_import')
        buffer_writer.register = register
        sys.path.insert(0, sys.argv[1])
        writer = threading.Thread(target=lambda: __import__('writer_import'),
                                  name='buffer-writer', daemon=True)
        writer.start()
        assert importing.wait(5)
        if operation == 'register':
            cache.register(target, buffer)
        else:
            getattr(cache, operation)(target, buffer=buffer)
        writer.join(5)
        assert not writer.is_alive()
        assert cache.reference_snapshot().get(checksum, (0, 0, False))[0] == 0
        faulthandler.cancel_dump_traceback_later()
    ''')
    result = subprocess.run(
        [sys.executable, '-c', script, str(tmp_path), operation],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
