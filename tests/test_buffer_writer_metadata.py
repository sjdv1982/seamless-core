from seamless import Buffer
from seamless.caching import buffer_writer
from tests.helpers.fake_remotes import install_fake_remotes


def test_hash_type_and_expression_metadata_for_one_checksum_do_not_deduplicate(
    monkeypatch,
):
    calls = []
    install_fake_remotes(monkeypatch, {}, {}, calls, jobserver_available=False)

    checksum = Buffer("metadata source", "str").get_checksum()
    result = Buffer("metadata result", "str").get_checksum()
    buffer_writer.flush()
    calls.clear()
    buffer_writer.register_hash_type(checksum, 17)
    buffer_writer.register_expression_result(
        (checksum.hex(), "value", "str", "str"), result
    )

    buffer_writer.flush()

    assert calls.count("database:set_hash_type") == 1
    assert calls.count("database:set") == 1
