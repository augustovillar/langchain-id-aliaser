import pytest

from langchain_id_aliaser import IdAliaser, UnknownAliasError

UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
UUID2 = "550e8400-e29b-41d4-a716-446655440000"


def test_uuid_detected_and_aliased_shorter():
    a = IdAliaser()
    out = a.aliasify(f"hello {UUID}")
    assert UUID not in out
    alias = out.replace("hello ", "")
    assert alias.startswith("id_")
    assert len(alias) < len(UUID)


def test_roundtrip_identity_string():
    a = IdAliaser()
    text = f"users {UUID} and {UUID2} are linked"
    assert a.restore(a.aliasify(text)) == text


def test_roundtrip_nested_dict_and_list():
    a = IdAliaser()
    obj = {"id": UUID, "items": [{"ref": UUID2}, "noise", 42]}
    assert a.restore(a.aliasify(obj)) == obj


def test_deterministic_same_uuid_same_alias():
    a = IdAliaser()
    first = a.aliasify(UUID)
    second = a.aliasify(f"again {UUID}")
    assert second == f"again {first}"


def test_multi_turn_stability_across_calls():
    a = IdAliaser()
    turn1 = a.aliasify(UUID)
    a.aliasify("unrelated text with no ids")
    turn3 = a.aliasify(f"see {UUID}")
    assert turn3 == f"see {turn1}"


def test_register_value_with_type_prefix():
    a = IdAliaser()
    a.register(value="cust_8f3k2", type="cust")
    out = a.aliasify("customer cust_8f3k2 here")
    assert "cust_8f3k2" not in out
    assert "cust_" in out
    assert a.restore(out) == "customer cust_8f3k2 here"


def test_register_pattern():
    a = IdAliaser(detect_uuids=False)
    a.register(pattern=r"order_\d+", type="ord")
    text = "order_991 and order_992"
    out = a.aliasify(text)
    assert "order_991" not in out and "order_992" not in out
    assert a.restore(out) == text


def test_ordinal_mode():
    a = IdAliaser(mode="ordinal")
    out = a.aliasify(f"{UUID} {UUID2}")
    assert out == "id1 id2"
    assert a.restore(out) == f"{UUID} {UUID2}"


def test_collision_resolution_keeps_unique_and_reversible(monkeypatch):
    # Force every value to the same short hash code so collisions must lengthen.
    import langchain_id_aliaser.core as core

    monkeypatch.setattr(core, "_base62", lambda num: "AAAAAAAA")
    a = IdAliaser()
    out = a.aliasify(f"{UUID} {UUID2}")
    parts = out.split()
    assert parts[0] != parts[1]  # collision was broken
    assert a.restore(out) == f"{UUID} {UUID2}"


def test_forgiving_restore_passes_unknown_through():
    a = IdAliaser()
    a.aliasify(UUID)  # populate mapping
    text = "known nothing here id_zzzzz extra"
    assert a.restore(text) == text  # unknown alias untouched, no error


def test_strict_restore_raises_on_unknown_alias():
    a = IdAliaser()
    a.aliasify(UUID)  # establishes alias shape/prefix
    with pytest.raises(UnknownAliasError):
        a.restore("the model said id_zzzzzz oops", strict=True)


def test_strict_restore_ok_for_known_alias():
    a = IdAliaser()
    aliased = a.aliasify(UUID)
    assert a.restore(aliased, strict=True) == UUID


def test_export_load_mapping_shares_state():
    a = IdAliaser()
    aliased = a.aliasify(UUID)
    b = IdAliaser()
    b.load_mapping(a.export_mapping())
    assert b.restore(aliased) == UUID
