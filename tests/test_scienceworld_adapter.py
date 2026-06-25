from acta.envs.scienceworld_adapter import (
    parse_scienceworld_task_id,
    remaining_gold_suffix,
)


def test_parse_scienceworld_task_id_defaults() -> None:
    assert parse_scienceworld_task_id("boil") == ("boil", 0, "easy")
    assert parse_scienceworld_task_id("boil:2") == ("boil", 2, "easy")
    assert parse_scienceworld_task_id("scienceworld://boil/3/easy") == (
        "boil",
        3,
        "easy",
    )


def test_remaining_gold_suffix_tracks_ordered_progress() -> None:
    gold = ("open door", "go kitchen", "take pot")
    assert remaining_gold_suffix(gold, ()) == gold
    assert remaining_gold_suffix(gold, ("look around", "open door")) == (
        "go kitchen",
        "take pot",
    )
    assert remaining_gold_suffix(gold, ("go kitchen",)) == gold
