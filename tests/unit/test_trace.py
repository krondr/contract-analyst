from contract_analyst.agent.trace import TraceLogger, read_trace, thought_coverage


def test_thought_coverage_is_the_share_of_tool_calls_with_a_thought():
    events = [{"event": "thought", "thought_source": s} for s in ("argument", "missing", "reasoning", "message")]
    assert thought_coverage(events + [{"event": "tool_call", "thought_source": "missing"}]) == 0.75
    assert thought_coverage([{"event": "plan"}]) is None


def test_repeated_thought_is_echoed_once_but_logged_every_time(tmp_path, capsys):
    trace = TraceLogger(tmp_path, "r1", console=True)
    for tool in ("build_chains", "consolidate"):
        trace.event("thought", "tools", tool=tool, thought="Собираю цепочки.", thought_source="reasoning")
    err = capsys.readouterr().err
    assert err.count("Собираю цепочки.") == 1 and "→ consolidate" in err
    assert [e["thought"] for e in read_trace(trace.path)] == ["Собираю цепочки."] * 2
