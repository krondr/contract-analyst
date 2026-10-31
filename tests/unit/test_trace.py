from contract_analyst.agent.trace import TraceLogger, read_trace


def test_repeated_thought_is_echoed_once_but_logged_every_time(tmp_path, capsys):
    trace = TraceLogger(tmp_path, "r1", console=True)
    for tool in ("build_chains", "consolidate"):
        trace.event("thought", "tools", tool=tool, thought="Собираю цепочки.", thought_source="reasoning")
    err = capsys.readouterr().err
    assert err.count("Собираю цепочки.") == 1 and "→ consolidate" in err
    assert [e["thought"] for e in read_trace(trace.path)] == ["Собираю цепочки."] * 2
