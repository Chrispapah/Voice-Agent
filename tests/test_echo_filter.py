from ai_sdr_agent.voice.echo_filter import RealtimeEchoGuard, normalize_for_echo_match


def test_normalize_for_echo_match_removes_punctuation_and_collapses_spaces():
    assert normalize_for_echo_match("  Hello,   HOW are you? ") == "hello how are you"


def test_realtime_echo_guard_detects_recent_partial_echo_by_similarity():
    guard = RealtimeEchoGuard()
    guard.record_agent_speech("Γεια σας, πώς μπορώ να σας βοηθήσω σήμερα;", now=10.0)

    match = guard.check("πώς μπορώ να σας βοηθήσω", now=11.0)

    assert match is not None
    assert match.reason == "similarity"


def test_realtime_echo_guard_detects_fuzzy_echo():
    guard = RealtimeEchoGuard()
    guard.record_agent_speech("Would you like to schedule a call tomorrow morning?", now=10.0)

    match = guard.check("would you like schedule call tomorrow morning", now=11.0)

    assert match is not None
    assert match.reason == "similarity"


def test_realtime_echo_guard_expires_old_agent_speech():
    guard = RealtimeEchoGuard(min_window_s=1.0, max_window_s=1.0, hangover_s=0.0)
    guard.record_agent_speech("Would you like to schedule a call?", now=10.0)

    assert guard.check("would you like to schedule a call", now=12.0) is None


def test_realtime_echo_guard_does_not_drop_short_common_replies():
    guard = RealtimeEchoGuard()
    guard.record_agent_speech("Yes, I can help with that.", now=10.0)

    assert guard.check("yes", now=10.5) is None
